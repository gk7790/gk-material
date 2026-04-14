"""
图片感知层修改 Demo —— Embedding 对抗裂变
=========================================

功能：
  1. 像素级高斯噪声注入（可控强度，人眼不可感知）
  2. 颜色通道微偏移（RGB 通道独立偏移 ±1~3）
  3. JPEG 重编码（质量微调 95→93）
  4. EXIF 元数据注入随机标记
  5. ★ Embedding 对抗扰动（PGD 攻击，基于预训练 ResNet50，在特征空间产生显著偏移）
"""
import random
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from src.utils.img_utils import DEFAULT_CONFIG, build_image_candidate
from src.core.config import config
from src.utils.sse_writer import sse
from src.utils.sse_writer import Progress
from src.core.task_manage import TaskExecutor
from src.utils.dir_utils import ensure_directory, build_output_path, build_url_with_base
from src.utils.img_utils import (
    type_file,
    image_dhash,
    hamming_distance,
    compute_image_md5,
    image_mad)
from PIL import Image, ImageOps, UnidentifiedImageError
import asyncio


METRICS_FILENAME = "variant_metrics.json"

def reader_file_json(uid: str) -> list[dict[str, Any]]:
    variants_json = config.FILE_ADDR_PATH / "output" / uid / "variants"
    file_path = variants_json / METRICS_FILENAME

    if not file_path.exists():
        return []

    results = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))

    return results

def append_metrics_record(output_dir: Path, record: dict[str, Any]) -> None:
    """
    把一条记录追加写入到一个日志文件（JSON Lines 格式）中
    :param output_dir:
    :param record:
    :return:
    """
    manifest_path = output_dir / METRICS_FILENAME
    ensure_directory(manifest_path.parent)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_image_variant(image: Image.Image, output_path: Path, cfg: dict[str, Any], rng: random.Random) -> None:
    jpeg_cfg = cfg.get("jpeg_quality", {})
    quality_min = int(jpeg_cfg.get("min", 90))
    quality_max = int(jpeg_cfg.get("max", quality_min))
    quality = rng.randint(min(quality_min, quality_max), max(quality_min, quality_max)) if jpeg_cfg.get("enabled",
                                                                                                        False) else quality_max
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="JPEG", quality=quality, optimize=True)


def process_image_video(path_name_list: list[tuple[Path, str]], uid: str, seed: int | None):
    date_str = datetime.now().strftime("%Y-%m-%d")
    variants_dir = config.FILE_ADDR_PATH / "output" / uid / "variants"
    variants_date_str = variants_dir / date_str
    sse.message(f"user:{uid}", "准备处理素材变体...")
    # 确保路径存在
    ensure_directory(variants_dir)
    ex = TaskExecutor(max_workers=10, retries=2)
    all_results: list[dict[str, Any]] = []
    progress = Progress(f"user:{uid}", len(path_name_list))

    for input_path, file_name in path_name_list:
        rng = random.Random(seed)
        ext = input_path.suffix.lower()
        file_type = type_file(ext)
        if file_type == "image":
            ex.submit(process_images, file_name, input_path, None, variants_date_str, DEFAULT_CONFIG, rng, uid, progress)
    ex.run()


def process_images(file_name: str, file_path: Path, logo_path: Path | None, output_dir: Path, cfg: dict[str, Any],
                   rng: random.Random, uid: str, progress: Progress) -> dict[str, Any]:
    variant_count = cfg["variant_count"]
    similarity_cfg = cfg.get("similarity", {})
    min_distance = int(similarity_cfg.get("min_dhash_distance", 0))
    use_similarity = bool(similarity_cfg.get("enabled", False))
    max_attempts = int(cfg.get("max_attempts_per_variant", 1))
    transforms = cfg.get("transforms", {})
    try:
        with Image.open(file_path) as src:
            source_image = ImageOps.exif_transpose(src).convert("RGBA").convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        return {"source": str(file_path), "output": str(output_dir.name), "error": str(exc)}

    try:
        source_md5 = image_dhash(source_image)
        accepted_hashes = [image_dhash(source_image)]

        item_dict: dict[str, Any] = {}

        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for variant_index in range(1, variant_count + 1):
            best_image: Image.Image | None = None
            best_distance = -1
            for attempt in range(1, max_attempts + 1):
                candidate = build_image_candidate(source_image, logo_path, cfg, rng)
                candidate_hash = image_dhash(candidate)
                distance = min(hamming_distance(candidate_hash, existing) for existing in accepted_hashes)

                # ✅ 记录最优候选（fallback 用）
                if distance > best_distance:
                    best_image = candidate
                    best_distance = distance

                # ❌ 不满足相似度要求 → 继续尝试
                if use_similarity and distance < min_distance:
                    continue

                # =========================
                # 成功生成
                # =========================
                output_path = build_output_path(file_path, output_dir, "image_variant", variant_index, ".jpg")
                save_image_variant(candidate, output_path, transforms, rng)
                output_md5 = compute_image_md5(output_path)

                file_path_host = build_url_with_base(file_path, config.FILE_ADDR_PATH, "/")
                output_path_host = build_url_with_base(output_path, config.FILE_ADDR_PATH, "/")

                # ✅ 每次新建 dict（避免引用问题）
                item_dict = {
                    "model": "image",
                    "date": date_str,
                    "original": file_name,
                    "source": file_path_host,
                    "output": output_path_host,
                    "dhash_distance": distance,
                    "mad": round(image_mad(source_image, candidate), 4),  # ✅ 修复 tuple bug
                    "md5_changed": output_md5 != source_md5,
                    "attempt": attempt,
                    "variant_index": variant_index,
                }
                accepted_hashes.append(candidate_hash)
                break
            else:
                # =========================
                # fallback（全部尝试失败）
                # =========================
                if best_image is not None:
                    output_path = build_output_path(file_path, output_dir, "image_variant", variant_index, ".jpg")
                    save_image_variant(best_image, output_path, transforms, rng)
                    file_path_host = build_url_with_base(file_path, config.FILE_ADDR_PATH, "/")
                    output_path_host = build_url_with_base(output_path, config.FILE_ADDR_PATH, "/")
                    item_dict = {
                        "model": "image",
                        "date": date_str,
                        "original": file_name,
                        "source": file_path_host,
                        "output": output_path_host,
                        "dhash_distance": best_distance,
                        "note": "未达阈值，保留最优",
                        "variant_index": variant_index,
                    }
            channel, step, total = progress.step(1.0 / variant_count)
            sse.progress(channel, step, total, f"正在处理{file_name}")

    except (Exception) as exc:
        sse.error(progress.channel(), str(exc))
        return {"source": str(file_path), "output": str(output_dir.name), "error": str(exc)}
    sse.info(channel, item_dict)
    return item_dict

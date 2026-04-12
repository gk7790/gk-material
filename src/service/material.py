"""
素材裂变核心处理模块

从 CLI 脚本提取的纯处理逻辑，不包含命令行参数解析。
提供图片/视频变体生成能力，供 FastAPI 路由调用。

依赖外部工具：ffmpeg, ffprobe（视频处理需要）
依赖 Python 包：Pillow, numpy
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

# ============================================================
# 常量
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
PREVIEW_SIZE = (32, 32)
LOGO_KEYWORDS = ("logo", "水印", "标志")
METRICS_FILENAME = "variant_metrics.jsonl"

RESAMPLING = getattr(Image, "Resampling", Image)


# ============================================================
# 默认配置
# ============================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "variant_count": 3,
    "image": {
        "max_attempts_per_variant": 6,
        "similarity": {
            "enabled": True,
            "min_dhash_distance": 7,
        },
        "transforms": {
            "crop_scale": {"enabled": True, "max_crop_ratio": 0.05},
            "rotation": {"enabled": True, "max_degrees": 1.2},
            "color": {"enabled": True, "brightness": 0.06, "contrast": 0.08, "saturation": 0.06},
            "same_family_recolor": {
                "enabled": True,
                "probability": 0.90,
                "min_saturation": 0.16,
                "min_value": 0.12,
                "histogram_bins": 48,
                "hue_window_degrees": 44.0,
                "feather_degrees": 20.0,
                "same_family_shift_degrees": 10.0,
                "value_offset_min": -0.08,
                "value_offset_max": 0.18,
            },
            "gamma": {"enabled": True, "min": 0.95, "max": 1.05},
            "noise": {"enabled": True, "pixels_min": 80, "pixels_max": 180, "range": 12},
            "blur": {"enabled": True, "probability": 0.30, "radius_max": 0.8},
            "sharpen": {"enabled": True, "probability": 0.35, "factor_min": 1.1, "factor_max": 1.8},
            "logo": {
                "enabled": True,
                "scale_min": 0.10,
                "scale_max": 0.18,
                "opacity_min": 0.72,
                "opacity_max": 0.95,
                "margin_ratio": 0.04,
                "rotate_max_degrees": 3.0,
            },
            "jpeg_quality": {"enabled": True, "min": 88, "max": 96},
        },
    },
    "video": {
        "max_attempts_per_variant": 4,
        "similarity": {
            "enabled": True,
            "min_average_distance": 4.0,
            "sample_positions": [0.12, 0.50, 0.88],
            "dhash_hash_size": 16,
        },
        "transforms": {
            "crop_scale": {"enabled": True, "max_crop_ratio": 0.04},
            "rotation": {"enabled": True, "probability": 0.65, "max_degrees": 0.9},
            "eq": {"enabled": True, "brightness": 0.03, "contrast": 0.06, "saturation": 0.08, "gamma": 0.06},
            "same_family_recolor": {
                "enabled": True,
                "probability": 0.90,
                "sample_position": 0.35,
                "min_saturation": 0.16,
                "min_value": 0.12,
                "histogram_bins": 48,
                "hue_window_degrees": 44.0,
                "feather_degrees": 24.0,
                "same_family_shift_degrees": 10.0,
                "value_offset_min": -0.08,
                "value_offset_max": 0.18,
            },
            "noise": {"enabled": True, "strength_min": 2, "strength_max": 5},
            "blur": {"enabled": True, "probability": 0.25, "sigma_max": 0.75},
            "border": {
                "enabled": True,
                "probability": 0.70,
                "thickness_min_ratio": 0.008,
                "thickness_max_ratio": 0.018,
                "opacity_min": 0.06,
                "opacity_max": 0.14,
            },
            "patch": {
                "enabled": True,
                "probability": 0.75,
                "width_min_ratio": 0.12,
                "width_max_ratio": 0.24,
                "height_min_ratio": 0.10,
                "height_max_ratio": 0.22,
                "opacity_min": 0.05,
                "opacity_max": 0.12,
            },
            "logo": {
                "enabled": True,
                "scale_min": 0.10,
                "scale_max": 0.18,
                "opacity_min": 0.72,
                "opacity_max": 0.92,
                "margin_ratio": 0.04,
            },
            "encoder": {"preset": "medium", "crf": 18},
        },
    },
}


# ============================================================
# 配置工具
# ============================================================

def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(override)


# 加载配置
def load_config(path: Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return deep_merge(config, loaded)


# ============================================================
# 通用工具
# ============================================================

# 确保目录存在
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_metrics_record(output_dir: Path, record: dict[str, Any]) -> None:
    manifest_path = output_dir / METRICS_FILENAME
    ensure_directory(manifest_path.parent)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_output_path(source_path: Path, output_dir: Path, label: str, index: int, suffix: str) -> Path:
    unique_marker = uuid.uuid4().hex[:6]
    return output_dir / f"{source_path.stem}_{label}_{index:02d}_{unique_marker}{suffix}"


def random_delta(rng: random.Random, magnitude: float) -> float:
    return rng.uniform(-magnitude, magnitude)


def safe_even(value: int, minimum: int = 2) -> int:
    value = max(minimum, int(value))
    if value % 2 == 1:
        value -= 1
    return max(minimum, value)


def collect_files(input_path: Path, extensions: set[str]) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        return []
    if input_path.is_dir():
        return sorted(
            [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in extensions],
            key=lambda item: item.name.lower(),
        )
    return []


def run_command(command: list[str], binary_output: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=not binary_output)


def ensure_tool_exists(tool_name: str) -> None:
    import shutil
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"未找到依赖工具：{tool_name}")


def is_logo_file(path: Path) -> bool:
    lowered_name = path.stem.lower()
    return any(keyword in lowered_name for keyword in LOGO_KEYWORDS)


# ============================================================
# 哈希 / 相似度
# ============================================================

def image_dhash(image: Image.Image, hash_size: int = 8) -> int:
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), RESAMPLING.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.uint8)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def image_preview(image: Image.Image) -> np.ndarray:
    preview = image.convert("RGB").resize(PREVIEW_SIZE, RESAMPLING.BILINEAR)
    return np.asarray(preview, dtype=np.float32)


def image_mad(left: Image.Image, right: Image.Image) -> float:
    if left.size != right.size:
        right = right.resize(left.size, RESAMPLING.BILINEAR)
    left_pixels = np.asarray(left.convert("RGB"), dtype=np.float32)
    right_pixels = np.asarray(right.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(left_pixels - right_pixels)))


# ============================================================
# HSV 颜色转换 & 同色系变调
# ============================================================

def rgb_to_hsv_arrays(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_float = rgb.astype(np.float32) / 255.0
    red, green, blue = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]
    maximum = np.max(rgb_float, axis=2)
    minimum = np.min(rgb_float, axis=2)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)
    mask = delta > 1e-6
    hue[mask & (maximum == red)] = np.mod((green[mask & (maximum == red)] - blue[mask & (maximum == red)]) / delta[mask & (maximum == red)], 6.0)
    hue[mask & (maximum == green)] = ((blue[mask & (maximum == green)] - red[mask & (maximum == green)]) / delta[mask & (maximum == green)]) + 2.0
    hue[mask & (maximum == blue)] = ((red[mask & (maximum == blue)] - green[mask & (maximum == blue)]) / delta[mask & (maximum == blue)]) + 4.0
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maximum)
    nonzero_value = maximum > 1e-6
    saturation[nonzero_value] = delta[nonzero_value] / maximum[nonzero_value]
    return hue, saturation, maximum


def hsv_to_rgb_array(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    hue = np.mod(hue, 1.0)
    saturation = np.clip(saturation, 0.0, 1.0)
    value = np.clip(value, 0.0, 1.0)
    hue_scaled = hue * 6.0
    sector = np.floor(hue_scaled).astype(np.int32) % 6
    fraction = hue_scaled - np.floor(hue_scaled)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    red = np.zeros_like(value)
    green = np.zeros_like(value)
    blue = np.zeros_like(value)
    for idx, (rv, gv, bv) in enumerate([(value, t, p), (q, value, p), (p, value, t), (p, q, value), (t, p, value), (value, p, q)]):
        m = sector == idx
        red[m], green[m], blue[m] = rv[m], gv[m], bv[m]
    return np.clip(np.round(np.stack([red, green, blue], axis=2) * 255.0), 0.0, 255.0).astype(np.uint8)


def circular_hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    distance = np.abs(hue - center)
    return np.minimum(distance, 1.0 - distance)


def detect_dominant_hue(rgb: np.ndarray, cfg: dict[str, Any]) -> float | None:
    min_saturation = float(cfg.get("min_saturation", 0.16))
    min_value = float(cfg.get("min_value", 0.12))
    histogram_bins = max(12, int(cfg.get("histogram_bins", 48)))
    hue, saturation, value = rgb_to_hsv_arrays(rgb)
    mask = (saturation >= min_saturation) & (value >= min_value)
    if int(np.count_nonzero(mask)) < max(64, rgb.shape[0] * rgb.shape[1] // 100):
        return None
    weights = (0.35 + 0.65 * value[mask]) * saturation[mask]
    histogram, edges = np.histogram(hue[mask], bins=histogram_bins, range=(0.0, 1.0), weights=weights)
    if histogram.size == 0 or float(histogram.max()) <= 0.0:
        return None
    top_index = int(np.argmax(histogram))
    return float((edges[top_index] + edges[top_index + 1]) / 2.0)


def build_same_family_recolor_spec(rgb: np.ndarray, cfg: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    if not cfg.get("enabled", False):
        return None
    if rng.random() > float(cfg.get("probability", 1.0)):
        return None
    dominant_hue = detect_dominant_hue(rgb, cfg)
    if dominant_hue is None:
        return None

    max_shift = float(cfg.get("same_family_shift_degrees", 10.0)) / 360.0
    window = float(cfg.get("hue_window_degrees", 44.0)) / 360.0
    feather = float(cfg.get("feather_degrees", 20.0)) / 360.0
    value_offset_min = float(cfg.get("value_offset_min", -0.08))
    value_offset_max = float(cfg.get("value_offset_max", 0.18))

    styles = ("lighter", "muted", "deeper", "richer", "fresh")
    style = rng.choice(styles)
    params = {
        "lighter": (0.72, 0.95, 1.04, 1.16, 0.02, value_offset_max, -max_shift * 0.4, max_shift * 0.4),
        "muted":   (0.55, 0.82, 1.01, 1.10, 0.02, min(0.12, value_offset_max * 0.7), -max_shift * 0.5, max_shift * 0.5),
        "deeper":  (1.04, 1.22, 0.82, 0.95, min(-0.10, value_offset_min), -0.02, -max_shift * 0.35, max_shift * 0.35),
        "richer":  (1.08, 1.28, 0.94, 1.05, max(-0.04, value_offset_min), min(0.06, value_offset_max), -max_shift * 0.65, max_shift * 0.65),
        "fresh":   (0.88, 1.08, 1.02, 1.14, 0.01, max(0.10, value_offset_max * 0.8), max_shift * 0.15, max_shift),
    }
    s_min, s_max, v_min, v_max, vo_min, vo_max, hs_min, hs_max = params[style]

    return {
        "style": style, "center_hue": dominant_hue,
        "hue_shift": rng.uniform(hs_min, hs_max),
        "hue_window": max(window, 1.0 / 360.0),
        "feather": max(feather, 1.0 / 360.0),
        "min_saturation": float(cfg.get("min_saturation", 0.16)),
        "min_value": float(cfg.get("min_value", 0.12)),
        "saturation_scale": rng.uniform(s_min, s_max),
        "value_scale": rng.uniform(v_min, v_max),
        "value_offset": rng.uniform(max(0.02, vo_min), vo_max),
    }


def apply_same_family_recolor_to_rgb(rgb: np.ndarray, spec: dict[str, Any] | None) -> np.ndarray:
    if spec is None:
        return rgb
    hue, saturation, value = rgb_to_hsv_arrays(rgb)
    distance = circular_hue_distance(hue, float(spec["center_hue"]))
    half_window = float(spec["hue_window"]) / 2.0
    feather = float(spec["feather"])

    influence = np.zeros_like(hue, dtype=np.float32)
    inner_mask = distance <= half_window
    influence[inner_mask] = 1.0
    outer_mask = (distance > half_window) & (distance <= (half_window + feather))
    if feather > 1e-6:
        influence[outer_mask] = 1.0 - ((distance[outer_mask] - half_window) / feather)
    valid_mask = (saturation >= float(spec["min_saturation"])) & (value >= float(spec["min_value"]))
    influence *= valid_mask.astype(np.float32)
    influence *= np.clip((saturation - float(spec["min_saturation"])) / max(1e-6, 1.0 - float(spec["min_saturation"])), 0.0, 1.0)

    if float(np.max(influence)) <= 0.0:
        return rgb
    hue = np.mod(hue + float(spec["hue_shift"]) * influence, 1.0)
    saturation = np.clip(saturation * (1.0 + (float(spec["saturation_scale"]) - 1.0) * influence), 0.0, 1.0)
    value = np.clip(value * (1.0 + (float(spec["value_scale"]) - 1.0) * influence) + float(spec["value_offset"]) * influence, 0.0, 1.0)
    return hsv_to_rgb_array(hue, saturation, value)


def apply_same_family_recolor_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    if not cfg.get("enabled", False):
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    spec = build_same_family_recolor_spec(rgb, cfg, rng)
    if spec is None:
        return image
    return Image.fromarray(apply_same_family_recolor_to_rgb(rgb, spec), mode="RGB")


# ============================================================
# 图片变换
# ============================================================

def border_fill_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.uint8)
    border = np.concatenate([pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]], axis=0)
    averages = border.mean(axis=0)
    return tuple(int(c) for c in averages[:3])


def apply_crop_scale_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    if not cfg.get("enabled", False):
        return image
    width, height = image.size
    max_crop_ratio = float(cfg.get("max_crop_ratio", 0.0))
    if max_crop_ratio <= 0.0:
        return image
    left = int(rng.uniform(0.0, max_crop_ratio) * width)
    right = int(rng.uniform(0.0, max_crop_ratio) * width)
    top = int(rng.uniform(0.0, max_crop_ratio) * height)
    bottom = int(rng.uniform(0.0, max_crop_ratio) * height)
    cropped = image.crop((left, top, max(2, width - left - right), max(2, height - top - bottom)))
    return cropped.resize((width, height), RESAMPLING.LANCZOS)


def apply_rotation_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    if not cfg.get("enabled", False):
        return image
    max_degrees = float(cfg.get("max_degrees", 0.0))
    if max_degrees <= 0.0:
        return image
    angle = rng.uniform(-max_degrees, max_degrees)
    if abs(angle) < 0.05:
        return image
    return image.rotate(angle, resample=RESAMPLING.BICUBIC, fillcolor=border_fill_color(image))


def apply_color_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    if not cfg.get("enabled", False):
        return image
    for attr, enhancer_cls in [("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast), ("saturation", ImageEnhance.Color)]:
        magnitude = float(cfg.get(attr, 0.0))
        if magnitude > 0.0:
            image = enhancer_cls(image).enhance(1.0 + random_delta(rng, magnitude))
    return image


def apply_gamma_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    if not cfg.get("enabled", False):
        return image
    minimum, maximum = float(cfg.get("min", 1.0)), float(cfg.get("max", 1.0))
    if minimum <= 0.0 or maximum <= 0.0:
        return image
    gamma = rng.uniform(minimum, maximum)
    if abs(gamma - 1.0) < 0.01:
        return image
    inverse_gamma = 1.0 / gamma
    lut = [int(round(((i / 255.0) ** inverse_gamma) * 255.0)) for i in range(256)]
    return image.point(lut * 3)


def apply_noise_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    if not cfg.get("enabled", False):
        return image
    width, height = image.size
    pixels_min, pixels_max, noise_range = int(cfg.get("pixels_min", 0)), int(cfg.get("pixels_max", 0)), int(cfg.get("range", 0))
    if pixels_max <= 0 or noise_range <= 0:
        return image
    count = rng.randint(max(0, pixels_min), max(pixels_min, pixels_max))
    result = image.copy()
    pixels = result.load()
    for _ in range(count):
        x, y = rng.randint(0, width - 1), rng.randint(0, height - 1)
        r, g, b = pixels[x, y]
        pixels[x, y] = (max(0, min(255, r + rng.randint(-noise_range, noise_range))),
                        max(0, min(255, g + rng.randint(-noise_range, noise_range))),
                        max(0, min(255, b + rng.randint(-noise_range, noise_range))))
    return result


def apply_blur_sharpen_image(image: Image.Image, transforms: dict[str, Any], rng: random.Random) -> Image.Image:
    blur_cfg = transforms.get("blur", {})
    if blur_cfg.get("enabled", False) and rng.random() < float(blur_cfg.get("probability", 0.0)):
        sigma_max = float(blur_cfg.get("radius_max", 0.0))
        if sigma_max > 0.0:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, sigma_max)))
    sharpen_cfg = transforms.get("sharpen", {})
    if sharpen_cfg.get("enabled", False) and rng.random() < float(sharpen_cfg.get("probability", 0.0)):
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(float(sharpen_cfg.get("factor_min", 1.0)), float(sharpen_cfg.get("factor_max", 1.0))))
    return image


def apply_logo_image(image: Image.Image, logo_path: Path | None, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    if logo_path is None or not cfg.get("enabled", False):
        return image
    width, height = image.size
    if width < 2 or height < 2:
        return image
    with Image.open(logo_path) as src_logo:
        logo = src_logo.convert("RGBA")
    scale_min, scale_max = float(cfg.get("scale_min", 0.1)), float(cfg.get("scale_max", scale_min))
    opacity = rng.uniform(float(cfg.get("opacity_min", 0.8)), float(cfg.get("opacity_max", opacity_min)))
    margin_ratio = float(cfg.get("margin_ratio", 0.04))
    rotate_max = float(cfg.get("rotate_max_degrees", 0.0))

    target_width = min(max(1, int(width * rng.uniform(scale_min, scale_max))), max(1, width - 2))
    resize_ratio = target_width / max(1, logo.width)
    target_height = max(1, int(logo.height * resize_ratio))
    logo = logo.resize((target_width, target_height), RESAMPLING.LANCZOS)

    if rotate_max > 0.0:
        logo = logo.rotate(rng.uniform(-rotate_max, rotate_max), resample=RESAMPLING.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))

    alpha = logo.getchannel("A").point(lambda v: int(v * opacity))
    logo.putalpha(alpha)
    lw, lh = logo.size
    mx, my = int(width * margin_ratio), int(height * margin_ratio)
    x = rng.randint(mx, max(mx, width - lw - mx))
    y = rng.randint(my, max(my, height - lh - my))

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste(logo, (x, y), logo)
    return Image.alpha_composite(base, overlay).convert("RGB")


def build_image_candidate(source_image: Image.Image, logo_path: Path | None, config: dict[str, Any], rng: random.Random) -> Image.Image:
    t = config["transforms"]
    image = source_image.copy()
    image = apply_crop_scale_image(image, t.get("crop_scale", {}), rng)
    image = apply_rotation_image(image, t.get("rotation", {}), rng)
    image = apply_color_image(image, t.get("color", {}), rng)
    image = apply_same_family_recolor_image(image, t.get("same_family_recolor", {}), rng)
    image = apply_gamma_image(image, t.get("gamma", {}), rng)
    image = apply_noise_image(image, t.get("noise", {}), rng)
    image = apply_blur_sharpen_image(image, t, rng)
    image = apply_logo_image(image, logo_path, t.get("logo", {}), rng)
    return image


def save_image_variant(image: Image.Image, output_path: Path, cfg: dict[str, Any], rng: random.Random) -> None:
    jpeg_cfg = cfg.get("jpeg_quality", {})
    quality_min = int(jpeg_cfg.get("min", 90))
    quality_max = int(jpeg_cfg.get("max", quality_min))
    quality = rng.randint(min(quality_min, quality_max), max(quality_min, quality_max)) if jpeg_cfg.get("enabled", False) else quality_max
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="JPEG", quality=quality, optimize=True)


# ============================================================
# 图片处理入口
# ============================================================

def process_images(
    inputs: list[Path],
    output_dir: Path,
    logo_path: Path | None,
    config: dict[str, Any],
    variant_count: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """处理图片列表，返回每个变体的信息。"""
    if rng is None:
        rng = random.Random()

    image_cfg = config["image"]
    similarity_cfg = image_cfg.get("similarity", {})
    min_distance = int(similarity_cfg.get("min_dhash_distance", 0))
    use_similarity = bool(similarity_cfg.get("enabled", False))
    max_attempts = int(image_cfg.get("max_attempts_per_variant", 1))
    all_results: list[dict[str, Any]] = []

    for input_path in inputs:
        try:
            with Image.open(input_path) as src:
                source_image = ImageOps.exif_transpose(src).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            all_results.append({"source": str(input_path), "error": str(exc)})
            continue

        accepted_hashes = [image_dhash(source_image)]

        for variant_index in range(1, variant_count + 1):
            best_image: Image.Image | None = None
            best_distance = -1

            for attempt in range(1, max_attempts + 1):
                candidate = build_image_candidate(source_image, logo_path, image_cfg, rng)
                candidate_hash = image_dhash(candidate)
                distance = min(hamming_distance(candidate_hash, existing) for existing in accepted_hashes)

                if distance > best_distance:
                    best_image = candidate
                    best_distance = distance

                if use_similarity and distance < min_distance:
                    continue

                output_path = build_output_path(input_path, output_dir, "image_variant", variant_index, ".jpg")
                save_image_variant(candidate, output_path, image_cfg["transforms"], rng)
                output_md5 = compute_file_md5(output_path)
                metrics = {
                    "source": str(input_path), "output": str(output_path.name),
                    "dhash_distance": distance, "mad": round(image_mad(source_image, candidate), 4),
                    "md5_changed": output_md5 != compute_file_md5(input_path),
                }
                all_results.append(metrics)
                accepted_hashes.append(candidate_hash)
                break
            else:
                if best_image is not None:
                    output_path = build_output_path(input_path, output_dir, "image_variant", variant_index, ".jpg")
                    save_image_variant(best_image, output_path, image_cfg["transforms"], rng)
                    all_results.append({
                        "source": str(input_path), "output": str(output_path.name),
                        "dhash_distance": best_distance, "note": "未达阈值，保留最优",
                    })

    return all_results


# ============================================================
# 视频处理
# ============================================================

def get_video_info(input_path: Path) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
               "-of", "json", str(input_path)]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe 读取视频信息失败。")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("未检测到视频流。")
    stream = streams[0]
    frame_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "25/1"
    fps = float(Fraction(frame_rate))
    if fps <= 0.0:
        fps = 25.0
    duration = 0.0
    try:
        duration = max(0.0, float(payload.get("format", {}).get("duration", "0")))
    except (TypeError, ValueError):
        pass
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": fps, "duration": duration}


def extract_video_frame(video_path: Path, timestamp: float) -> Image.Image:
    command = ["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(video_path),
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    result = run_command(command, binary_output=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore").strip() or "抽帧失败。")
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def video_signature(video_path: Path, duration: float, sample_positions: list[float], hash_size: int = 8) -> tuple[tuple[int, np.ndarray], ...]:
    if duration <= 0.0:
        frame = extract_video_frame(video_path, 0.0)
        return ((image_dhash(frame, hash_size=hash_size), image_preview(frame)),)
    return tuple(
        (image_dhash(f := extract_video_frame(video_path, min(duration * min(0.98, max(0.0, pos)), max(0.0, duration - 0.05)), hash_size=hash_size), image_preview(f))
        ) for pos in sample_positions)


def average_video_distance(
    left: tuple[tuple[int, np.ndarray], ...],
    right: tuple[tuple[int, np.ndarray], ...],
) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return float(sum(
        hamming_distance(left[i][0], right[i][0]) + float(np.mean(np.abs(left[i][1] - right[i][1])))
        for i in range(size)
    ) / size)


def build_video_filters(width: int, height: int, transforms: dict[str, Any], rng: random.Random) -> list[str]:
    filters: list[str] = []

    # crop
    crop_cfg = transforms.get("crop_scale", {})
    if crop_cfg.get("enabled", False):
        max_crop_ratio = float(crop_cfg.get("max_crop_ratio", 0.0))
        if max_crop_ratio > 0.0:
            l = int(rng.uniform(0.0, max_crop_ratio) * width)
            r = int(rng.uniform(0.0, max_crop_ratio) * width)
            t = int(rng.uniform(0.0, max_crop_ratio) * height)
            b = int(rng.uniform(0.0, max_crop_ratio) * height)
            cw, ch = safe_even(width - l - r), safe_even(height - t - b)
            x, y = max(0, min(l, width - cw)), max(0, min(t, height - ch))
            filters.extend([f"crop={cw}:{ch}:{x}:{y}", f"scale={width}:{height}"])

    # rotation
    rot_cfg = transforms.get("rotation", {})
    if rot_cfg.get("enabled", False) and rng.random() < float(rot_cfg.get("probability", 0.0)):
        max_deg = float(rot_cfg.get("max_degrees", 0.0))
        if max_deg > 0.0:
            angle = rng.uniform(-max_deg, max_deg)
            if abs(angle) >= 0.05:
                filters.append(f"rotate={math.radians(angle):.7f}:ow=iw:oh=ih:c=black@0")

    # eq
    eq_cfg = transforms.get("eq", {})
    if eq_cfg.get("enabled", False):
        br = random_delta(rng, float(eq_cfg.get("brightness", 0.0)))
        ct = 1.0 + random_delta(rng, float(eq_cfg.get("contrast", 0.0)))
        sa = 1.0 + random_delta(rng, float(eq_cfg.get("saturation", 0.0)))
        gm = 1.0 + random_delta(rng, float(eq_cfg.get("gamma", 0.0)))
        filters.append(f"eq=brightness={br:.4f}:contrast={ct:.4f}:saturation={sa:.4f}:gamma={gm:.4f}")

    # noise
    noise_cfg = transforms.get("noise", {})
    if noise_cfg.get("enabled", False):
        s_min, s_max = int(noise_cfg.get("strength_min", 0)), int(noise_cfg.get("strength_max", 0))
        if s_max > 0:
            strength = rng.randint(min(s_min, s_max), max(s_min, s_max))
            if strength > 0:
                filters.append(f"noise=alls={strength}:allf=t")

    # blur
    blur_cfg = transforms.get("blur", {})
    if blur_cfg.get("enabled", False) and rng.random() < float(blur_cfg.get("probability", 0.0)):
        sigma_max = float(blur_cfg.get("sigma_max", 0.0))
        if sigma_max > 0.0:
            filters.append(f"gblur=sigma={rng.uniform(0.15, sigma_max):.3f}")

    # border
    border_cfg = transforms.get("border", {})
    if border_cfg.get("enabled", False) and rng.random() < float(border_cfg.get("probability", 0.0)):
        min_r, max_r = float(border_cfg.get("thickness_min_ratio", 0.0)), float(border_cfg.get("thickness_max_ratio", min_r))
        opa = rng.uniform(float(border_cfg.get("opacity_min", 0.06)), float(border_cfg.get("opacity_max", 0.14)))
        bt = max(1, int(min(width, height) * rng.uniform(min_r, max_r)))
        color = f"{'black' if rng.random() < 0.5 else 'white'}@{opa:.3f}"
        filters.extend([
            f"drawbox=x=0:y=0:w=iw:h={bt}:color={color}:t=fill",
            f"drawbox=x=0:y=ih-{bt}:w=iw:h={bt}:color={color}:t=fill",
            f"drawbox=x=0:y=0:w={bt}:h=ih:color={color}:t=fill",
            f"drawbox=x=iw-{bt}:y=0:w={bt}:h=ih:color={color}:t=fill",
        ])

    # patch
    patch_cfg = transforms.get("patch", {})
    if patch_cfg.get("enabled", False) and rng.random() < float(patch_cfg.get("probability", 0.0)):
        pw = min(width, max(8, int(width * rng.uniform(float(patch_cfg.get("width_min_ratio", 0.12)), float(patch_cfg.get("width_max_ratio", 0.24))))))
        ph = min(height, max(8, int(height * rng.uniform(float(patch_cfg.get("height_min_ratio", 0.10)), float(patch_cfg.get("height_max_ratio", 0.22))))))
        px, py = rng.randint(0, max(0, width - pw)), rng.randint(0, max(0, height - ph))
        popa = rng.uniform(float(patch_cfg.get("opacity_min", 0.05)), float(patch_cfg.get("opacity_max", 0.12)))
        pcolor = f"{'black' if rng.random() < 0.5 else 'white'}@{popa:.3f}"
        filters.append(f"drawbox=x={px}:y={py}:w={pw}:h={ph}:color={pcolor}:t=fill")

    return filters


def pick_video_logo_spec(width: int, height: int, logo_path: Path | None, cfg: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    if logo_path is None or not cfg.get("enabled", False):
        return None
    with Image.open(logo_path) as src_logo:
        lw, lh = src_logo.size
    if lw <= 0 or lh <= 0:
        return None
    scale_min, scale_max = float(cfg.get("scale_min", 0.1)), float(cfg.get("scale_max", scale_min))
    opacity = rng.uniform(float(cfg.get("opacity_min", 0.8)), float(cfg.get("opacity_max", opacity_min)))
    margin = float(cfg.get("margin_ratio", 0.04))
    tw = safe_even(min(max(2, int(width * rng.uniform(scale_min, scale_max))), max(2, width - 2)))
    th = safe_even(min(max(2, int(lh * tw / lw)), max(2, height - 2)))
    mx, my = int(width * margin), int(height * margin)
    return {"path": logo_path, "width": tw, "height": th, "x": rng.randint(mx, max(mx, width - tw - mx)), "y": rng.randint(my, max(my, height - th - my)), "opacity": opacity}


def build_video_command(input_path: Path, output_path: Path, filters: list[str], logo_spec: dict[str, Any] | None, transforms: dict[str, Any], audio_mode: str) -> list[str]:
    enc = transforms.get("encoder", {})
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(input_path)]
    if logo_spec is not None:
        command.extend(["-i", str(logo_spec["path"])])
        base = ",".join(filters) if filters else "null"
        fc = (f"[0:v]{base}[base];[1:v]scale={logo_spec['width']}:{logo_spec['height']},format=rgba,"
              f"colorchannelmixer=aa={logo_spec['opacity']:.3f}[logo];[base][logo]overlay={logo_spec['x']}:{logo_spec['y']}:format=auto[vout]")
        command.extend(["-filter_complex", fc, "-map", "[vout]"])
    else:
        if filters:
            command.extend(["-vf", ",".join(filters)])
        command.extend(["-map", "0:v:0"])
    command.extend(["-map", "0:a?", "-c:v", "libx264", "-preset", str(enc.get("preset", "medium")),
                     "-crf", str(int(enc.get("crf", 18))), "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    command.extend(["-c:a", "copy"] if audio_mode == "copy" else ["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-shortest", str(output_path)])
    return command


def read_exact(stream: Any, expected_size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < expected_size:
        chunk = stream.read(expected_size - len(buffer))
        if not chunk:
            return None if not buffer else bytes(buffer)
        buffer.extend(chunk)
    return bytes(buffer)


def overlay_logo_on_frame_array(frame: np.ndarray, logo_rgba: np.ndarray, x: int, y: int) -> None:
    lh, lw = logo_rgba.shape[:2]
    fh, fw = frame.shape[:2]
    x_end, y_end = min(fw, x + lw), min(fh, y + lh)
    if x_end <= x or y_end <= y:
        return
    crop = logo_rgba[:y_end - y, :x_end - x]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    frame[y:y_end, x:x_end] = (crop[:, :, :3].astype(np.float32) * alpha + frame[y:y_end, x:x_end].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def render_video_variant_with_recolor(
    input_path: Path, output_path: Path, width: int, height: int, fps: float,
    filters: list[str], recolor_spec: dict[str, Any], logo_spec: dict[str, Any] | None,
    transforms: dict[str, Any], audio_mode: str,
) -> tuple[bool, str]:
    frame_size = width * height * 3
    logo_rgba = None
    if logo_spec is not None:
        with Image.open(logo_spec["path"]) as img:
            logo = img.convert("RGBA").resize((int(logo_spec["width"]), int(logo_spec["height"])), RESAMPLING.LANCZOS)
        alpha = logo.getchannel("A").point(lambda v: int(v * float(logo_spec["opacity"])))
        logo.putalpha(alpha)
        logo_rgba = np.asarray(logo, dtype=np.uint8)

    enc = transforms.get("encoder", {})
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(input_path)] + (["-vf", ",".join(filters)] if filters else []) +
        ["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", f"{fps:.6f}", "-i", "-", "-i", str(input_path), "-map", "0:v:0", "-map", "1:a?",
         "-c:v", "libx264", "-preset", str(enc.get("preset", "medium")), "-crf", str(int(enc.get("crf", 18))),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart"] +
        (["-c:a", "copy"] if audio_mode == "copy" else ["-c:a", "aac", "-b:a", "192k"]) +
        ["-shortest", str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        while True:
            raw = read_exact(decoder.stdout, frame_size)
            if raw is None:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            frame = apply_same_family_recolor_to_rgb(frame, recolor_spec)
            if logo_rgba is not None and logo_spec is not None:
                overlay_logo_on_frame_array(frame, logo_rgba, int(logo_spec["x"]), int(logo_spec["y"]))
            encoder.stdin.write(frame.tobytes())
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        errors = []
        for proc in (decoder, encoder):
            proc.wait()
            errors.append(proc.stderr.read().decode("utf-8", errors="ignore").strip())
        if decoder.returncode != 0:
            return False, errors[0] or "解码失败。"
        if encoder.returncode != 0:
            return False, errors[1] or "编码失败。"
        return True, ""
    finally:
        for stream in (decoder.stdout, decoder.stderr, encoder.stderr):
            if stream:
                stream.close()
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        for proc in (decoder, encoder):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def build_video_recolor_spec(video_path: Path, duration: float, cfg: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    if not cfg.get("enabled", False):
        return None
    try:
        sample_pos = float(cfg.get("sample_position", 0.35))
        ts = min(duration * min(0.98, max(0.0, sample_pos)), max(0.0, duration - 0.05)) if duration > 0 else 0.0
        frame = extract_video_frame(video_path, ts)
        rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        return build_same_family_recolor_spec(rgb, cfg, rng)
    except RuntimeError:
        return None


# ============================================================
# 视频处理入口
# ============================================================

def process_videos(
    inputs: list[Path],
    output_dir: Path,
    logo_path: Path | None,
    config: dict[str, Any],
    variant_count: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """处理视频列表，返回每个变体的信息。"""
    ensure_tool_exists("ffmpeg")
    ensure_tool_exists("ffprobe")

    if rng is None:
        rng = random.Random()

    video_cfg = config["video"]
    similarity_cfg = video_cfg.get("similarity", {})
    sample_positions = [float(p) for p in similarity_cfg.get("sample_positions", [0.5])]
    hash_size = int(similarity_cfg.get("dhash_hash_size", 8))
    min_distance = float(similarity_cfg.get("min_average_distance", 0.0))
    use_similarity = bool(similarity_cfg.get("enabled", False))
    max_attempts = int(video_cfg.get("max_attempts_per_variant", 1))
    transforms = video_cfg["transforms"]
    recolor_cfg = transforms.get("same_family_recolor", {})
    all_results: list[dict[str, Any]] = []

    for input_path in inputs:
        try:
            info = get_video_info(input_path)
            source_sig = video_signature(input_path, info["duration"], sample_positions, hash_size)
        except RuntimeError as exc:
            all_results.append({"source": str(input_path), "error": str(exc)})
            continue

        accepted_signatures = [source_sig]

        for vi in range(1, variant_count + 1):
            best_distance = -1.0
            best_output: Path | None = None
            best_sig = None

            for attempt in range(1, max_attempts + 1):
                filters = build_video_filters(info["width"], info["height"], transforms, rng)
                recolor_spec = build_video_recolor_spec(input_path, info["duration"], recolor_cfg, rng)
                logo_spec = pick_video_logo_spec(info["width"], info["height"], logo_path, transforms.get("logo", {}), rng)
                output_path = build_output_path(input_path, output_dir, "video_variant", vi, ".mp4")

                # 渲染
                success = False
                if recolor_spec is None:
                    for audio_mode in ("copy", "aac"):
                        cmd = build_video_command(input_path, output_path, filters, logo_spec, transforms, audio_mode)
                        result = run_command(cmd)
                        if result.returncode == 0:
                            success = True
                            break
                else:
                    for audio_mode in ("copy", "aac"):
                        ok, err = render_video_variant_with_recolor(
                            input_path, output_path, info["width"], info["height"], info["fps"],
                            filters, recolor_spec, logo_spec, transforms, audio_mode)
                        if ok:
                            success = True
                            break

                if not success:
                    if output_path.exists():
                        output_path.unlink()
                    continue
                if not output_path.exists():
                    continue

                try:
                    cand_sig = video_signature(output_path, info["duration"], sample_positions, hash_size)
                except RuntimeError:
                    if output_path.exists():
                        output_path.unlink()
                    continue

                distance = min(average_video_distance(cand_sig, existing) for existing in accepted_signatures)
                is_best = distance > best_distance
                if is_best:
                    if best_output and best_output.exists() and best_output != output_path:
                        best_output.unlink()
                    best_distance, best_output, best_sig = distance, output_path, cand_sig

                if use_similarity and distance < min_distance:
                    if output_path.exists() and not is_best:
                        output_path.unlink()
                    continue

                # 清理旧的最优
                if best_output and best_output != output_path and best_output.exists():
                    best_output.unlink()
                    best_output, best_sig = output_path, cand_sig

                all_results.append({"source": str(input_path), "output": str(output_path.name), "distance": round(distance, 2)})
                accepted_signatures.append(cand_sig)
                break
            else:
                if best_output and best_output.exists():
                    all_results.append({"source": str(input_path), "output": str(best_output.name), "distance": round(best_distance, 2), "note": "未达阈值，保留最优"})
                    if best_sig:
                        accepted_signatures.append(best_sig)

    return all_results

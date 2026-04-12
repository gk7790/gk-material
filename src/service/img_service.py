"""
图片感知层修改 Demo —— Embedding 对抗裂变
=========================================

功能：
  1. 像素级高斯噪声注入（可控强度，人眼不可感知）
  2. 颜色通道微偏移（RGB 通道独立偏移 ±1~3）
  3. JPEG 重编码（质量微调 95→93）
  4. EXIF 元数据注入随机标记
  5. ★ Embedding 对抗扰动（PGD 攻击，基于预训练 ResNet50，在特征空间产生显著偏移）

验证指标：
  - SSIM（结构相似性）：> 0.99 代表人眼不可感知
  - 感知哈希距离（dHash）：衡量视觉相似度
  - MD5：衡量文件级变化
  - Embedding 余弦距离：衡量深度特征空间偏移

使用方式：
  python demo_image_adversarial.py --input photo.jpg --count 3
  python demo_image_adversarial.py --input photo.jpg --count 5 --eps 8 --noise 2
"""

import hashlib
import io
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms


# ============================================================
#  数据结构
# ============================================================

@dataclass
class VariantResult:
    """单个变体的处理结果。"""
    variant_name: str
    strategy: str
    md5_original: str
    md5_variant: str
    ssim: float
    dhash_distance: int
    embedding_cosine_dist: float | None
    file_size_original: int
    file_size_variant: int
    elapsed_ms: float
    output_path: str

@dataclass
class ProcessReport:
    """整批处理报告。"""
    input_path: str
    md5_original: str
    strategies_applied: list[str]
    variants: list[VariantResult] = field(default_factory=list)


# ============================================================
#  模型：预训练 ResNet50 特征提取器
# ============================================================

class EmbeddingExtractor:
    """
    使用预训练 ResNet50 提取图像 embedding（avgpool 后的 2048 维特征）。
    CPU 模式，无需 GPU。
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()

        # 切掉最后的 FC 层，只保留特征提取
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # 图像预处理（与训练时一致）
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, pil_image: Image.Image) -> torch.Tensor:
        """提取 2048 维 embedding，返回 L2 归一化后的向量。"""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        features = self.feature_extractor(tensor)          # (1, 2048, 1, 1)
        features = features.flatten(1)                     # (1, 2048)
        features = F.normalize(features, p=2, dim=1)      # L2 归一化
        return features.squeeze(0)                         # (2048,)


# ============================================================
#  感知层修改策略
# ============================================================

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB → OpenCV BGR numpy。"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """OpenCV BGR numpy → PIL RGB。"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def add_gaussian_noise(pil_img: Image.Image, sigma: float = 1.0, seed: int | None = None) -> Image.Image:
    """
    像素级高斯噪声注入。
    sigma=1.0 时 SSIM 通常 > 0.998，人眼完全不可感知。
    """
    rng = np.random.RandomState(seed)
    arr = np.array(pil_img, dtype=np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def shift_color_channels(pil_img: Image.Image, max_shift: int = 2, seed: int | None = None) -> Image.Image:
    """
    颜色通道微偏移。对 R/G/B 三个通道独立加一个 [-max_shift, +max_shift] 的随机偏移。
    max_shift=2 时 SSIM 通常 > 0.996。
    """
    rng = np.random.RandomState(seed)
    arr = np.array(pil_img, dtype=np.int16)
    for c in range(3):
        shift = rng.randint(-max_shift, max_shift + 1)
        if shift != 0:
            arr[:, :, c] += shift
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jpeg_reencode(pil_img: Image.Image, quality: int = 93) -> Image.Image:
    """
    JPEG 重编码。将图片以指定质量重新编码为 JPEG 再解码回来。
    改变文件字节但视觉差异极小。
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def inject_exif_metadata(pil_img: Image.Image, variant_id: str) -> Image.Image:
    """
    向 EXIF 注入随机标记，不改变任何像素。
    """
    from PIL.PngImagePlugin import PngInfo

    # 尝试 JPEG EXIF
    if pil_img.format == "JPEG":
        exif = pil_img.getexif()
        exif[0x0131] = f"Variant-{variant_id}"      # Software tag
        exif[0x9286] = f"v{variant_id}"             # UserComment
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", exif=exif)
        buf.seek(0)
        return Image.open(buf).copy()

    # PNG 方案
    pnginfo = PngInfo()
    pnginfo.add_text("Variant", variant_id)
    pnginfo.add_text("Software", f"AdversarialForge/{variant_id}")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", pnginfo=pnginfo)
    buf.seek(0)
    return Image.open(buf).copy()


def generate_adversarial_embedding_perturbation(
    pil_img: Image.Image,
    extractor: EmbeddingExtractor,
    target_distance: float = 0.15,
    eps: float = 8.0,
    step_size: float = 1.0,
    steps: int = 20,
    seed: int | None = None,
) -> Image.Image:
    """
    ★ Embedding 对抗扰动（PGD 攻击）

    在预训练 ResNet50 的 embedding 空间中产生显著偏移，
    直接在原图分辨率上操作像素，避免 resize 导致的质量损失。

    原理：
      1. 用 ResNet50 提取原图 embedding（通过 transform pipeline）
      2. 生成一个随机目标方向（单位向量）
      3. 使用 PGD (Projected Gradient Descent) 在原图像素空间迭代优化：
         - 像素值归一化到 [0,1]，作为可微张量
         - 每一步通过 transform pipeline (Resize→CenterCrop→Normalize) 送入模型
         - loss = cosine_similarity(embedding, original_embedding)
         - 反向传播梯度回原图像素，做扰动更新
      4. 投影到 eps-ball 内，保证像素变化不超过 eps/255

    参数：
      target_distance: 目标余弦距离（0.0=相同, 1.0=正交, 2.0=完全相反）
      eps: 像素最大扰动范围 (0-255)，2.0 对应约 0.8% 人眼几乎不可见
      step_size: PGD 每步的步长
      steps: PGD 迭代次数
    """
    device = extractor.device

    # ---------- 构建全分辨率可微张量 ----------
    img_np = np.array(pil_img).astype(np.float32) / 255.0   # (H, W, 3), [0, 1]
    original_pixel = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    original_pixel.requires_grad_(False)

    # ---------- 原始 embedding（用标准 transform） ----------
    original_embedding = extractor.extract(pil_img).unsqueeze(0).to(device)  # (1, 2048)

    # ---------- 随机目标方向 ----------
    rng_gen = torch.Generator(device=device)
    if seed is not None:
        rng_gen.manual_seed(seed)
    else:
        rng_gen.manual_seed(int(time.time() * 1000) % (2**32))

    target_dir = torch.randn(1, 2048, generator=rng_gen, device=device)
    target_dir = F.normalize(target_dir, p=2, dim=1)

    # 确保目标方向与原始 embedding 不完全平行
    dot = torch.sum(target_dir * original_embedding, dim=1)
    if torch.abs(dot) > 0.99:
        target_dir = target_dir - dot.unsqueeze(1) * original_embedding
        target_dir = F.normalize(target_dir, p=2, dim=1)

    # ---------- PGD 攻击（全分辨率） ----------
    perturbed = original_pixel.clone().detach().requires_grad_(True)

    # ImageNet 归一化参数
    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # 构建可微的前向函数：pixel tensor → transform → model → embedding
    def forward_through_model(pixel_tensor: torch.Tensor) -> torch.Tensor:
        """(1,3,H,W) @ [0,1] → (1,2048) L2-normalized embedding."""
        # ImageNet normalize
        normalized = (pixel_tensor - mean_tensor) / std_tensor
        # Resize to 256 then center crop 224（与 extractor.transform 一致）
        resized = F.interpolate(normalized, size=256, mode="bilinear", align_corners=False)
        _, _, h, w = resized.shape
        top = (h - 224) // 2
        left = (w - 224) // 2
        cropped = resized[:, :, top:top+224, left:left+224]
        # 提取特征
        features = extractor.feature_extractor(cropped)  # (1, 2048, 1, 1)
        features = features.flatten(1)                    # (1, 2048)
        return F.normalize(features, p=2, dim=1)

    for step in range(steps):
        # 前向传播
        features = forward_through_model(perturbed)

        # Loss: 最小化 cosine_similarity，让 embedding 远离原图
        cosine_sim = F.cosine_similarity(features, original_embedding, dim=1)
        loss = cosine_sim.mean()

        extractor.feature_extractor.zero_grad()
        loss.backward()

        # 梯度下降步进
        perturbed_data = perturbed.data - step_size * perturbed.grad.sign()

        # 投影到 eps-ball
        delta = perturbed_data - original_pixel
        delta = torch.clamp(delta, -eps / 255.0, eps / 255.0)
        perturbed_data = original_pixel + delta

        # 像素值裁剪到 [0, 1]
        perturbed_data = torch.clamp(perturbed_data, 0.0, 1.0)

        perturbed = perturbed_data.detach().requires_grad_(True)

        # 提前停止
        with torch.no_grad():
            current_emb = forward_through_model(perturbed)
            current_dist = (1.0 - F.cosine_similarity(current_emb, original_embedding, dim=1).item())
            if current_dist >= target_distance:
                break

    # ---------- 还原为 PIL 图像 ----------
    arr = perturbed.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================
#  组合策略管道
# ============================================================

def apply_combined_pipeline(
    pil_img: Image.Image,
    extractor: EmbeddingExtractor,
    variant_index: int,
    seed: int | None = None,
    noise_sigma: float = 1.0,
    channel_shift: int = 2,
    jpeg_quality: int = 93,
    adv_eps: float = 2.0,
    adv_target_dist: float = 0.10,
    adv_steps: int = 30,
) -> Image.Image:
    """
    按顺序叠加多层感知层修改，生成一个变体。

    策略执行顺序（每一步都改变文件）：
      1. Embedding 对抗扰动（PGD，核心策略）
      2. 像素高斯噪声注入
      3. 颜色通道微偏移
      4. JPEG 重编码
      5. EXIF 元数据注入
    """
    rng = random.Random(seed)
    result = pil_img.copy()

    # 1. Embedding 对抗扰动
    sub_seed = rng.randint(0, 2**32)
    result = generate_adversarial_embedding_perturbation(
        result, extractor,
        target_distance=adv_target_dist,
        eps=adv_eps,
        steps=adv_steps,
        seed=sub_seed,
    )

    # 2. 像素高斯噪声
    sub_seed = rng.randint(0, 2**32)
    result = add_gaussian_noise(result, sigma=noise_sigma, seed=sub_seed)

    # 3. 颜色通道偏移
    sub_seed = rng.randint(0, 2**32)
    result = shift_color_channels(result, max_shift=channel_shift, seed=sub_seed)

    # 4. JPEG 重编码
    result = jpeg_reencode(result, quality=jpeg_quality)

    # 5. EXIF 元数据
    variant_id = f"adv-{variant_index}-{rng.randint(1000, 9999)}"
    result = inject_exif_metadata(result, variant_id)

    return result


# ============================================================
#  验证指标
# ============================================================

def compute_md5(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """计算两张图片的 SSIM。返回 0~1，越接近 1 越相似。"""
    arr1 = np.array(img1.convert("L"))
    arr2 = np.array(img2.convert("L"))
    if arr1.shape != arr2.shape:
        arr2 = np.array(img2.convert("L").resize((arr1.shape[1], arr1.shape[0])))
    return ssim(arr1, arr2, data_range=255)


def compute_dhash_distance(img1: Image.Image, img2: Image.Image) -> int:
    """计算 dHash 汉明距离。0=完全相同，64=完全不同。"""
    h1 = imagehash.dhash(img1)
    h2 = imagehash.dhash(img2)
    return h1 - h2


def compute_embedding_distance(
    img1: Image.Image, img2: Image.Image, extractor: EmbeddingExtractor
) -> float:
    """计算两张图片 embedding 的余弦距离 (0=相同, 2=完全相反)。"""
    e1 = extractor.extract(img1)
    e2 = extractor.extract(img2)
    cosine_sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    return 1.0 - cosine_sim


def pil_to_bytes(pil_img: Image.Image, format: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=95)
    return buf.getvalue()




# ============================================================
#  主流程
# ============================================================

def process_single_image(
    input_path: str,
    output_dir: str,
    count: int = 3,
    seed: int | None = None,
    noise_sigma: float = 1.0,
    channel_shift: int = 2,
    jpeg_quality: int = 93,
    adv_eps: float = 2.0,
    adv_target_dist: float = 0.10,
    adv_steps: int = 30,
) -> ProcessReport:
    """
    对单张图片生成 count 个感知层变体。

    Returns:
        ProcessReport 包含所有变体的处理结果和验证指标。
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 加载原图
    try:
        with Image.open(input_path) as img:
            pil_img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        print(f"跳过，原因：{exc}")

    original_bytes = pil_to_bytes(pil_img, format="JPEG")
    original_md5 = compute_md5(original_bytes)

    print(f"\n{'='*60}")
    print(f"  图片感知层修改 Demo — Embedding 对抗裂变")
    print(f"{'='*60}")
    print(f"  输入: {input_path}")
    print(f"  尺寸: {pil_img.size}")
    print(f"  原图 MD5: {original_md5}")
    print(f"  生成数量: {count}")
    print(f"  参数: noise={noise_sigma}, shift={channel_shift}, "
          f"jpeg_q={jpeg_quality}, eps={adv_eps}, "
          f"target_dist={adv_target_dist}, steps={adv_steps}")
    print(f"{'='*60}\n")

    # 加载模型
    print("⏳ 加载 ResNet50 特征提取器...")
    t0 = time.time()
    extractor = EmbeddingExtractor(device="cpu")
    print(f"✅ 模型加载完成 ({(time.time()-t0)*1000:.0f}ms)\n")

    # 原始 embedding
    print("⏳ 计算原图 embedding...")
    original_embedding = extractor.extract(pil_img)
    print(f"✅ 原图 embedding 维度: {original_embedding.shape}\n")

    strategies = [
        "adversarial_embedding_pgd",
        "gaussian_noise",
        "color_channel_shift",
        "jpeg_reencode",
        "exif_injection",
    ]

    report = ProcessReport(
        input_path=input_path,
        md5_original=original_md5,
        strategies_applied=strategies,
    )

    rng = random.Random(seed)

    for i in range(count):
        print(f"--- 变体 {i+1}/{count} ---")
        sub_seed = rng.randint(0, 2**32)

        t_start = time.time()

        # 生成变体
        variant_img = apply_combined_pipeline(
            pil_img, extractor,
            variant_index=i,
            seed=sub_seed,
            noise_sigma=noise_sigma,
            channel_shift=channel_shift,
            jpeg_quality=jpeg_quality,
            adv_eps=adv_eps,
            adv_target_dist=adv_target_dist,
            adv_steps=adv_steps,
        )

        # 保存
        stem = Path(input_path).stem
        suffix = Path(input_path).suffix or ".jpg"
        variant_name = f"{stem}_variant_{i+1}.jpg"
        output_path = output_dir_path / variant_name
        variant_img.save(str(output_path), quality=95)

        # 验证指标
        variant_bytes = pil_to_bytes(variant_img, format="JPEG")
        variant_md5 = compute_md5(variant_bytes)

        ssim_val = compute_ssim(pil_img, variant_img)
        dhash_dist = compute_dhash_distance(pil_img, variant_img)
        emb_dist = compute_embedding_distance(pil_img, variant_img, extractor)

        elapsed = (time.time() - t_start) * 1000

        result = VariantResult(
            variant_name=variant_name,
            strategy="+".join(strategies),
            md5_original=original_md5,
            md5_variant=variant_md5,
            ssim=round(ssim_val, 6),
            dhash_distance=dhash_dist,
            embedding_cosine_dist=round(emb_dist, 6),
            file_size_original=len(original_bytes),
            file_size_variant=len(variant_bytes),
            elapsed_ms=round(elapsed, 1),
            output_path=str(output_path),
        )
        report.variants.append(result)

        # 打印结果
        md5_changed = "✅" if result.md5_original != result.md5_variant else "❌"
        print(f"  MD5: {result.md5_variant[:16]}...  {md5_changed}")
        print(f"  SSIM: {result.ssim:.6f}  {'✅' if result.ssim > 0.99 else '⚠️'}")
        print(f"  dHash距离: {result.dhash_distance}  {'✅' if result.dhash_distance > 0 else '⚠️'}")
        print(f"  Embedding余弦距离: {result.embedding_cosine_dist:.6f}  {'✅' if result.embedding_cosine_dist > 0.05 else '⚠️'}")
        print(f"  文件大小: {result.file_size_original:,} → {result.file_size_variant:,} bytes")
        print(f"  耗时: {result.elapsed_ms:.0f}ms")
        print(f"  保存: {output_path}")
        print()

    # 汇总
    print(f"{'='*60}")
    print(f"  处理汇总")
    print(f"{'='*60}")
    for v in report.variants:
        print(f"  {v.variant_name}: SSIM={v.ssim:.4f}  "
              f"dHash={v.dhash_distance}  "
              f"EmbDist={v.embedding_cosine_dist:.4f}  "
              f"MD5={'✅已变' if v.md5_original != v.md5_variant else '❌未变'}  "
              f"{v.elapsed_ms:.0f}ms")
    print(f"{'='*60}")

    return report

def main():
    # params = {
    #     imgPath: "C:\\Users\\Administrator\\Pictures\\u=1798323211,726217455&fm=3074&app=3074&f=PNG.png",
    #     "output": "./output/adversarial_variants",
    #     "count": 3,
    #     "seed": None,
    #     "noise": 0.5,
    #     "shift": 1,
    #     "jpeg-quality": 93,
    #     "eps": 2.0,
    #     "steps": 30,
    #     "target-dist": 0.10
    # }
    params = {
        "imgPath": r"C:\\Users\\Administrator\\Pictures\\u=1798323211,726217455&fm=3074&app=3074&f=PNG.png",
        "output": "./output/adversarial_variants",
        "count": 3,
        "seed": None,
        "noise": 1.5,  # 增加随机噪声(可以增加 噪声)
        "shift": 2,  # 增加通道偏移(RGB通道偏移)
        "jpeg_quality": 80,  # 轻度压缩(注意过低视觉会有明显失真)
        "eps": 5.0,  # 增大对抗扰动(太大可能肉眼可见变形，需要平衡)
        "steps": 60,  # 增加优化步数(迭代步数)
        "target_dist": 0.3  # Embedding 距离目标增大 (可以尝试 0.3~0.5 → CLIP 特征距离会明显增加)
    }

    report = process_single_image(
        input_path=params["imgPath"],
        output_dir=params["output"],
        count=params["count"],
        seed=params["seed"],
        noise_sigma=params["noise"],
        channel_shift=params["shift"],
        jpeg_quality=params["jpeg_quality"],
        adv_eps=params["eps"],
        adv_target_dist=params["target_dist"],
        adv_steps=params["steps"],
    )

    # 保存报告 JSON
    report_path = "_report.json"
    report_dict = {
        "input_path": report.input_path,
        "md5_original": report.md5_original,
        "strategies_applied": report.strategies_applied,
        "variants": [
            {
                "variant_name": v.variant_name,
                "strategy": v.strategy,
                "md5_original": v.md5_original,
                "md5_variant": v.md5_variant,
                "md5_changed": v.md5_original != v.md5_variant,
                "ssim": v.ssim,
                "dhash_distance": v.dhash_distance,
                "embedding_cosine_dist": v.embedding_cosine_dist,
                "file_size_original": v.file_size_original,
                "file_size_variant": v.file_size_variant,
                "elapsed_ms": v.elapsed_ms,
                "output_path": v.output_path,
            }
            for v in report.variants
        ],
    }
    report_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📄 报告已保存: {report_path}")


if __name__ == "__main__":
    main()


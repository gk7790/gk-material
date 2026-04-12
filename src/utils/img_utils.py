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
"""

import random
import hashlib
from pathlib import Path
from typing import Any
from typing import Union
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, UnidentifiedImageError



SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
PREVIEW_SIZE = (32, 32)
DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_FILE_READY_AGE_SECONDS = 2.0
METRICS_FILENAME = "variant_metrics.jsonl"

RESAMPLING = getattr(Image, "Resampling", Image)
LAST_MESSAGE: str | None = None


# ============================================================
# 默认配置
# ============================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "variant_count": 3,
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
}

# ============================================================
# 通用工具 start
# ============================================================

def ensure_directory(path: Path) -> None:
    """
    确认路径是否存在, 如果不存在则创建路径
    :param path: 路径
    :return: None
    """
    path.mkdir(parents=True, exist_ok=True)

def compute_image_md5(file: Union[Path, bytes]) -> str:
    """
    计算图片的 MD5 值。
    参数： file (Path | bytes)：图片文件路径或图片的字节数据。
    返回：str：MD5 十六进制字符串。
    """
    digest = hashlib.md5()

    if isinstance(file, Path):
        # 从文件读取数据，分块计算 MD5，适合大文件
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    elif isinstance(file, bytes):
        # 直接计算内存字节数据的 MD5
        digest.update(file)
    else:
        raise TypeError("file 参数必须是 Path 或 bytes 类型")
    return digest.hexdigest()

def random_delta(rng: random.Random, magnitude: float) -> float:
    return rng.uniform(-magnitude, magnitude)

def type_file(ext: str) -> str | None:
    """根据扩展名判断文件类型。返回 'image' / 'video' / None。"""
    ext = ext.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    return None

# ============================================================
# 通用工具 end
# ============================================================

# ============================================================
# 图片变换 start
# ============================================================

def border_fill_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.uint8)
    border = np.concatenate([pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]], axis=0)
    averages = border.mean(axis=0)
    return tuple(int(c) for c in averages[:3])


# ============================================================
# 图片变换 end
# ============================================================

# ============================================================
# 哈希 / 相似度 start
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
# 哈希 / 相似度 end
# ============================================================


# ============================================================
# HSV 颜色转换 & 同色系变调 start
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
# HSV 颜色转换 & 同色系变调 end
# ============================================================


# ============================================================
#  感知层修改策略 start
# ============================================================


def apply_crop_scale_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    """
    图片增强/变体处理函数
    随机裁剪然后缩放图片回原尺寸
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """

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

    crop_width = max(2, width - left - right)
    crop_height = max(2, height - top - bottom)
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), RESAMPLING.LANCZOS)


def apply_rotation_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    """
    随机旋转图片一定角度
    让图片有轻微旋转变化，同时保持原始大小和视觉边界处理。
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """
    if not cfg.get("enabled", False):
        return image

    max_degrees = float(cfg.get("max_degrees", 0.0))
    if max_degrees <= 0.0:
        return image

    angle = rng.uniform(-max_degrees, max_degrees)
    if abs(angle) < 0.05:
        return image

    return image.rotate(
        angle,
        resample=RESAMPLING.BICUBIC,
        fillcolor=border_fill_color(image),
    )


def apply_color_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    """
    随机调整亮度、对比度和饱和度，生成略有变化的图片。通常用于 图片变体生成 或 训练神经网络的数据增强。
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """
    if not cfg.get("enabled", False):
        return image

    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    saturation = float(cfg.get("saturation", 0.0))

    if brightness > 0.0:
        factor = 1.0 + random_delta(rng, brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if contrast > 0.0:
        factor = 1.0 + random_delta(rng, contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if saturation > 0.0:
        factor = 1.0 + random_delta(rng, saturation)
        image = ImageEnhance.Color(image).enhance(factor)
    return image


def apply_same_family_recolor_image(
    image: Image.Image,
    cfg: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    """
    在保持整体色系一致的情况下，对图片进行颜色微调或变换
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """
    if not cfg.get("enabled", False):
        return image

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    spec = build_same_family_recolor_spec(rgb, cfg, rng)
    if spec is None:
        return image

    recolored = apply_same_family_recolor_to_rgb(rgb, spec)
    return Image.fromarray(recolored, mode="RGB")


def apply_gamma_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    """
    伽马校正增强函数，主要作用是对图片进行 亮度非线性调整，生成略有亮暗变化的图片变体
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """
    if not cfg.get("enabled", False):
        return image

    minimum = float(cfg.get("min", 1.0))
    maximum = float(cfg.get("max", 1.0))
    if minimum <= 0.0 or maximum <= 0.0:
        return image

    gamma = rng.uniform(minimum, maximum)
    if abs(gamma - 1.0) < 0.01:
        return image

    inverse_gamma = 1.0 / gamma
    lut = [int(round(((idx / 255.0) ** inverse_gamma) * 255.0)) for idx in range(256)]
    return image.point(lut * 3)

def apply_noise_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    """
    随机像素噪声增强函数，作用是 在图片上随机添加少量噪点，生成略有扰动的图片变体
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """
    if not cfg.get("enabled", False):
        return image

    width, height = image.size
    pixels_min = int(cfg.get("pixels_min", 0))
    pixels_max = int(cfg.get("pixels_max", 0))
    noise_range = int(cfg.get("range", 0))
    if pixels_max <= 0 or noise_range <= 0:
        return image

    count = rng.randint(max(0, pixels_min), max(pixels_min, pixels_max))
    result = image.copy()
    pixels = result.load()

    for _ in range(count):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        red, green, blue = pixels[x, y]
        pixels[x, y] = (
            max(0, min(255, red + rng.randint(-noise_range, noise_range))),
            max(0, min(255, green + rng.randint(-noise_range, noise_range))),
            max(0, min(255, blue + rng.randint(-noise_range, noise_range))),
        )

    return result

def apply_blur_sharpen_image(image: Image.Image, transforms: dict[str, Any], rng: random.Random) -> Image.Image:
    """
    模糊/锐化增强函数，主要作用是对图片随机添加 高斯模糊或锐化效果
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """
    blur_cfg = transforms.get("blur", {})
    sharpen_cfg = transforms.get("sharpen", {})

    if blur_cfg.get("enabled", False) and rng.random() < float(blur_cfg.get("probability", 0.0)):
        radius_max = float(blur_cfg.get("radius_max", 0.0))
        if radius_max > 0.0:
            radius = rng.uniform(0.15, radius_max)
            image = image.filter(ImageFilter.GaussianBlur(radius))

    if sharpen_cfg.get("enabled", False) and rng.random() < float(sharpen_cfg.get("probability", 0.0)):
        factor_min = float(sharpen_cfg.get("factor_min", 1.0))
        factor_max = float(sharpen_cfg.get("factor_max", factor_min))
        factor = rng.uniform(factor_min, factor_max)
        image = ImageEnhance.Sharpness(image).enhance(factor)

    return image


def apply_logo_image(
    image: Image.Image,
    logo_path: Path | None,
    cfg: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    """
    给图片添加随机 logo 的增强函数，功能是把指定的 logo 图片贴到目标图片上，并可随机控制 大小、旋转角度、透明度、位置。常用于生成带水印或品牌标识的图片变体。
    :param image: 图片
    :param cfg: 配置
    :param rng: 随机
    :return: 返回图片
    """
    if logo_path is None or not cfg.get("enabled", False):
        return image

    width, height = image.size
    if width < 2 or height < 2:
        return image

    with Image.open(logo_path) as src_logo:
        logo = src_logo.convert("RGBA")

    scale_min = float(cfg.get("scale_min", 0.1))
    scale_max = float(cfg.get("scale_max", scale_min))
    opacity_min = float(cfg.get("opacity_min", 0.8))
    opacity_max = float(cfg.get("opacity_max", opacity_min))
    margin_ratio = float(cfg.get("margin_ratio", 0.04))
    rotate_max = float(cfg.get("rotate_max_degrees", 0.0))

    target_width = max(1, int(width * rng.uniform(scale_min, scale_max)))
    target_width = min(target_width, max(1, width - 2))
    resize_ratio = target_width / max(1, logo.width)
    target_height = max(1, int(logo.height * resize_ratio))
    logo = logo.resize((target_width, target_height), RESAMPLING.LANCZOS)

    if rotate_max > 0.0:
        angle = rng.uniform(-rotate_max, rotate_max)
        logo = logo.rotate(
            angle,
            resample=RESAMPLING.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )

    opacity = rng.uniform(opacity_min, opacity_max)
    alpha = logo.getchannel("A").point(lambda value: int(value * opacity))
    logo.putalpha(alpha)

    logo_width, logo_height = logo.size
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    x_min = margin_x
    y_min = margin_y
    x_max = max(x_min, width - logo_width - margin_x)
    y_max = max(y_min, height - logo_height - margin_y)
    x = rng.randint(x_min, x_max)
    y = rng.randint(y_min, y_max)

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste(logo, (x, y), logo)
    return Image.alpha_composite(base, overlay).convert("RGB")


def build_image_candidate(
    source_image: Image.Image,
    logo_path: Path | None,
    config: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    """
    加工图片
    :param source_image:
    :param logo_path:
    :param config:
    :param rng:
    :return:
    """
    transforms = config["transforms"]
    image = source_image.copy()
    image = apply_crop_scale_image(image, transforms.get("crop_scale", {}), rng)
    image = apply_rotation_image(image, transforms.get("rotation", {}), rng)
    image = apply_color_image(image, transforms.get("color", {}), rng)
    image = apply_same_family_recolor_image(image, transforms.get("same_family_recolor", {}), rng)
    image = apply_gamma_image(image, transforms.get("gamma", {}), rng)
    image = apply_noise_image(image, transforms.get("noise", {}), rng)
    image = apply_blur_sharpen_image(image, transforms, rng)
    image = apply_logo_image(image, logo_path, transforms.get("logo", {}), rng)
    return image

# ============================================================
#  感知层修改策略 end
# ============================================================

def build_image_metrics(
    source_path: Path,
    output_path: Path,
    source_image: Image.Image,
    candidate_image: Image.Image,
    source_md5: str,
    output_md5: str,
) -> dict[str, Any]:
    source_hash = image_dhash(source_image)
    output_hash = image_dhash(candidate_image)
    mad = image_mad(source_image, candidate_image)
    hash_distance = hamming_distance(source_hash, output_hash)
    return {
        "mode": "image",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_md5": source_md5,
        "output_md5": output_md5,
        "md5_changed": source_md5 != output_md5,
        "source_dhash": f"{source_hash:016x}",
        "output_dhash": f"{output_hash:016x}",
        "dhash_changed": source_hash != output_hash,
        "dhash_distance": hash_distance,
        "mad": round(mad, 4),
    }



def main():
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

    input_path = r"C:\\Users\\Administrator\\Pictures\\u=1798323211,726217455&fm=3074&app=3074&f=PNG.png"
    logo_path = r"E:\Documents\海外推广\image\logo.png"
    config = DEFAULT_CONFIG
    rng = random.Random()

    try:
        with Image.open(input_path) as src:
            source_image = ImageOps.exif_transpose(src).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        print(f"\n 报告已保存 {exc}")

    accepted_hashes = [image_dhash(source_image)]

    smd5 = image_dhash(source_image)
    print(f"\n s: {smd5}")

    candidate = build_image_candidate(source_image, None, config, rng)
    cmd5 = image_dhash(candidate)
    print(f"\n s: {cmd5}")

    distance = min(hamming_distance(cmd5, existing) for existing in accepted_hashes)
    print(f"\n s: {distance}")


if __name__ == "__main__":
    main()








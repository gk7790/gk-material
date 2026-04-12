from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
import shutil
import subprocess
import sys
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
PREVIEW_SIZE = (32, 32)
LOGO_KEYWORDS = ("logo", "水印", "标志")
DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_MATERIAL_WATCH_DIR = DOWNLOADS_DIR / "待处理素材"
DEFAULT_MATERIAL_OUTPUT_DIR = DOWNLOADS_DIR / "已处理素材"
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_FILE_READY_AGE_SECONDS = 2.0
METRICS_FILENAME = "variant_metrics.jsonl"

RESAMPLING = getattr(Image, "Resampling", Image)
LAST_MESSAGE: str | None = None

DEFAULT_CONFIG: dict[str, Any] = {
    "variant_count": 3,
    "image": {
        "max_attempts_per_variant": 6,
        "similarity": {
            "enabled": True,
            "min_dhash_distance": 7,
        },
        "transforms": {
            "crop_scale": {
                "enabled": True,
                "max_crop_ratio": 0.05,
            },
            "rotation": {
                "enabled": True,
                "max_degrees": 1.2,
            },
            "color": {
                "enabled": True,
                "brightness": 0.06,
                "contrast": 0.08,
                "saturation": 0.06,
            },
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
            "gamma": {
                "enabled": True,
                "min": 0.95,
                "max": 1.05,
            },
            "noise": {
                "enabled": True,
                "pixels_min": 80,
                "pixels_max": 180,
                "range": 12,
            },
            "blur": {
                "enabled": True,
                "probability": 0.30,
                "radius_max": 0.8,
            },
            "sharpen": {
                "enabled": True,
                "probability": 0.35,
                "factor_min": 1.1,
                "factor_max": 1.8,
            },
            "logo": {
                "enabled": True,
                "scale_min": 0.10,
                "scale_max": 0.18,
                "opacity_min": 0.72,
                "opacity_max": 0.95,
                "margin_ratio": 0.04,
                "rotate_max_degrees": 3.0,
            },
            "jpeg_quality": {
                "enabled": True,
                "min": 88,
                "max": 96,
            },
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
            "crop_scale": {
                "enabled": True,
                "max_crop_ratio": 0.04,
            },
            "rotation": {
                "enabled": True,
                "probability": 0.65,
                "max_degrees": 0.9,
            },
            "eq": {
                "enabled": True,
                "brightness": 0.03,
                "contrast": 0.06,
                "saturation": 0.08,
                "gamma": 0.06,
            },
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
            "noise": {
                "enabled": True,
                "strength_min": 2,
                "strength_max": 5,
            },
            "blur": {
                "enabled": True,
                "probability": 0.25,
                "sigma_max": 0.75,
            },
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
            "encoder": {
                "preset": "medium",
                "crf": 18,
            },
        },
    },
}


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(override)


def load_config(path: Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return deep_merge(config, loaded)


def announce(message: str) -> None:
    global LAST_MESSAGE
    if LAST_MESSAGE != message:
        print(message)
        LAST_MESSAGE = message


def ensure_tool_exists(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"未找到依赖工具：{tool_name}")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_state_file(state_file: Path | None, watch_dir: Path) -> Path:
    if state_file is not None:
        return state_file
    return watch_dir / ".processed_state.json"


def load_state(state_file: Path) -> dict[str, str]:
    if not state_file.exists():
        return {}

    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state_file: Path, state: dict[str, str]) -> None:
    ensure_directory(state_file.parent)
    temp_file = state_file.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(state_file)


def file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"


def is_file_ready(path: Path, file_ready_age_seconds: float) -> bool:
    try:
        age_seconds = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age_seconds >= file_ready_age_seconds


def is_logo_file(path: Path) -> bool:
    lowered_name = path.stem.lower()
    return any(keyword in lowered_name for keyword in LOGO_KEYWORDS)


def pick_latest_logo(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def compute_file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_manifest_path(output_dir: Path) -> Path:
    return output_dir / METRICS_FILENAME


def append_metrics_record(output_dir: Path, record: dict[str, Any]) -> None:
    manifest_path = metrics_manifest_path(output_dir)
    ensure_directory(manifest_path.parent)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_files(input_path: Path, extensions: set[str]) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        return []

    if input_path.is_dir():
        return sorted(
            [
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            ],
            key=lambda item: item.name.lower(),
        )

    return []


def random_delta(rng: random.Random, magnitude: float) -> float:
    return rng.uniform(-magnitude, magnitude)


def safe_even(value: int, minimum: int = 2) -> int:
    value = max(minimum, int(value))
    if value % 2 == 1:
        value -= 1
    return max(minimum, value)


def build_output_path(source_path: Path, output_dir: Path, label: str, index: int, suffix: str) -> Path:
    unique_marker = uuid.uuid4().hex[:6]
    return output_dir / f"{source_path.stem}_{label}_{index:02d}_{unique_marker}{suffix}"


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


def dhash_hex_width(hash_size: int) -> int:
    return max(16, (hash_size * hash_size) // 4)


def image_mad(left: Image.Image, right: Image.Image) -> float:
    if left.size != right.size:
        right = right.resize(left.size, RESAMPLING.BILINEAR)
    left_pixels = np.asarray(left.convert("RGB"), dtype=np.float32)
    right_pixels = np.asarray(right.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(left_pixels - right_pixels)))


def signature_hashes(signature: tuple[tuple[int, np.ndarray], ...]) -> list[int]:
    return [item[0] for item in signature]


def signature_previews(signature: tuple[tuple[int, np.ndarray], ...]) -> list[np.ndarray]:
    return [item[1] for item in signature]


def average_signature_hash_distance(
    left: tuple[tuple[int, np.ndarray], ...],
    right: tuple[tuple[int, np.ndarray], ...],
) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    distances = [hamming_distance(left[idx][0], right[idx][0]) for idx in range(size)]
    return float(sum(distances) / len(distances))


def average_signature_mad(
    left: tuple[tuple[int, np.ndarray], ...],
    right: tuple[tuple[int, np.ndarray], ...],
) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    distances = [
        float(np.mean(np.abs(left[idx][1] - right[idx][1])))
        for idx in range(size)
    ]
    return float(sum(distances) / len(distances))


def average_video_distance(
    left: tuple[tuple[int, np.ndarray], ...],
    right: tuple[tuple[int, np.ndarray], ...],
) -> float:
    if not left or not right:
        return 0.0

    size = min(len(left), len(right))
    distances: list[float] = []
    for idx in range(size):
        left_hash, left_preview = left[idx]
        right_hash, right_preview = right[idx]
        hash_distance = hamming_distance(left_hash, right_hash)
        preview_distance = float(np.mean(np.abs(left_preview - right_preview)))
        distances.append(hash_distance + preview_distance)
    return float(sum(distances) / len(distances))


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


def build_video_metrics(
    source_path: Path,
    output_path: Path,
    source_signature: tuple[tuple[int, np.ndarray], ...],
    candidate_signature: tuple[tuple[int, np.ndarray], ...],
    source_md5: str,
    output_md5: str,
    sample_positions: list[float],
    hash_size: int,
) -> dict[str, Any]:
    average_hash_distance = average_signature_hash_distance(source_signature, candidate_signature)
    average_mad = average_signature_mad(source_signature, candidate_signature)
    hash_width = dhash_hex_width(hash_size)
    source_hash_list = [f"{value:0{hash_width}x}" for value in signature_hashes(source_signature)]
    output_hash_list = [f"{value:0{hash_width}x}" for value in signature_hashes(candidate_signature)]
    return {
        "mode": "video",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_md5": source_md5,
        "output_md5": output_md5,
        "md5_changed": source_md5 != output_md5,
        "sample_positions": sample_positions,
        "dhash_hash_size": hash_size,
        "source_dhashes": source_hash_list,
        "output_dhashes": output_hash_list,
        "dhash_changed": source_hash_list != output_hash_list,
        "average_dhash_distance": round(average_hash_distance, 4),
        "mad": round(average_mad, 4),
    }


def border_fill_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.uint8)
    border = np.concatenate(
        [
            pixels[0, :, :],
            pixels[-1, :, :],
            pixels[:, 0, :],
            pixels[:, -1, :],
        ],
        axis=0,
    )
    averages = border.mean(axis=0)
    return tuple(int(channel) for channel in averages[:3])


def rgb_to_hsv_arrays(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_float = rgb.astype(np.float32) / 255.0
    red = rgb_float[:, :, 0]
    green = rgb_float[:, :, 1]
    blue = rgb_float[:, :, 2]

    maximum = np.max(rgb_float, axis=2)
    minimum = np.min(rgb_float, axis=2)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)
    mask = delta > 1e-6

    red_mask = mask & (maximum == red)
    green_mask = mask & (maximum == green)
    blue_mask = mask & (maximum == blue)

    hue[red_mask] = np.mod((green[red_mask] - blue[red_mask]) / delta[red_mask], 6.0)
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2.0
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4.0
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maximum)
    nonzero_value = maximum > 1e-6
    saturation[nonzero_value] = delta[nonzero_value] / maximum[nonzero_value]
    value = maximum
    return hue, saturation, value


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

    mask = sector == 0
    red[mask], green[mask], blue[mask] = value[mask], t[mask], p[mask]
    mask = sector == 1
    red[mask], green[mask], blue[mask] = q[mask], value[mask], p[mask]
    mask = sector == 2
    red[mask], green[mask], blue[mask] = p[mask], value[mask], t[mask]
    mask = sector == 3
    red[mask], green[mask], blue[mask] = p[mask], q[mask], value[mask]
    mask = sector == 4
    red[mask], green[mask], blue[mask] = t[mask], p[mask], value[mask]
    mask = sector == 5
    red[mask], green[mask], blue[mask] = value[mask], p[mask], q[mask]

    rgb = np.stack([red, green, blue], axis=2)
    return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


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
    center = float((edges[top_index] + edges[top_index + 1]) / 2.0)
    return center


def build_same_family_recolor_spec(
    rgb: np.ndarray,
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
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

    if style == "lighter":
        saturation_scale = rng.uniform(0.72, 0.95)
        value_scale = rng.uniform(1.04, 1.16)
        value_offset = rng.uniform(max(0.02, value_offset_min), value_offset_max)
        hue_shift = rng.uniform(-max_shift * 0.4, max_shift * 0.4)
    elif style == "muted":
        saturation_scale = rng.uniform(0.55, 0.82)
        value_scale = rng.uniform(1.01, 1.10)
        value_offset = rng.uniform(max(0.02, value_offset_min), max(0.12, value_offset_max * 0.7))
        hue_shift = rng.uniform(-max_shift * 0.5, max_shift * 0.5)
    elif style == "deeper":
        saturation_scale = rng.uniform(1.04, 1.22)
        value_scale = rng.uniform(0.82, 0.95)
        value_offset = rng.uniform(min(-0.10, value_offset_min), min(-0.02, value_offset_max))
        hue_shift = rng.uniform(-max_shift * 0.35, max_shift * 0.35)
    elif style == "richer":
        saturation_scale = rng.uniform(1.08, 1.28)
        value_scale = rng.uniform(0.94, 1.05)
        value_offset = rng.uniform(max(-0.04, value_offset_min), min(0.06, value_offset_max))
        hue_shift = rng.uniform(-max_shift * 0.65, max_shift * 0.65)
    else:
        saturation_scale = rng.uniform(0.88, 1.08)
        value_scale = rng.uniform(1.02, 1.14)
        value_offset = rng.uniform(max(0.01, value_offset_min), max(0.10, value_offset_max * 0.8))
        hue_shift = rng.uniform(max_shift * 0.15, max_shift)

    return {
        "style": style,
        "center_hue": dominant_hue,
        "hue_shift": hue_shift,
        "hue_window": max(window, 1.0 / 360.0),
        "feather": max(feather, 1.0 / 360.0),
        "min_saturation": float(cfg.get("min_saturation", 0.16)),
        "min_value": float(cfg.get("min_value", 0.12)),
        "saturation_scale": saturation_scale,
        "value_scale": value_scale,
        "value_offset": value_offset,
    }


def apply_same_family_recolor_to_rgb(
    rgb: np.ndarray,
    spec: dict[str, Any] | None,
) -> np.ndarray:
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
    saturation = np.clip(
        saturation * (1.0 + (float(spec["saturation_scale"]) - 1.0) * influence),
        0.0,
        1.0,
    )
    value = np.clip(
        value * (1.0 + (float(spec["value_scale"]) - 1.0) * influence) + float(spec["value_offset"]) * influence,
        0.0,
        1.0,
    )
    return hsv_to_rgb_array(hue, saturation, value)


def apply_same_family_recolor_image(
    image: Image.Image,
    cfg: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    if not cfg.get("enabled", False):
        return image

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    spec = build_same_family_recolor_spec(rgb, cfg, rng)
    if spec is None:
        return image

    recolored = apply_same_family_recolor_to_rgb(rgb, spec)
    return Image.fromarray(recolored, mode="RGB")


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

    crop_width = max(2, width - left - right)
    crop_height = max(2, height - top - bottom)
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
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

    return image.rotate(
        angle,
        resample=RESAMPLING.BICUBIC,
        fillcolor=border_fill_color(image),
    )


def apply_color_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
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


def apply_gamma_image(image: Image.Image, cfg: dict[str, Any], rng: random.Random) -> Image.Image:
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


def save_image_variant(
    image: Image.Image,
    output_path: Path,
    cfg: dict[str, Any],
    rng: random.Random,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    jpeg_cfg = cfg.get("jpeg_quality", {})
    quality_min = int(jpeg_cfg.get("min", 90))
    quality_max = int(jpeg_cfg.get("max", quality_min))
    quality = quality_max
    if jpeg_cfg.get("enabled", False):
        quality = rng.randint(min(quality_min, quality_max), max(quality_min, quality_max))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="JPEG", quality=quality, optimize=True)


def build_image_candidate(
    source_image: Image.Image,
    logo_path: Path | None,
    config: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
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


def process_images(
    inputs: list[Path],
    output_dir: Path,
    logo_path: Path | None,
    config: dict[str, Any],
    variant_count: int,
    rng: random.Random,
    dry_run: bool,
) -> dict[Path, bool]:
    image_cfg = config["image"]
    similarity_cfg = image_cfg.get("similarity", {})
    min_distance = int(similarity_cfg.get("min_dhash_distance", 0))
    use_similarity = bool(similarity_cfg.get("enabled", False))
    max_attempts = int(image_cfg.get("max_attempts_per_variant", 1))
    results: dict[Path, bool] = {}

    for input_path in inputs:
        print(f"\n[图片] 处理文件：{input_path}")
        results[input_path] = False
        try:
            with Image.open(input_path) as src:
                source_image = ImageOps.exif_transpose(src).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            print(f"跳过，原因：{exc}")
            continue

        source_md5 = compute_file_md5(input_path) if not dry_run else ""
        accepted_hashes = [image_dhash(source_image)]

        for variant_index in range(1, variant_count + 1):
            accepted = False
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
                    print(
                        f"  变体 {variant_index:02d} 第 {attempt} 次尝试被过滤，"
                        f"dHash 距离 {distance} < {min_distance}"
                    )
                    continue

                output_path = build_output_path(input_path, output_dir, "image_variant", variant_index, ".jpg")
                save_image_variant(candidate, output_path, image_cfg["transforms"], rng, dry_run)
                if not dry_run:
                    output_md5 = compute_file_md5(output_path)
                    metrics = build_image_metrics(
                        source_path=input_path,
                        output_path=output_path,
                        source_image=source_image,
                        candidate_image=candidate,
                        source_md5=source_md5,
                        output_md5=output_md5,
                    )
                    append_metrics_record(output_dir, metrics)
                    print(
                        f"  已生成：{output_path}（dHash 距离：{distance}，"
                        f"MAD：{metrics['mad']:.2f}，MD5变化：{metrics['md5_changed']}，"
                        f"感知Hash变化：{metrics['dhash_changed']}）"
                    )
                else:
                    print(f"  已生成：{output_path}（dHash 距离：{distance}）")
                accepted_hashes.append(candidate_hash)
                results[input_path] = True
                accepted = True
                break

            if accepted:
                continue

            if best_image is None:
                print(f"  变体 {variant_index:02d} 生成失败。")
                continue

            output_path = build_output_path(input_path, output_dir, "image_variant", variant_index, ".jpg")
            save_image_variant(best_image, output_path, image_cfg["transforms"], rng, dry_run)
            accepted_hashes.append(image_dhash(best_image))
            if not dry_run:
                output_md5 = compute_file_md5(output_path)
                metrics = build_image_metrics(
                    source_path=input_path,
                    output_path=output_path,
                    source_image=source_image,
                    candidate_image=best_image,
                    source_md5=source_md5,
                    output_md5=output_md5,
                )
                append_metrics_record(output_dir, metrics)
                print(
                    f"  变体 {variant_index:02d} 未达到相似度阈值，"
                    f"已保留最优候选：{output_path}（最佳距离：{best_distance}，"
                    f"MAD：{metrics['mad']:.2f}，MD5变化：{metrics['md5_changed']}，"
                    f"感知Hash变化：{metrics['dhash_changed']}）"
                )
            else:
                print(
                    f"  变体 {variant_index:02d} 未达到相似度阈值，"
                    f"已保留最优候选：{output_path}（最佳距离：{best_distance}）"
                )
            results[input_path] = True

    return results


def run_command(command: list[str], binary_output: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=not binary_output,
    )


def get_video_info(input_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        str(input_path),
    ]
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

    duration_text = payload.get("format", {}).get("duration", "0")
    duration = 0.0
    try:
        duration = max(0.0, float(duration_text))
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
    }


def choose_video_crop(width: int, height: int, cfg: dict[str, Any], rng: random.Random) -> tuple[int, int, int, int] | None:
    if not cfg.get("enabled", False):
        return None

    max_crop_ratio = float(cfg.get("max_crop_ratio", 0.0))
    if max_crop_ratio <= 0.0:
        return None

    left = int(rng.uniform(0.0, max_crop_ratio) * width)
    right = int(rng.uniform(0.0, max_crop_ratio) * width)
    top = int(rng.uniform(0.0, max_crop_ratio) * height)
    bottom = int(rng.uniform(0.0, max_crop_ratio) * height)

    crop_width = safe_even(width - left - right)
    crop_height = safe_even(height - top - bottom)
    x = max(0, min(left, width - crop_width))
    y = max(0, min(top, height - crop_height))
    if x % 2 == 1:
        x -= 1
    if y % 2 == 1:
        y -= 1
    return crop_width, crop_height, x, y


def read_exact(stream: Any, expected_size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < expected_size:
        chunk = stream.read(expected_size - len(buffer))
        if not chunk:
            if not buffer:
                return None
            raise RuntimeError("视频帧数据读取不完整，处理已中断。")
        buffer.extend(chunk)
    return bytes(buffer)


def extract_recolor_reference_frame(
    video_path: Path,
    duration: float,
    cfg: dict[str, Any],
) -> Image.Image:
    sample_position = float(cfg.get("sample_position", 0.35))
    ratio = min(0.98, max(0.0, sample_position))
    timestamp = 0.0 if duration <= 0.0 else min(duration * ratio, max(0.0, duration - 0.05))
    return extract_video_frame(video_path, timestamp)


def build_video_recolor_spec(
    video_path: Path,
    duration: float,
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    if not cfg.get("enabled", False):
        return None
    try:
        reference_frame = extract_recolor_reference_frame(video_path, duration, cfg)
    except RuntimeError:
        return None
    rgb = np.asarray(reference_frame.convert("RGB"), dtype=np.uint8)
    return build_same_family_recolor_spec(rgb, cfg, rng)


def build_video_filters(
    width: int,
    height: int,
    transforms: dict[str, Any],
    rng: random.Random,
) -> list[str]:
    filters: list[str] = []

    crop = choose_video_crop(width, height, transforms.get("crop_scale", {}), rng)
    if crop is not None:
        crop_width, crop_height, x, y = crop
        filters.append(f"crop={crop_width}:{crop_height}:{x}:{y}")
        filters.append(f"scale={width}:{height}")

    rotation_cfg = transforms.get("rotation", {})
    if rotation_cfg.get("enabled", False) and rng.random() < float(rotation_cfg.get("probability", 0.0)):
        max_degrees = float(rotation_cfg.get("max_degrees", 0.0))
        if max_degrees > 0.0:
            angle_degrees = rng.uniform(-max_degrees, max_degrees)
            if abs(angle_degrees) >= 0.05:
                angle_radians = math.radians(angle_degrees)
                filters.append(f"rotate={angle_radians:.7f}:ow=iw:oh=ih:c=black@0")

    eq_cfg = transforms.get("eq", {})
    if eq_cfg.get("enabled", False):
        brightness = random_delta(rng, float(eq_cfg.get("brightness", 0.0)))
        contrast = 1.0 + random_delta(rng, float(eq_cfg.get("contrast", 0.0)))
        saturation = 1.0 + random_delta(rng, float(eq_cfg.get("saturation", 0.0)))
        gamma = 1.0 + random_delta(rng, float(eq_cfg.get("gamma", 0.0)))
        filters.append(
            "eq="
            f"brightness={brightness:.4f}:"
            f"contrast={contrast:.4f}:"
            f"saturation={saturation:.4f}:"
            f"gamma={gamma:.4f}"
        )

    noise_cfg = transforms.get("noise", {})
    if noise_cfg.get("enabled", False):
        strength_min = int(noise_cfg.get("strength_min", 0))
        strength_max = int(noise_cfg.get("strength_max", strength_min))
        if strength_max > 0:
            strength = rng.randint(min(strength_min, strength_max), max(strength_min, strength_max))
            if strength > 0:
                filters.append(f"noise=alls={strength}:allf=t")

    blur_cfg = transforms.get("blur", {})
    if blur_cfg.get("enabled", False) and rng.random() < float(blur_cfg.get("probability", 0.0)):
        sigma_max = float(blur_cfg.get("sigma_max", 0.0))
        if sigma_max > 0.0:
            sigma = rng.uniform(0.15, sigma_max)
            filters.append(f"gblur=sigma={sigma:.3f}")

    border_cfg = transforms.get("border", {})
    if border_cfg.get("enabled", False) and rng.random() < float(border_cfg.get("probability", 0.0)):
        min_ratio = float(border_cfg.get("thickness_min_ratio", 0.0))
        max_ratio = float(border_cfg.get("thickness_max_ratio", min_ratio))
        opacity_min = float(border_cfg.get("opacity_min", 0.06))
        opacity_max = float(border_cfg.get("opacity_max", opacity_min))
        border_thickness = max(1, int(min(width, height) * rng.uniform(min_ratio, max_ratio)))
        border_opacity = rng.uniform(opacity_min, opacity_max)
        border_color = f"{'black' if rng.random() < 0.5 else 'white'}@{border_opacity:.3f}"
        filters.extend(
            [
                f"drawbox=x=0:y=0:w=iw:h={border_thickness}:color={border_color}:t=fill",
                f"drawbox=x=0:y=ih-{border_thickness}:w=iw:h={border_thickness}:color={border_color}:t=fill",
                f"drawbox=x=0:y=0:w={border_thickness}:h=ih:color={border_color}:t=fill",
                f"drawbox=x=iw-{border_thickness}:y=0:w={border_thickness}:h=ih:color={border_color}:t=fill",
            ]
        )

    patch_cfg = transforms.get("patch", {})
    if patch_cfg.get("enabled", False) and rng.random() < float(patch_cfg.get("probability", 0.0)):
        patch_width = max(
            8,
            int(width * rng.uniform(
                float(patch_cfg.get("width_min_ratio", 0.12)),
                float(patch_cfg.get("width_max_ratio", 0.24)),
            )),
        )
        patch_height = max(
            8,
            int(height * rng.uniform(
                float(patch_cfg.get("height_min_ratio", 0.10)),
                float(patch_cfg.get("height_max_ratio", 0.22)),
            )),
        )
        patch_width = min(width, patch_width)
        patch_height = min(height, patch_height)
        patch_x = rng.randint(0, max(0, width - patch_width))
        patch_y = rng.randint(0, max(0, height - patch_height))
        patch_opacity = rng.uniform(
            float(patch_cfg.get("opacity_min", 0.05)),
            float(patch_cfg.get("opacity_max", 0.12)),
        )
        patch_color = f"{'black' if rng.random() < 0.5 else 'white'}@{patch_opacity:.3f}"
        filters.append(
            f"drawbox=x={patch_x}:y={patch_y}:w={patch_width}:h={patch_height}:"
            f"color={patch_color}:t=fill"
        )

    return filters


def pick_video_logo_spec(
    width: int,
    height: int,
    logo_path: Path | None,
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    if logo_path is None or not cfg.get("enabled", False):
        return None

    with Image.open(logo_path) as src_logo:
        logo = src_logo.convert("RGBA")
        logo_width, logo_height = logo.size

    if logo_width <= 0 or logo_height <= 0:
        return None

    scale_min = float(cfg.get("scale_min", 0.1))
    scale_max = float(cfg.get("scale_max", scale_min))
    opacity_min = float(cfg.get("opacity_min", 0.8))
    opacity_max = float(cfg.get("opacity_max", opacity_min))
    margin_ratio = float(cfg.get("margin_ratio", 0.04))

    target_width = max(2, int(width * rng.uniform(scale_min, scale_max)))
    resize_ratio = target_width / logo_width
    target_height = max(2, int(logo_height * resize_ratio))
    target_width = safe_even(min(target_width, max(2, width - 2)))
    target_height = safe_even(min(target_height, max(2, height - 2)))

    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    x_min = margin_x
    y_min = margin_y
    x_max = max(x_min, width - target_width - margin_x)
    y_max = max(y_min, height - target_height - margin_y)
    x = rng.randint(x_min, x_max)
    y = rng.randint(y_min, y_max)

    return {
        "path": logo_path,
        "width": target_width,
        "height": target_height,
        "x": x,
        "y": y,
        "opacity": rng.uniform(opacity_min, opacity_max),
    }


def load_logo_rgba_array(logo_spec: dict[str, Any] | None) -> np.ndarray | None:
    if logo_spec is None:
        return None
    with Image.open(logo_spec["path"]) as src_logo:
        logo = src_logo.convert("RGBA").resize(
            (int(logo_spec["width"]), int(logo_spec["height"])),
            RESAMPLING.LANCZOS,
        )
    alpha = logo.getchannel("A").point(lambda value: int(value * float(logo_spec["opacity"])))
    logo.putalpha(alpha)
    return np.asarray(logo, dtype=np.uint8)


def overlay_logo_on_frame_array(frame: np.ndarray, logo_rgba: np.ndarray, x: int, y: int) -> None:
    logo_height, logo_width = logo_rgba.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    x_end = min(frame_width, x + logo_width)
    y_end = min(frame_height, y + logo_height)
    if x_end <= x or y_end <= y:
        return

    logo_crop = logo_rgba[: y_end - y, : x_end - x]
    alpha = logo_crop[:, :, 3:4].astype(np.float32) / 255.0
    logo_rgb = logo_crop[:, :, :3].astype(np.float32)
    background = frame[y:y_end, x:x_end].astype(np.float32)
    blended = (logo_rgb * alpha + background * (1.0 - alpha)).astype(np.uint8)
    frame[y:y_end, x:x_end] = blended


def build_video_command_from_filters(
    input_path: Path,
    output_path: Path,
    filters: list[str],
    logo_spec: dict[str, Any] | None,
    transforms: dict[str, Any],
    audio_mode: str,
) -> list[str]:
    encoder_cfg = transforms.get("encoder", {})
    preset = str(encoder_cfg.get("preset", "medium"))
    crf = str(int(encoder_cfg.get("crf", 18)))

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
    ]

    if logo_spec is not None:
        command.extend(["-i", str(logo_spec["path"])])
        base_filter = ",".join(filters) if filters else "null"
        filter_complex = (
            f"[0:v]{base_filter}[base];"
            f"[1:v]scale={logo_spec['width']}:{logo_spec['height']},"
            "format=rgba,"
            f"colorchannelmixer=aa={logo_spec['opacity']:.3f}[logo];"
            f"[base][logo]overlay={logo_spec['x']}:{logo_spec['y']}:format=auto[vout]"
        )
        command.extend(["-filter_complex", filter_complex, "-map", "[vout]"])
    else:
        if filters:
            command.extend(["-vf", ",".join(filters)])
        command.extend(["-map", "0:v:0"])

    command.extend(
        [
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )

    if audio_mode == "copy":
        command.extend(["-c:a", "copy"])
    else:
        command.extend(["-c:a", "aac", "-b:a", "192k"])

    command.extend(["-shortest", str(output_path)])
    return command


def build_video_decoder_command(input_path: Path, filters: list[str]) -> list[str]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
    ]
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-an",
            "-",
        ]
    )
    return command


def build_rawvideo_encoder_command(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    transforms: dict[str, Any],
    audio_mode: str,
) -> list[str]:
    encoder_cfg = transforms.get("encoder", {})
    preset = str(encoder_cfg.get("preset", "medium"))
    crf = str(int(encoder_cfg.get("crf", 18)))
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if audio_mode == "copy":
        command.extend(["-c:a", "copy"])
    else:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-shortest", str(output_path)])
    return command


def render_video_variant_with_recolor(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    filters: list[str],
    recolor_spec: dict[str, Any],
    logo_spec: dict[str, Any] | None,
    transforms: dict[str, Any],
    audio_mode: str,
) -> tuple[bool, str]:
    frame_size = width * height * 3
    logo_rgba = load_logo_rgba_array(logo_spec)
    decoder_command = build_video_decoder_command(input_path, filters)
    encoder_command = build_rawvideo_encoder_command(
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        transforms=transforms,
        audio_mode=audio_mode,
    )

    decoder = subprocess.Popen(
        decoder_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        encoder_command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        while True:
            raw_frame = read_exact(decoder.stdout, frame_size)
            if raw_frame is None:
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
            frame = apply_same_family_recolor_to_rgb(frame, recolor_spec)

            if logo_rgba is not None and logo_spec is not None:
                overlay_logo_on_frame_array(frame, logo_rgba, int(logo_spec["x"]), int(logo_spec["y"]))

            encoder.stdin.write(frame.tobytes())

        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()

        decode_code = decoder.wait()
        encode_code = encoder.wait()
        decode_error = decoder.stderr.read().decode("utf-8", errors="ignore").strip()
        encode_error = encoder.stderr.read().decode("utf-8", errors="ignore").strip()

        if decode_code != 0:
            return False, decode_error or "ffmpeg 解码失败。"
        if encode_code != 0:
            return False, encode_error or "ffmpeg 编码失败。"
        return True, ""

    except BrokenPipeError:
        decode_code = decoder.wait()
        encode_code = encoder.wait()
        decode_error = decoder.stderr.read().decode("utf-8", errors="ignore").strip()
        encode_error = encoder.stderr.read().decode("utf-8", errors="ignore").strip()
        if decode_code != 0:
            return False, decode_error or "ffmpeg 解码失败。"
        if encode_code != 0:
            return False, encode_error or "ffmpeg 编码失败。"
        return False, "编码管道意外断开。"

    finally:
        if decoder.stdout:
            decoder.stdout.close()
        if decoder.stderr:
            decoder.stderr.close()
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.stderr:
            encoder.stderr.close()
        if decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()


def extract_video_frame(video_path: Path, timestamp: float) -> Image.Image:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = run_command(command, binary_output=True)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr or "抽帧失败。")

    image = Image.open(io.BytesIO(result.stdout))
    return image.convert("RGB")


def video_signature(
    video_path: Path,
    duration: float,
    sample_positions: list[float],
    hash_size: int = 8,
) -> tuple[tuple[int, np.ndarray], ...]:
    if duration <= 0.0:
        frame = extract_video_frame(video_path, 0.0)
        return ((image_dhash(frame, hash_size=hash_size), image_preview(frame)),)

    signatures: list[tuple[int, np.ndarray]] = []
    for position in sample_positions:
        ratio = min(0.98, max(0.0, float(position)))
        timestamp = min(duration * ratio, max(0.0, duration - 0.05))
        frame = extract_video_frame(video_path, timestamp)
        signatures.append((image_dhash(frame, hash_size=hash_size), image_preview(frame)))
    return tuple(signatures)


def render_video_variant(command: list[str]) -> tuple[bool, str]:
    result = run_command(command)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def process_videos(
    inputs: list[Path],
    output_dir: Path,
    logo_path: Path | None,
    config: dict[str, Any],
    variant_count: int,
    rng: random.Random,
    dry_run: bool,
) -> dict[Path, bool]:
    ensure_tool_exists("ffmpeg")
    ensure_tool_exists("ffprobe")

    video_cfg = config["video"]
    similarity_cfg = video_cfg.get("similarity", {})
    sample_positions = [float(item) for item in similarity_cfg.get("sample_positions", [0.5])]
    hash_size = int(similarity_cfg.get("dhash_hash_size", 8))
    min_distance = float(
        similarity_cfg.get(
            "min_average_distance",
            similarity_cfg.get("min_average_dhash_distance", 0.0),
        )
    )
    use_similarity = bool(similarity_cfg.get("enabled", False))
    max_attempts = int(video_cfg.get("max_attempts_per_variant", 1))
    transforms = video_cfg["transforms"]
    recolor_cfg = transforms.get("same_family_recolor", {})
    results: dict[Path, bool] = {}

    for input_path in inputs:
        print(f"\n[视频] 处理文件：{input_path}")
        results[input_path] = False
        try:
            info = get_video_info(input_path)
            source_signature = video_signature(
                input_path,
                info["duration"],
                sample_positions,
                hash_size=hash_size,
            )
        except RuntimeError as exc:
            print(f"跳过，原因：{exc}")
            continue

        source_md5 = compute_file_md5(input_path) if not dry_run else ""
        accepted_signatures = [source_signature]

        for variant_index in range(1, variant_count + 1):
            accepted = False
            best_distance = -1.0
            best_output_path: Path | None = None
            best_signature: tuple[tuple[int, np.ndarray], ...] | None = None

            for attempt in range(1, max_attempts + 1):
                filters = build_video_filters(info["width"], info["height"], transforms, rng)
                recolor_spec = build_video_recolor_spec(
                    video_path=input_path,
                    duration=info["duration"],
                    cfg=recolor_cfg,
                    rng=rng,
                )
                logo_spec = pick_video_logo_spec(
                    info["width"],
                    info["height"],
                    logo_path,
                    transforms.get("logo", {}),
                    rng,
                )
                output_path = build_output_path(input_path, output_dir, "video_variant", variant_index, ".mp4")

                if dry_run:
                    print(f"  [dry-run] 变体 {variant_index:02d} 第 {attempt} 次命令预演：{output_path}")
                    accepted = True
                    break

                if recolor_spec is None:
                    command = build_video_command_from_filters(
                        input_path=input_path,
                        output_path=output_path,
                        filters=filters,
                        logo_spec=logo_spec,
                        transforms=transforms,
                        audio_mode="copy",
                    )
                    success, error = render_video_variant(command)
                    if not success:
                        command = build_video_command_from_filters(
                            input_path=input_path,
                            output_path=output_path,
                            filters=filters,
                            logo_spec=logo_spec,
                            transforms=transforms,
                            audio_mode="aac",
                        )
                        success, error = render_video_variant(command)
                else:
                    success, error = render_video_variant_with_recolor(
                        input_path=input_path,
                        output_path=output_path,
                        width=info["width"],
                        height=info["height"],
                        fps=info["fps"],
                        filters=filters,
                        recolor_spec=recolor_spec,
                        logo_spec=logo_spec,
                        transforms=transforms,
                        audio_mode="copy",
                    )
                    if not success:
                        success, error = render_video_variant_with_recolor(
                            input_path=input_path,
                            output_path=output_path,
                            width=info["width"],
                            height=info["height"],
                            fps=info["fps"],
                            filters=filters,
                            recolor_spec=recolor_spec,
                            logo_spec=logo_spec,
                            transforms=transforms,
                            audio_mode="aac",
                        )

                if not success:
                    if output_path.exists():
                        output_path.unlink()
                    print(f"  变体 {variant_index:02d} 第 {attempt} 次渲染失败：{error}")
                    continue

                if not output_path.exists():
                    print(
                        f"  变体 {variant_index:02d} 第 {attempt} 次渲染未产出文件，"
                        "已跳过本次候选。"
                    )
                    continue

                try:
                    candidate_signature = video_signature(
                        output_path,
                        info["duration"],
                        sample_positions,
                        hash_size=hash_size,
                    )
                except RuntimeError as exc:
                    if output_path.exists():
                        output_path.unlink()
                    print(f"  变体 {variant_index:02d} 第 {attempt} 次抽帧失败：{exc}")
                    continue

                distance = min(
                    average_video_distance(candidate_signature, existing)
                    for existing in accepted_signatures
                )
                is_new_best = distance > best_distance
                if is_new_best:
                    if (
                        best_output_path is not None
                        and best_output_path != output_path
                        and best_output_path.exists()
                    ):
                        best_output_path.unlink()
                    best_distance = distance
                    best_output_path = output_path
                    best_signature = candidate_signature

                if use_similarity and distance < min_distance:
                    print(
                        f"  变体 {variant_index:02d} 第 {attempt} 次尝试被过滤，"
                        f"平均综合距离 {distance:.2f} < {min_distance:.2f}"
                    )
                    if output_path.exists() and not is_new_best:
                        output_path.unlink()
                    continue

                if (
                    best_output_path is not None
                    and best_output_path != output_path
                    and best_output_path.exists()
                ):
                    best_output_path.unlink()
                    best_output_path = output_path
                    best_signature = candidate_signature

                if not dry_run:
                    output_md5 = compute_file_md5(output_path)
                    metrics = build_video_metrics(
                        source_path=input_path,
                        output_path=output_path,
                        source_signature=source_signature,
                        candidate_signature=candidate_signature,
                        source_md5=source_md5,
                        output_md5=output_md5,
                        sample_positions=sample_positions,
                        hash_size=hash_size,
                    )
                    append_metrics_record(output_dir, metrics)
                    print(
                        f"  已生成：{output_path}（平均综合距离：{distance:.2f}，"
                        f"MAD：{metrics['mad']:.2f}，平均Hash距离：{metrics['average_dhash_distance']:.2f}，"
                        f"MD5变化：{metrics['md5_changed']}，感知Hash变化：{metrics['dhash_changed']}）"
                    )
                else:
                    print(f"  已生成：{output_path}（平均综合距离：{distance:.2f}）")
                accepted_signatures.append(candidate_signature)
                results[input_path] = True
                accepted = True
                break

            if accepted:
                continue

            if best_output_path is not None and best_output_path.exists():
                if best_signature is not None:
                    accepted_signatures.append(best_signature)
                    if not dry_run:
                        output_md5 = compute_file_md5(best_output_path)
                        metrics = build_video_metrics(
                            source_path=input_path,
                            output_path=best_output_path,
                            source_signature=source_signature,
                            candidate_signature=best_signature,
                            source_md5=source_md5,
                            output_md5=output_md5,
                            sample_positions=sample_positions,
                            hash_size=hash_size,
                        )
                        append_metrics_record(output_dir, metrics)
                        print(
                            f"  变体 {variant_index:02d} 未达到相似度阈值，"
                            f"已保留最优候选：{best_output_path}（最佳距离：{best_distance:.2f}，"
                            f"MAD：{metrics['mad']:.2f}，平均Hash距离：{metrics['average_dhash_distance']:.2f}，"
                            f"MD5变化：{metrics['md5_changed']}，感知Hash变化：{metrics['dhash_changed']}）"
                        )
                    else:
                        print(
                            f"  变体 {variant_index:02d} 未达到相似度阈值，"
                            f"已保留最优候选：{best_output_path}（最佳距离：{best_distance:.2f}）"
                        )
                    results[input_path] = True
                continue

            print(f"  变体 {variant_index:02d} 生成失败。")

    return results


def resolve_variant_count(cli_value: int | None, config: dict[str, Any]) -> int:
    if cli_value is not None:
        return max(1, cli_value)
    return max(1, int(config.get("variant_count", 1)))


def list_ready_watch_materials(
    watch_dir: Path,
    file_ready_age_seconds: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    logos: list[Path] = []

    for path in watch_dir.rglob("*"):
        if path.name.startswith(".") or not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(watch_dir).parts):
            continue
        if not is_file_ready(path, file_ready_age_seconds):
            continue

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS and is_logo_file(path):
            logos.append(path)
        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(path)
        elif suffix in SUPPORTED_VIDEO_EXTENSIONS:
            videos.append(path)

    images.sort(key=lambda item: str(item).lower())
    videos.sort(key=lambda item: str(item).lower())
    logos.sort(key=lambda item: str(item).lower())
    return images, videos, logos


def process_watch_once(
    watch_dir: Path,
    output_dir: Path,
    logo_path: Path | None,
    state_file: Path,
    config: dict[str, Any],
    variant_count: int,
    rng: random.Random,
    file_ready_age_seconds: float,
    dry_run: bool,
) -> None:
    ensure_directory(watch_dir)
    ensure_directory(output_dir)
    state = load_state(state_file)
    image_files, video_files, logo_files = list_ready_watch_materials(watch_dir, file_ready_age_seconds)

    if not image_files and not video_files and not logo_files:
        announce(f"监听中：请把待处理素材放到 {watch_dir}")
        return

    resolved_logo_path = logo_path
    if resolved_logo_path is None:
        resolved_logo_path = pick_latest_logo(logo_files)

    pending_images = [
        path
        for path in image_files
        if state.get(str(path)) != file_signature(path)
    ]
    pending_videos = [
        path
        for path in video_files
        if state.get(str(path)) != file_signature(path)
    ]

    if not pending_images and not pending_videos:
        announce("当前没有新的待处理素材，继续监听。")
        return

    logo_message = resolved_logo_path.name if resolved_logo_path is not None else "无"
    announce(
        f"开始处理素材：图片 {len(pending_images)} 张，视频 {len(pending_videos)} 个，"
        f"logo：{logo_message}，输出目录：{output_dir}"
    )

    if pending_images:
        image_results = process_images(
            inputs=pending_images,
            output_dir=output_dir,
            logo_path=resolved_logo_path,
            config=config,
            variant_count=variant_count,
            rng=rng,
            dry_run=dry_run,
        )
        if not dry_run:
            for path, success in image_results.items():
                if success:
                    state[str(path)] = file_signature(path)

    if pending_videos:
        video_results = process_videos(
            inputs=pending_videos,
            output_dir=output_dir,
            logo_path=resolved_logo_path,
            config=config,
            variant_count=variant_count,
            rng=rng,
            dry_run=dry_run,
        )
        if not dry_run:
            for path, success in video_results.items():
                if success:
                    state[str(path)] = file_signature(path)

    if not dry_run:
        save_state(state_file, state)
    announce(f"本轮处理完成，结果已保存到：{output_dir}")


def run_watch_loop(
    watch_dir: Path,
    output_dir: Path,
    logo_path: Path | None,
    state_file: Path | None,
    config: dict[str, Any],
    variant_count: int,
    rng: random.Random,
    poll_interval_seconds: float,
    file_ready_age_seconds: float,
    dry_run: bool,
    once: bool,
) -> int:
    resolved_state_file = resolve_state_file(state_file, watch_dir)

    if logo_path is not None and not logo_path.exists():
        print(f"logo 文件不存在：{logo_path}", file=sys.stderr)
        return 1

    print(f"自动监听目录：{watch_dir}")
    print(f"生成结果目录：{output_dir}")
    print(f"状态文件：{resolved_state_file}")
    print("按 Ctrl+C 可停止监听。")

    while True:
        try:
            process_watch_once(
                watch_dir=watch_dir,
                output_dir=output_dir,
                logo_path=logo_path,
                state_file=resolved_state_file,
                config=config,
                variant_count=variant_count,
                rng=rng,
                file_ready_age_seconds=file_ready_age_seconds,
                dry_run=dry_run,
            )

            if once:
                return 0
            time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            print("\n已停止监听。")
            return 0
        except Exception as exc:
            print(f"处理时出现异常：{exc}")
            if once:
                return 1
            time.sleep(poll_interval_seconds)


def default_config_path() -> Path:
    return Path(__file__).with_name("configs").joinpath("default_variant_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="素材裂变原型：支持图片与视频的配置化变体生成。")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="JSON 配置文件路径。默认读取仓库内的 configs/default_variant_config.json",
    )
    parser.add_argument("--count", type=int, help="覆盖配置中的变体数量。")
    parser.add_argument("--seed", type=int, help="固定随机种子，便于复现。")
    parser.add_argument("--dry-run", action="store_true", help="只演示流程，不落盘文件。")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    image_parser = subparsers.add_parser("image", help="处理图片素材。")
    image_parser.add_argument("--input", type=Path, required=True, help="单张图片或图片目录。")
    image_parser.add_argument("--output-dir", type=Path, required=True, help="输出目录。")
    image_parser.add_argument("--logo", type=Path, help="可选 logo 路径。")

    video_parser = subparsers.add_parser("video", help="处理视频素材。")
    video_parser.add_argument("--input", type=Path, required=True, help="单个视频或视频目录。")
    video_parser.add_argument("--output-dir", type=Path, required=True, help="输出目录。")
    video_parser.add_argument("--logo", type=Path, help="可选 logo 路径。")

    watch_parser = subparsers.add_parser("watch", help="统一监听目录并自动处理图片/视频素材。")
    watch_parser.add_argument(
        "--watch-dir",
        type=Path,
        default=DEFAULT_MATERIAL_WATCH_DIR,
        help=f"监听目录，默认：{DEFAULT_MATERIAL_WATCH_DIR}",
    )
    watch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MATERIAL_OUTPUT_DIR,
        help=f"输出目录，默认：{DEFAULT_MATERIAL_OUTPUT_DIR}",
    )
    watch_parser.add_argument("--logo", type=Path, help="可选固定 logo 路径；不传则从监听目录自动识别。")
    watch_parser.add_argument("--state-file", type=Path, help="可选状态文件路径。")
    watch_parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="轮询间隔秒数。",
    )
    watch_parser.add_argument(
        "--file-ready-age",
        type=float,
        default=DEFAULT_FILE_READY_AGE_SECONDS,
        help="文件静置多久后才开始处理，单位秒。",
    )
    watch_parser.add_argument("--once", action="store_true", help="只扫描处理一轮后退出。")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config
    if config_path and not config_path.exists():
        print(f"配置文件不存在：{config_path}", file=sys.stderr)
        return 1

    try:
        config = load_config(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"配置文件读取失败：{exc}", file=sys.stderr)
        return 1

    variant_count = resolve_variant_count(args.count, config)
    rng = random.Random(args.seed)

    if args.mode == "image":
        output_dir = args.output_dir
        ensure_directory(output_dir)
        logo_path = args.logo
        if logo_path is not None and not logo_path.exists():
            print(f"logo 文件不存在：{logo_path}", file=sys.stderr)
            return 1
        inputs = collect_files(args.input, SUPPORTED_IMAGE_EXTENSIONS)
        if not inputs:
            print("没有找到可处理的图片文件。", file=sys.stderr)
            return 1
        process_images(inputs, output_dir, logo_path, config, variant_count, rng, args.dry_run)
        return 0

    if args.mode == "video":
        output_dir = args.output_dir
        ensure_directory(output_dir)
        logo_path = args.logo
        if logo_path is not None and not logo_path.exists():
            print(f"logo 文件不存在：{logo_path}", file=sys.stderr)
            return 1
        inputs = collect_files(args.input, SUPPORTED_VIDEO_EXTENSIONS)
        if not inputs:
            print("没有找到可处理的视频文件。", file=sys.stderr)
            return 1
        process_videos(inputs, output_dir, logo_path, config, variant_count, rng, args.dry_run)
        return 0

    if args.mode == "watch":
        return run_watch_loop(
            watch_dir=args.watch_dir,
            output_dir=args.output_dir,
            logo_path=args.logo,
            state_file=args.state_file,
            config=config,
            variant_count=variant_count,
            rng=rng,
            poll_interval_seconds=float(args.poll_interval),
            file_ready_age_seconds=float(args.file_ready_age),
            dry_run=args.dry_run,
            once=bool(args.once),
        )

    print("未识别的模式。", file=sys.stderr)
    return 1

def process_image():
    try:
        intput = Path("E:\\Documents\\海外推广\\image")
        logo_path = Path("E:\\Documents\\海外推广\\image\\logo-small.2f8f2cf2.png")
        output_dir = Path("E:\\Documents\\海外推广\\image_variants")
        rng = random.Random(None)
        ensure_directory(output_dir)
        if logo_path is not None and not logo_path.exists():
            print(f"logo 文件不存在：{logo_path}", file=sys.stderr)
            return 1
        inputs = collect_files(intput, SUPPORTED_IMAGE_EXTENSIONS)
        if not inputs:
            print("没有找到可处理的图片文件。", file=sys.stderr)
            return 1
        process_images(inputs, output_dir, logo_path, DEFAULT_CONFIG, 3, rng, False)
    except Exception as e:
        # 捕获所有异常并抛出
        raise RuntimeError(f"处理图片时出错: {e}") from e

if __name__ == "__main__":
    raise process_image()

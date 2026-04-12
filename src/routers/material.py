"""
素材裂变 API 路由

提供图片/视频变体生成的 HTTP 接口。
自动识别上传文件的类型（图片/视频），并分发到对应处理流程。
"""

import asyncio
import random
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.service.material import (
    DEFAULT_CONFIG,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    ensure_directory,
    load_config,
    process_images,
    process_videos,
)

router = APIRouter(prefix="/material", tags=["material"])

# 输出目录
_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
_VARIANTS_DIR = _OUTPUT_DIR / "variants"
_TMP_UPLOAD_DIR = _OUTPUT_DIR / "tmp_uploads"

# 文件大小限制 (字节)
_IMAGE_MAX_SIZE = 50 * 1024 * 1024       # 50 MB
_VIDEO_MAX_SIZE = 500 * 1024 * 1024      # 500 MB


# ---------- 请求/响应模型 ----------

class TaskResponse(BaseModel):
    task_id: str
    message: str
    file_type_summary: dict[str, int] = Field(
        default_factory=dict,
        description="各类型文件数量，如 {'image': 2, 'video': 1}",
    )


class VariantResultItem(BaseModel):
    original: str
    output: list[str] | None = None
    error: str | None = None
    dhash_distance: float | None = None
    mad: float | None = None
    download_urls: list[str] | None = None


class SyncVariantResponse(BaseModel):
    variant_count: int
    processed: int
    file_type_summary: dict[str, int]
    results: list[VariantResultItem]


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # pending | processing | completed | failed
    result: list[VariantResultItem] | None = None
    error: str | None = None


# ---------- 内存任务存储 ----------

_tasks: dict[str, dict[str, Any]] = {}


# ---------- 辅助函数 ----------

def _classify_file(ext: str) -> str | None:
    """根据扩展名判断文件类型。返回 'image' / 'video' / None。"""
    ext = ext.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    return None


def _build_result_item(original_name: str, result: dict[str, Any]) -> VariantResultItem:
    """将 process_images / process_videos 返回的单条结果统一为 VariantResultItem。"""
    if "error" in result:
        return VariantResultItem(original=original_name, error=result["error"])

    raw_output = result.get("output")

    # 🔥 关键修复：统一转成 list
    if isinstance(raw_output, str):
        output_files = [raw_output]
    elif isinstance(raw_output, list):
        output_files = raw_output
    else:
        output_files = []

    download_urls = (
        [f"/material/download/{f}" for f in output_files]
        if output_files else None
    )

    return VariantResultItem(
        original=original_name,
        output=output_files or None,
        dhash_distance=result.get("dhash_distance"),
        mad=result.get("mad"),
        download_urls=download_urls,
    )


# ---------- API ----------

@router.get("/config")
async def get_config():
    """获取当前默认配置。"""
    return {"config": DEFAULT_CONFIG}


@router.post("/variants", response_model=SyncVariantResponse)
async def create_variants(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    variant_count: int = 3,
    seed: int | None = None,
):
    """
    统一素材变体接口 —— 自动识别图片/视频并做相应处理。

    - **纯图片上传**：同步处理，直接返回变体结果。
    - **包含视频（或混合）**：异步处理，返回 task_id 供轮询。

    - files: 素材文件列表（图片或视频）
    - variant_count: 每个素材生成几个变体
    - seed: 随机种子（可选）
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")

    # ---------- 1. 校验 & 保存上传文件 ----------
    ensure_directory(_TMP_UPLOAD_DIR)

    image_paths: list[tuple[Path, str]] = []   # (路径, 原始文件名)
    video_paths: list[tuple[Path, str]] = []   # (路径, 原始文件名)
    all_paths: list[Path] = []                 # 用于清理

    for file in files:
        if file.filename is None:
            continue

        ext = Path(file.filename).suffix.lower()
        file_type = _classify_file(ext)

        if file_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {ext}，"
                       f"图片支持 {SUPPORTED_IMAGE_EXTENSIONS}，"
                       f"视频支持 {SUPPORTED_VIDEO_EXTENSIONS}",
            )

        content = await file.read()
        max_size = _IMAGE_MAX_SIZE if file_type == "image" else _VIDEO_MAX_SIZE
        size_label = "50MB" if file_type == "image" else "500MB"

        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件 {file.filename} 超过 {size_label} 限制",
            )

        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = _TMP_UPLOAD_DIR / unique_name
        save_path.write_bytes(content)
        all_paths.append(save_path)

        if file_type == "image":
            image_paths.append((save_path, file.filename))
        else:
            video_paths.append((save_path, file.filename))

    file_type_summary: dict[str, int] = {}
    if image_paths:
        file_type_summary["image"] = len(image_paths)
    if video_paths:
        file_type_summary["video"] = len(video_paths)

    # ---------- 2. 纯图片 → 同步处理 ----------
    if not video_paths:
        ensure_directory(_VARIANTS_DIR)
        config = load_config(None)
        rng = random.Random(seed)

        try:
            input_paths = [p for p, _ in image_paths]
            raw_results = process_images(
                inputs=input_paths,
                output_dir=_VARIANTS_DIR,
                logo_path=None,
                config=config,
                variant_count=variant_count,
                rng=rng,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")
        finally:
            for p in all_paths:
                if p.exists():
                    p.unlink()

        items = [
            _build_result_item(
                original_name=image_paths[i][1] if i < len(image_paths) else "unknown",
                result=r,
            )
            for i, r in enumerate(raw_results)
        ]

        return SyncVariantResponse(
            variant_count=variant_count,
            processed=len([r for r in raw_results if "error" not in r]),
            file_type_summary=file_type_summary,
            results=items,
        )

    # ---------- 3. 包含视频 → 异步处理 ----------
    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {
        "status": "pending",
        "image_paths": [(str(p), name) for p, name in image_paths],
        "video_paths": [(str(p), name) for p, name in video_paths],
        "result": None,
        "error": None,
    }

    async def _background_process():
        _tasks[task_id]["status"] = "processing"
        combined_results: list[dict[str, Any]] = []

        try:
            config = load_config(None)
            rng = random.Random(seed)
            ensure_directory(_VARIANTS_DIR)

            # 处理图片
            if image_paths:
                img_input = [p for p, _ in image_paths]
                img_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: process_images(
                        inputs=img_input,
                        output_dir=_VARIANTS_DIR,
                        logo_path=None,
                        config=config,
                        variant_count=variant_count,
                        rng=rng,
                    ),
                )
                for i, r in enumerate(img_results):
                    combined_results.append(
                        _build_result_item(
                            original_name=image_paths[i][1] if i < len(image_paths) else "unknown",
                            result=r,
                        ).model_dump(),
                    )

            # 处理视频
            if video_paths:
                vid_input = [p for p, _ in video_paths]
                vid_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: process_videos(
                        inputs=vid_input,
                        output_dir=_VARIANTS_DIR,
                        logo_path=None,
                        config=config,
                        variant_count=variant_count,
                        rng=rng,
                    ),
                )
                for i, r in enumerate(vid_results):
                    combined_results.append(
                        _build_result_item(
                            original_name=video_paths[i][1] if i < len(video_paths) else "unknown",
                            result=r,
                        ).model_dump(),
                    )

            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["result"] = combined_results

        except Exception as e:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(e)

        finally:
            for p in all_paths:
                if p.exists():
                    p.unlink()

    asyncio.create_task(_background_process())

    return TaskResponse(
        task_id=task_id,
        message="处理已提交，请通过 /material/tasks/{task_id} 查询进度",
        file_type_summary=file_type_summary,
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询异步处理任务状态（视频 / 混合素材）。"""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error"),
    )


@router.get("/download/{filename}")
async def download_variant(filename: str):
    """下载生成的变体文件。"""
    file_path = _VARIANTS_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")


@router.get("/files")
async def list_variant_files():
    """列出所有已生成的变体文件。"""
    ensure_directory(_VARIANTS_DIR)
    files = []
    for f in sorted(_VARIANTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if f.is_file():
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            suffix = f.suffix.lower()
            file_type = (
                "image" if suffix in {".jpg", ".jpeg", ".png", ".webp"}
                else "video" if suffix in {".mp4", ".webm", ".mov"}
                else "other"
            )
            files.append({
                "name": f.name,
                "size_mb": size_mb,
                "type": file_type,
                "url": f"/material/download/{f.name}",
            })
    return {"files": files}

"""
物品路由
"""

import uuid
import time
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.requests import Request
from src.core.response import R
from src.service.materials import process_image_video, reader_file_json
from pathlib import Path
from pydantic import BaseModel, Field
from src.utils.img_utils import type_file, SUPPORTED_VIDEO_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS
from src.core.config import BASE_DIR, config
from src.utils.dir_utils import ensure_directory


router = APIRouter(prefix="/page", tags=["page"])

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

@router.get("")
async def home():
    return RedirectResponse(url=f"/page/upload?uid={uuid.uuid4()}", status_code=302)

@router.get("/upload", response_class=HTMLResponse)
async def upload_page(uid: str):
    """返回上传页面 HTML。"""
    html_path = BASE_DIR / "templates" / "upload.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@router.get("/variants")
async def create_variants(request: Request):
    # 获取单个 header
    headers = request.headers
    uid = request.headers.get("uid")
    result = reader_file_json(uid)
    return R.ok(result)




@router.post("/variants")
async def create_variants(request: Request, files: list[UploadFile]):
    """
    统一素材变体接口 —— 自动识别图片/视频并做相应处理。

    - **纯图片上传**：同步处理，直接返回变体结果。
    - **包含视频（或混合）**：异步处理，返回 task_id 供轮询。

    - files: 素材文件列表（图片或视频）
    - variant_count: 每个素材生成几个变体
    - seed: 随机种子（可选）
    """
    headers = request.headers
    # 获取单个 header
    uid = headers.get("uid")

    if not uid or not uid.strip():
        return R.fail("请提供窗口uid")

    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")

    file_paths: list[tuple[Path, str]] = []   # (路径, 原始文件名)
    all_paths: list[Path] = []                 # 用于清理

    tmp_upload_dir = config.FILE_ADDR_PATH / "output" / uid / "tmp_uploads" / datetime.now().strftime("%Y-%m-%d")

    # 确保路径存在
    ensure_directory(tmp_upload_dir)

    for file in files:
        if file.filename is None:
            continue

        ext = Path(file.filename).suffix.lower()
        file_type = type_file(ext)

        if file_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {ext}，"
                       f"图片支持 {SUPPORTED_IMAGE_EXTENSIONS}，"  # noqa: F821
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
        temp_name = int(time.time() * 1000)
        unique_name = f"{temp_name}{ext}"
        save_path = tmp_upload_dir / unique_name
        save_path.write_bytes(content)
        all_paths.append(save_path)
        file_paths.append((save_path, file.filename))

    if not file_paths:
        raise HTTPException(status_code=400, detail=f"未找到需要处理的文件")

    if file_paths:
        raw_results = process_image_video(path_name_list=file_paths, uid=uid, seed = 3)
        return R.ok(raw_results)

    return R.ok()







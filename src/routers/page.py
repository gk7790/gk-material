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
from src.utils.sse_writer import sse
import time
import asyncio
import json

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

@router.get("/stream")
async def stream(uid: str, request: Request):
    channel = f"user:{uid}"
    return await sse.stream(channel, request)

@router.post("/variants")
async def create_variants(request: Request):
    """文件上传并处理 - SSE 流式返回"""
    form = await request.form()
    files = form.getlist("files")

    uid = request.headers.get("uid")

    # ❌ 不再创建 event_generator
    # ❌ 不再 create_task

    asyncio.create_task(handle_upload(uid, files))

    return R.ok(uid)

async def handle_upload(uid: str, files):

    channel = f"user:{uid}"
    try:
        # =========================
        # 校验
        # =========================
        if not uid:
            sse.error(channel, "缺少 uid")
            return

        if not files:
            sse.error(channel, "请上传文件")
            return

        sse.message(channel, f"收到 {len(files)} 个文件")

        # =========================
        # 准备路径
        # =========================
        tmp_upload_dir = Path(f"./tmp/{uid}/{datetime.now().date()}")
        tmp_upload_dir.mkdir(parents=True, exist_ok=True)

        file_paths = []

        # =========================
        # 文件处理
        # =========================
        for idx, file in enumerate(files, 1):

            if not file.filename:
                continue

            content = await file.read()

            save_path = tmp_upload_dir / f"{int(time.time() * 1000)}_{file.filename}"
            save_path.write_bytes(content)

            file_paths.append((save_path, file.filename))

            # ⭐ 进度
            sse.message(channel, f"正在上传数据 {file.filename}")

        if not file_paths:
            sse.success(channel, "没有可处理文件")
            return

        # =========================
        # 核心处理逻辑
        # =========================
        sse.success(channel, "开始处理图片/视频")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            process_image_video,
            file_paths,
            uid,
            3
        )
        # =========================
        # 完成
        # =========================
        sse.doen(channel, "处理完成")

    except Exception as e:
        sse.error(channel, f"系统错误: {str(e)}")
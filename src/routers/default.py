"""
默认路由
"""

from fastapi import APIRouter

from src.core.config import config
from src.core.models import HealthResponse

router = APIRouter(tags=["default"])

@router.get("/")
async def root():
    """根路径，返回欢迎信息。"""
    return {"message": f"欢迎使用 {config.APP_NAME}！请访问 /docs 查看 API 文档。"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        app_name=config.APP_NAME,
        env=config.APP_ENV,
    )

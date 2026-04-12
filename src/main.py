"""
FastAPI 应用入口
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.core.config import config
from contextlib import asynccontextmanager
from src.routers import default, material, page
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# 应用生命周期
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print(f"🚀 {config.APP_NAME} 已启动 | env={config.APP_ENV} | debug={config.DEBUG}")
    print("🚀 应用已启动")
    yield
    # 关闭时执行
    print("👋 应用已关闭")


# --------------------------------------------------
# 创建 FastAPI 实例
# --------------------------------------------------
app = FastAPI(
    title=config.APP_NAME,
    debug=config.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 挂载路由
# --------------------------------------------------

app.include_router(default.router)
app.include_router(page.router)
app.include_router(material.router)

app.mount("/static", StaticFiles(directory=f"{config.FILE_ADDR}"), name="static")

# --------------------------------------------------
# 直接运行: uv run src/main.py
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.is_dev,
    )

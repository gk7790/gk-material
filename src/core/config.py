"""
应用配置模块 —— 通过 python-dotenv 加载 .env 环境变量
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：src 的上级
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env 文件（优先项目根目录，其次当前工作目录）
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(".env", override=False)

class Settings:
    """应用配置，所有值均从环境变量读取，提供默认值作为兜底。"""

    # ---------- 应用基础 ----------
    APP_NAME: str = os.getenv("APP_NAME", "FastAPI Demo")
    APP_ENV: str = os.getenv("APP_ENV", "production")          # development | production
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    FILE_ADDR: str = os.getenv("FILE_ADDR", "C:/material")

    FILE_ADDR_PATH: str = Path(FILE_ADDR)
    # ---------- 服务 ----------
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    BASE_HOST: str = f"http://{HOST}:{PORT}"

    @property
    def is_dev(self) -> bool:
        return self.APP_ENV == "development"


config = Settings()

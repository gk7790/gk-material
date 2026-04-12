"""
数据模型 —— 所有 Pydantic 模型集中管理
"""

from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    env: str

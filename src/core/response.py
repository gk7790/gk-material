from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

T = TypeVar("T")

app = FastAPI()
router = APIRouter()

# ========================
# 1. 统一响应模型（放在最前面）
# ========================
class R(BaseModel, Generic[T]):
    code: int = 0
    msg: str = "success"
    data: Optional[T] = None

    @classmethod
    def ok(cls, data: Any = None, msg: str = "success") -> "R":
        """成功响应"""
        return cls(code=0, msg=msg, data=data)

    @classmethod
    def fail(cls, msg: str = "fail", data: Any = None) -> "R":
        """失败响应"""
        return cls(code=3, msg=msg, data=data)

# ========================
# 2. 自定义业务异常
# ========================
class BizException(Exception):
    def __init__(self, msg: str = "业务异常", code: int = 400, data: Any = None):
        self.msg = msg
        self.code = code
        self.data = data

# ========================
# 3. 全局异常处理器（依赖 R）
# ========================
@app.exception_handler(BizException)
async def biz_exception_handler(request: Request, exc: BizException):
    return JSONResponse(
        status_code=200,
        content=R.fail(msg=exc.msg, code=exc.code, data=exc.data).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=200,
        content=R.fail(msg="参数校验失败", code=422, data=exc.errors()).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=R.fail(msg="服务器内部错误", code=500).model_dump(),
    )
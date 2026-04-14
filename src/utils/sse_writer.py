import asyncio
import time
from fastapi import Request
from sse_starlette.sse import EventSourceResponse
from collections import defaultdict
from typing import Any
import math
import json
import threading


class Progress:
    def __init__(self, channel, total):
        self.channel = channel
        self.total = total
        self.current = 0
        self.lock = threading.Lock()

    def step(self, step: float = 1):
        with self.lock:
            curr = self.current + step
            self.current = min(curr, self.total)
            return self.channel, self.current, self.total

    def cover(self, step: float):
        with self.lock:
            self.current = min(step, self.total)
            return self.channel, self.current, self.total

    def ceil(self):
        with self.lock:
            curr = math.ceil(self.current)
            self.current = min(curr, self.total)
            return self.channel, self.current, self.total

    def channel(self):
        return self.channel


# =========================
# SSE Manager（核心）
# =========================
class SSEManager:

    def __init__(self):
        self.channels: dict[str, asyncio.Queue] = {}

    # -------------------------
    # subscribe
    # -------------------------
    def subscribe(self, channel: str):
        # 替换旧连接
        if channel in self.channels:
            try:
                old = self.channels[channel]
                old.put_nowait({"event": "close", "data": "replaced"})
            except Exception:
                pass
        q = asyncio.Queue(maxsize=200)
        self.channels[channel] = q
        return q

    def unsubscribe(self, channel: str):
        self.channels.pop(channel, None)

    # -------------------------
    # ⭐ 底层发送
    # -------------------------
    def emit(self, channel: str, event: str, data: Any):
        if channel not in self.channels:
            return
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False)

        try:
            self.channels[channel].put_nowait({
                "event": event,
                "data": data
            })
        except asyncio.QueueFull:
            # 防止阻塞
            pass

    # =========================
    # ⭐ 语义化 API（重点）
    # =========================
    def message(self, channel: str, msg: str):
        self.emit(channel, "message", msg)

    def progress(self, channel: str, current: float, total: float, msg: str = ""):
        percent = 0 if total == 0 else round(current / total * 100, 2)
        self.emit(channel, "progress", {
            "current": round(current, 2),
            "total": total,
            "percentage": percent,
            "message": msg
        })

    def info(self, channel: str, msg: str):
        self.emit(channel, "info", msg)

    def success(self, channel: str, data: Any):
        self.emit(channel, "success", data)

    def warning(self, channel: str, msg: str):
        self.emit(channel, "warning", msg)

    def fail(self, channel: str, msg: str):
        self.emit(channel, "fail", msg)

    def error(self, channel: str, msg: str):
        self.emit(channel, "error", msg)

    def close(self, channel: str, msg: str = "done"):
        self.emit(channel, "close", msg)

    def doen(self, channel: str, msg: str = "done", total: int = 100):
        self.emit(channel, "progress", {
            "current": total,
            "total": total,
            "percentage": 100,
            "message": msg
        })
        self.emit(channel, "close", msg)

    # -------------------------
    # SSE stream
    # -------------------------

    async def stream(self, channel: str, request: Request):

        q = self.subscribe(channel)

        async def gen():
            try:
                yield {"event": "connected", "data": "ok"}

                idle = 0
                max_idle = 5

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15)
                        idle = 0
                        if msg["event"] == "close":
                            yield msg
                            break
                        yield msg
                    except asyncio.TimeoutError:
                        if channel not in self.channels:
                            idle += 1
                        yield {"event": "ping", "data": "keep-alive"}

                        # channel 不存在才计 idle（你要的逻辑）
                        if channel not in self.channels and idle >= max_idle:
                            yield {"event": "close", "data": "timeout"}
                            break

            except Exception as e:
                yield {"event": "error", "data": str(e)}
            finally:
                self.unsubscribe(channel)

        return EventSourceResponse(gen())


# 全局实例
sse = SSEManager()

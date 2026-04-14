"""
线程池 + 任务队列 + 顺序保证 + 返回值 + 错误隔离（生产级）
- 线程安全：submit / run / run_stream 均加锁
- 单次执行：run / run_stream 执行后会自动快照并清空任务队列
- 高并发友好：每次执行取走任务快照，不影响后续 submit
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, List, Dict, Generator, Optional
import threading
import time


class TaskExecutor:

    def __init__(self, max_workers: int = 5, retries: int = 0, timeout: Optional[float] = None):
        self.max_workers = max_workers
        self.retries = retries
        self.timeout = timeout
        self._tasks: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def submit(self, func: Callable, *args, **kwargs):
        """
        添加任务（按顺序，线程安全）
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")
        with self._lock:
            self._tasks.append({
                "func": func,
                "args": args,
                "kwargs": kwargs,
            })

    def _snapshot_and_clear(self) -> List[Dict[str, Any]]:
        """
        原子操作：取出当前所有任务并清空队列。
        后续 submit 的任务不会混入本次执行。
        """
        with self._lock:
            tasks = list(self._tasks)
            self._tasks.clear()
        return tasks

    def _run_with_retry(self, func: Callable, args: tuple, kwargs: dict):
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))  # 递增退避
        raise last_exc  # type: ignore

    # ------------------------------------------------------------------
    # run: 同步批量执行，返回全部结果
    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        """
        执行当前所有任务并返回结构化结果。
        执行后会清空任务队列，重复调用不会重复执行。
        """
        tasks = self._snapshot_and_clear()
        if not tasks:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self._run_with_retry, t["func"], t["args"], t["kwargs"]): idx
                for idx, t in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result(timeout=self.timeout)
                    results[idx] = {"success": True, "result": res}
                except Exception as e:
                    results[idx] = {"success": False, "error": str(e)}

        return results

    # ------------------------------------------------------------------
    # run_stream: 流式返回（一个任务完成就 yield）
    # ------------------------------------------------------------------
    def run_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        流式执行：每完成一个任务就 yield 一次。
        执行后会清空任务队列。
        """
        tasks = self._snapshot_and_clear()
        if not tasks:
            return

        total = len(tasks)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self._run_with_retry, t["func"], t["args"], t["kwargs"]): idx
                for idx, t in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                try:
                    res = future.result(timeout=self.timeout)
                    yield {
                        "index": idx,
                        "success": True,
                        "result": res,
                        "progress": round(completed / total, 4),
                        "completed": completed,
                        "total": total,
                    }
                except Exception as e:
                    yield {
                        "index": idx,
                        "success": False,
                        "error": str(e),
                        "progress": round(completed / total, 4),
                        "completed": completed,
                        "total": total,
                    }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def task_count(self) -> int:
        """返回当前队列中待执行的任务数（线程安全）"""
        with self._lock:
            return len(self._tasks)

    def clear(self):
        """手动清空任务队列（线程安全）"""
        with self._lock:
            self._tasks.clear()

    def __len__(self) -> int:
        return self.task_count()


# ======================================================================
# 线程安全单例
# ======================================================================
class _TaskSingle:
    """内部单例，延迟初始化，线程安全"""
    _instance: Optional["TaskExecutor"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> TaskExecutor:
        if cls._instance is None:
            with cls._lock:
                # double-check
                if cls._instance is None:
                    cls._instance = TaskExecutor(
                        max_workers=min((os.cpu_count() or 1) * 2, 120),
                        retries=1,
                    )
        return cls._instance

    @classmethod
    def reset(cls, executor: TaskExecutor):
        """测试用：替换实例"""
        with cls._lock:
            cls._instance = executor


# 对外暴露的便捷函数
def init_executor(max_workers: Optional[int] = None):
    cpu = os.cpu_count() or 1
    if max_workers is None:
        max_workers = cpu * 2
    max_workers = min(max_workers, 120)
    _TaskSingle.reset(TaskExecutor(max_workers=max_workers, retries=1))


def get_executor() -> TaskExecutor:
    return _TaskSingle.get()

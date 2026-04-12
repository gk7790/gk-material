"""
线程池 + 任务队列 + 顺序保证 + 返回值 + 错误隔离（生产级）
可配置线程池
顺序提交任务
并发执行
等待全部完成
获取返回值（且可区分成功/失败）
"""


from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, List, Dict
import time


class TaskExecutor:

    def __init__(self, max_workers=5, retries=0, timeout=None):
        self.max_workers = max_workers
        self.retries = retries
        self.timeout = timeout
        self.tasks = []

    def submit(self, func: Callable, *args, **kwargs):
        """
        添加任务（按顺序）
        """
        self.tasks.append({
            "func": func,
            "args": args,
            "kwargs": kwargs
        })

    def _run_with_retry(self, func, args, kwargs):
        for i in range(self.retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == self.retries:
                    raise
                time.sleep(0.5)

    def run(self) -> List[Dict[str, Any]]:
        """
        执行所有任务，返回结构化结果
        """
        results = [None] * len(self.tasks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(
                    self._run_with_retry,
                    task["func"],
                    task["args"],
                    task["kwargs"]
                ): idx
                for idx, task in enumerate(self.tasks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    res = future.result(timeout=self.timeout)
                    results[idx] = {"success": True, "result": res}
                except Exception as e:
                    results[idx] = {"success": False, "error": str(e)}
        return results

    def clear(self):
        self.tasks.clear()

from src.utils.task_utils import TaskExecutor
import os

class TaskSingle:
    executor = None

def init_executor(max_workers=None):
    cpu = os.cpu_count() or 1
    if max_workers is None:
        max_workers = cpu * 2
    max_workers = min(max_workers, 120)
    TaskSingle.executor = TaskExecutor(max_workers=max_workers, retries=1)

def get_executor():
    return TaskSingle.executor


import time
from src.utils.task_utils import TaskExecutor

def task1(x):
    time.sleep(1)
    return x * 10





# --------------------------------------------------
if __name__ == "__main__":
    executor = TaskExecutor(max_workers=3, retries=1)

    for i in range(5):
        executor.submit(task1, i)

    results = executor.run()

    for r in results:
        if r["success"]:
            print("成功:", r["result"])
        else:
            print("失败:", r["error"])


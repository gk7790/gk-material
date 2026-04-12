
from pathlib import Path
import uuid
import shutil

# 确保目录存在
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_text(self, *paths, content: str, encoding="utf-8") -> Path:
    """
    写入文本
    """
    file_path = self.path(*paths)
    self.ensure_parent(*paths)
    file_path.write_text(content, encoding=encoding)
    return file_path

def read_text(self, *paths, encoding="utf-8") -> str:
    """
    读取文本
    """
    file_path = self.path(*paths)
    return file_path.read_text(encoding=encoding)


def delete(self, *paths) -> None:
    """
    删除文件或目录
    """
    target = self.path(*paths)
    if target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)

def exists(self, *paths) -> bool:
    return self.path(*paths).exists()


def build_output_path(source_path: Path, output_dir: Path, label: str, index: int, suffix: str) -> Path:
    """
    基于输入文件路径生成一个“唯一且结构化”的输出文件路径
    {原文件名}_{标签}_{序号}_{随机串}.{后缀}
    source_path.stem 原文件名（不带后缀）
    :param source_path: 原始文件路径
    :param output_dir: 输出目录
    :param label: 标记用途（通常是处理类型）
    :param index: 当前生成的第几个变体/结果
    :param suffix: 文件后缀（必须带点）
    :return: 拼接文件名
    """
    unique_marker = uuid.uuid4().hex[:6]
    return output_dir / f"{source_path.stem}_{label}_{index:02d}_{unique_marker}{suffix}"

def build_url_with_base(
    file_path: str | Path,
    base_dir: str | Path,
    base_url: str,
    url_prefix: str = "static",
) -> str:
    try:
        relative_path = Path(file_path).resolve().relative_to(Path(base_dir).resolve()).as_posix()
    except ValueError:
        # 路径不在 base_dir 下，直接用文件名兜底
        relative_path = Path(file_path).name
    return f"{base_url.rstrip('/')}/{url_prefix.strip('/')}/{relative_path}"



"""本模块作用：在整个智能体中负责发现和加载本地 PDF 论文文件，为后续解析与切块提供输入。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PAPERS_DIR = PROJECT_ROOT / "data" / "raw_papers"


def get_raw_papers_dir(raw_papers_dir: str | Path | None = None) -> Path:
    """返回原始论文目录路径。

    输入：
        raw_papers_dir: 可选，自定义的论文目录路径。
    输出：
        标准化后的目录路径对象。
    异常：
        当路径字符串非法时，可能抛出 OSError。
    """

    if raw_papers_dir is None:
        return DEFAULT_RAW_PAPERS_DIR
    return Path(raw_papers_dir).expanduser().resolve()


def ensure_raw_papers_dir(raw_papers_dir: str | Path | None = None) -> Path:
    """确保原始论文目录存在。

    输入：
        raw_papers_dir: 可选，自定义的论文目录路径。
    输出：
        已确保存在的目录路径对象。
    异常：
        当目录创建失败时，抛出 OSError。
    """

    target_dir = get_raw_papers_dir(raw_papers_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def list_pdf_files(
    raw_papers_dir: str | Path | None = None,
    recursive: bool = False,
) -> list[Path]:
    """列出原始论文目录中的 PDF 文件。

    输入：
        raw_papers_dir: 可选，自定义的论文目录路径。
        recursive: 是否递归扫描子目录。
    输出：
        按文件名排序后的 PDF 路径列表。
    异常：
        当目录创建失败或目录无法访问时，抛出 OSError。
    """

    target_dir = ensure_raw_papers_dir(raw_papers_dir)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = [path for path in target_dir.glob(pattern) if path.is_file()]
    return sorted(pdf_files, key=lambda item: item.name.lower())


def run_loader_demo() -> None:
    """执行 loader 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印目录和 PDF 扫描结果。
    异常：
        当目录创建失败时，抛出 OSError。
    """

    target_dir = ensure_raw_papers_dir()
    pdf_files = list_pdf_files(target_dir)

    print("Loader Demo")
    print(f"论文目录：{target_dir}")
    print(f"发现 PDF 数量：{len(pdf_files)}")

    if not pdf_files:
        print("提示：请将待处理的 PDF 放入 data/raw_papers/ 后再次运行。")
        return

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"{index}. {pdf_path.name}")


if __name__ == "__main__":
    run_loader_demo()

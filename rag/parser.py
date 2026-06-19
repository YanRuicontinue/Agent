"""本模块作用：在整个智能体中负责将本地 PDF 论文解析为统一文档结构，供后续切块与检索使用。"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path


# 下面这段处理用于兼容直接执行当前文件和以模块方式导入两种场景。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@dataclass
class ParsedPage:
    """本数据结构作用：保存单页解析结果，便于后续定位文本来源页码。"""

    page_number: int
    text: str
    start_char: int
    end_char: int


@dataclass
class ParsedDocument:
    """本数据结构作用：保存单篇论文的最小解析结果，作为切块与检索准备输入。"""

    document_id: str
    file_name: str
    source_path: str
    total_pages: int
    total_characters: int
    full_text: str
    pages: list[ParsedPage]


def get_pdf_reader_class():
    """返回 PDF 读取器类型。

    输入：
        无。
    输出：
        pypdf 中的 PdfReader 类型。
    异常：
        当未安装 pypdf 时，抛出 ImportError。
    """

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "缺少 pypdf 依赖，请先执行 `pip install -r requirements.txt`。"
        ) from exc
    return PdfReader


def build_document_id(pdf_path: str | Path) -> str:
    """根据 PDF 文件路径生成稳定的文档编号。

    输入：
        pdf_path: PDF 文件路径。
    输出：
        适合作为文档编号的字符串。
    异常：
        无。
    """

    path = Path(pdf_path)
    normalized_name = re.sub(r"\W+", "_", path.stem.strip().lower()).strip("_")
    return normalized_name or "unknown_document"


def normalize_text(text: str) -> str:
    """清洗解析得到的原始文本。

    输入：
        text: 原始文本。
    输出：
        做过基础空白清洗后的文本。
    异常：
        无。
    """

    cleaned_text = text.replace("\r", "\n")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    return cleaned_text.strip()


def parse_pdf_file(pdf_path: str | Path) -> ParsedDocument:
    """解析单个 PDF 文件。

    输入：
        pdf_path: 待解析的 PDF 文件路径。
    输出：
        统一的论文解析结果对象。
    异常：
        当文件不存在时，抛出 FileNotFoundError。
        当文件不是 PDF 时，抛出 ValueError。
        当缺少解析依赖时，抛出 ImportError。
        当 PDF 无法解析或提取不到有效文本时，抛出 RuntimeError。
    """

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"不是 PDF 文件：{path}")

    PdfReader = get_pdf_reader_class()

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"无法打开 PDF 文件：{path.name}") from exc

    pages: list[ParsedPage] = []
    text_segments: list[str] = []
    current_char = 0

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise RuntimeError(
                f"第 {page_index} 页文本提取失败：{path.name}"
            ) from exc

        page_text = normalize_text(raw_text)
        if text_segments and page_text:
            # 页与页之间在 full_text 中插入两个换行，对应字符偏移也要同步前移。
            current_char += 2

        start_char = current_char
        if page_text:
            text_segments.append(page_text)
            # 记录当前页文本写入后的字符末端位置，供后续页码定位使用。
            current_char += len(page_text)
        end_char = current_char

        pages.append(
            ParsedPage(
                page_number=page_index,
                text=page_text,
                start_char=start_char,
                end_char=end_char,
            )
        )

    full_text = "\n\n".join(text_segments).strip()
    if not full_text:
        raise RuntimeError(
            f"未能从 PDF 中提取到有效文本：{path.name}。该文件可能是扫描版、加密文件或内容异常。"
        )

    return ParsedDocument(
        document_id=build_document_id(path),
        file_name=path.name,
        source_path=str(path),
        total_pages=len(pages),
        total_characters=len(full_text),
        full_text=full_text,
        pages=pages,
    )


def parse_pdf_files(
    pdf_paths: list[str | Path],
) -> tuple[list[ParsedDocument], list[dict[str, str]]]:
    """批量解析 PDF 文件，并记录失败信息。

    输入：
        pdf_paths: 待解析的 PDF 路径列表。
    输出：
        第一个返回值是成功解析的文档列表。
        第二个返回值是失败记录列表，每条记录包含文件名、路径和错误信息。
    异常：
        无。函数内部会捕获单文件解析异常，并转为失败记录返回。
    """

    documents: list[ParsedDocument] = []
    errors: list[dict[str, str]] = []

    for pdf_path in pdf_paths:
        current_path = Path(pdf_path).expanduser().resolve()
        try:
            documents.append(parse_pdf_file(current_path))
        except Exception as exc:
            # 单个文件失败不影响批量任务，统一落到 errors 供上层提示用户。
            errors.append(
                {
                    "file_name": current_path.name,
                    "source_path": str(current_path),
                    "error_message": str(exc),
                }
            )

    return documents, errors


def run_parser_demo() -> None:
    """执行 parser 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印解析统计与失败信息。
    异常：
        无。演示内部会自行处理可预期问题并输出提示。
    """

    from rag.loader import list_pdf_files

    pdf_files = list_pdf_files()
    print("Parser Demo")
    print(f"待解析 PDF 数量：{len(pdf_files)}")

    if not pdf_files:
        print("提示：请先将 PDF 文件放入 data/raw_papers/。")
        return

    documents, errors = parse_pdf_files(pdf_files)
    print(f"解析成功数量：{len(documents)}")
    print(f"解析失败数量：{len(errors)}")

    if documents:
        sample = documents[0]
        print(f"示例文档：{sample.file_name}")
        print(f"总页数：{sample.total_pages}")
        print(f"总字符数：{sample.total_characters}")
        print(f"文本预览：{sample.full_text[:120]}")

    if errors:
        print("失败示例：")
        print(errors[0])


if __name__ == "__main__":
    run_parser_demo()


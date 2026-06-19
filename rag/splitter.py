"""本模块作用：在整个智能体中负责将论文文本切成基础块，并整理为后续检索可直接使用的最小数据结构。"""

import sys
from dataclasses import dataclass
from pathlib import Path


# 下面这段处理用于兼容直接执行当前文件和以模块方式导入两种场景。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rag.loader import list_pdf_files
from rag.parser import ParsedDocument, parse_pdf_files


@dataclass
class TextChunk:
    """本数据结构作用：保存单个文本块及其最小检索元数据。"""

    chunk_id: str
    document_id: str
    file_name: str
    source_path: str
    chunk_index: int
    start_char: int
    end_char: int
    page_numbers: list[int]
    text: str


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """校验切块参数是否合法。

    输入：
        chunk_size: 单块最大字符数。
        chunk_overlap: 相邻块之间的重叠字符数。
    输出：
        无。
    异常：
        当切块参数不合法时，抛出 ValueError。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0。")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size。")


def find_chunk_page_numbers(
    document: ParsedDocument,
    start_char: int,
    end_char: int,
) -> list[int]:
    """根据字符范围推断文本块覆盖的页码。

    输入：
        document: 已解析的论文对象。
        start_char: 文本块起始字符位置。
        end_char: 文本块结束字符位置。
    输出：
        与文本块有交集的页码列表。
    异常：
        无。
    """

    page_numbers: list[int] = []
    for page in document.pages:
        has_overlap = start_char < page.end_char and end_char > page.start_char
        if has_overlap and page.page_number not in page_numbers:
            page_numbers.append(page.page_number)
    return page_numbers


def split_document(
    document: ParsedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    """对单篇论文进行基础切块。

    输入：
        document: 已解析的论文对象。
        chunk_size: 单块最大字符数。
        chunk_overlap: 相邻块之间的重叠字符数。
    输出：
        文本块对象列表。
    异常：
        当切块参数不合法时，抛出 ValueError。
    """

    validate_chunk_params(chunk_size, chunk_overlap)
    full_text = document.full_text.strip()
    if not full_text:
        return []

    chunks: list[TextChunk] = []
    start_char = 0
    chunk_index = 0
    # 步长由「块大小 - 重叠长度」决定，保证相邻块有上下文交集。
    step = chunk_size - chunk_overlap

    while start_char < len(full_text):
        end_char = min(start_char + chunk_size, len(full_text))
        chunk_text = full_text[start_char:end_char].strip()

        if chunk_text:
            # 每个块都携带字符区间和页码，便于问答时展示来源依据。
            chunks.append(
                TextChunk(
                    chunk_id=f"{document.document_id}_chunk_{chunk_index:04d}",
                    document_id=document.document_id,
                    file_name=document.file_name,
                    source_path=document.source_path,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    page_numbers=find_chunk_page_numbers(document, start_char, end_char),
                    text=chunk_text,
                )
            )
            chunk_index += 1

        if end_char >= len(full_text):
            break
        start_char += step

    return chunks


def build_chunk_records(chunks: list[TextChunk]) -> list[dict[str, object]]:
    """将文本块对象转换为后续检索可直接使用的最小记录结构。

    输入：
        chunks: 文本块对象列表。
    输出：
        记录列表，每条记录包含文本和最小元数据。
    异常：
        无。
    """

    records: list[dict[str, object]] = []
    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": {
                    "file_name": chunk.file_name,
                    "source_path": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "page_numbers": chunk.page_numbers,
                },
            }
        )
    return records


def build_knowledge_base_records(
    documents: list[ParsedDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict[str, object]]:
    """将多篇论文转换为最小知识库记录列表。

    输入：
        documents: 已解析的论文对象列表。
        chunk_size: 单块最大字符数。
        chunk_overlap: 相邻块之间的重叠字符数。
    输出：
        面向后续检索的最小记录列表。
    异常：
        当切块参数不合法时，抛出 ValueError。
    """

    all_chunks: list[TextChunk] = []
    for document in documents:
        # 多篇论文统一切块后再转记录，形成一个可直接检索的扁平库。
        all_chunks.extend(split_document(document, chunk_size, chunk_overlap))
    return build_chunk_records(all_chunks)


def run_splitter_demo() -> None:
    """执行 splitter 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印切块结果和示例记录。
    异常：
        无。演示内部会自行处理可预期问题并输出提示。
    """

    pdf_files = list_pdf_files()
    print("Splitter Demo")
    print(f"待处理 PDF 数量：{len(pdf_files)}")

    if not pdf_files:
        print("提示：请先将 PDF 文件放入 data/raw_papers/。")
        return

    documents, errors = parse_pdf_files(pdf_files)
    records = build_knowledge_base_records(documents)

    print(f"解析成功数量：{len(documents)}")
    print(f"切块记录数量：{len(records)}")
    print(f"解析失败数量：{len(errors)}")

    if records:
        sample = records[0]
        print(f"示例 chunk_id：{sample['chunk_id']}")
        print(f"示例 metadata：{sample['metadata']}")
        print(f"文本预览：{sample['text'][:120]}")

    if errors:
        print("失败示例：")
        print(errors[0])


if __name__ == "__main__":
    run_splitter_demo()


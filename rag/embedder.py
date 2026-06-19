"""本模块作用：在整个智能体中负责构建、保存、加载本地向量索引，为第四周 embedding 检索与 RAG 增强提供基础能力。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from llm_dashscope import DashScopeClient


@dataclass
class EmbeddingIndex:
    """本数据结构作用：保存本地向量索引的最小状态。"""

    model_name: str
    dimensions: int
    chunk_fingerprint: str
    index_path: str
    vectors: dict[str, list[float]] = field(default_factory=dict)


def build_chunk_fingerprint(chunk_records: list[dict[str, object]]) -> str:
    """为当前切块记录生成稳定指纹。

    输入：
        chunk_records: 知识库文本块记录列表。
    输出：
        稳定指纹字符串。
    异常：
        无。
    """

    fingerprint_source = "|".join(str(item.get("chunk_id", "")) for item in chunk_records)
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()


def get_default_embedding_index_path(
    processed_data_dir: str | Path,
    model_name: str,
    dimensions: int,
) -> Path:
    """返回默认向量索引文件路径。

    输入：
        processed_data_dir: 处理后数据目录。
        model_name: 向量模型名称。
        dimensions: 向量维度。
    输出：
        向量索引文件路径对象。
    异常：
        无。
    """

    target_dir = Path(processed_data_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"embedding_index_{model_name}_dim{dimensions}.json"
    return target_dir / file_name


def save_embedding_index(index: EmbeddingIndex) -> None:
    """将向量索引保存到本地文件。

    输入：
        index: 待保存的向量索引对象。
    输出：
        无。
    异常：
        当文件写入失败时，抛出 OSError。
    """

    payload = {
        "model_name": index.model_name,
        "dimensions": index.dimensions,
        "chunk_fingerprint": index.chunk_fingerprint,
        "vectors": index.vectors,
    }
    Path(index.index_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_embedding_index(index_path: str | Path) -> EmbeddingIndex | None:
    """从本地文件加载向量索引。

    输入：
        index_path: 向量索引文件路径。
    输出：
        成功时返回向量索引对象，不存在时返回 None。
    异常：
        当文件内容损坏时，抛出 ValueError。
    """

    path = Path(index_path).expanduser().resolve()
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    vectors = payload.get("vectors", {})
    if not isinstance(vectors, dict):
        raise ValueError("向量索引文件中的 vectors 格式不正确。")

    return EmbeddingIndex(
        model_name=str(payload.get("model_name", "")),
        dimensions=int(payload.get("dimensions", 0)),
        chunk_fingerprint=str(payload.get("chunk_fingerprint", "")),
        index_path=str(path),
        vectors={str(key): [float(item) for item in value] for key, value in vectors.items()},
    )


def is_embedding_index_compatible(
    index: EmbeddingIndex,
    chunk_records: list[dict[str, object]],
    model_name: str,
    dimensions: int,
) -> bool:
    """判断已加载向量索引是否可用于当前知识库。

    输入：
        index: 已加载的向量索引。
        chunk_records: 当前知识库切块记录。
        model_name: 目标向量模型名称。
        dimensions: 目标向量维度。
    输出：
        若兼容则返回 True，否则返回 False。
    异常：
        无。
    """

    return (
        index.model_name == model_name
        and index.dimensions == dimensions
        and index.chunk_fingerprint == build_chunk_fingerprint(chunk_records)
    )


def build_embedding_index(
    *,
    chunk_records: list[dict[str, object]],
    client: DashScopeClient,
    model_name: str,
    dimensions: int,
    index_path: str | Path,
    batch_size: int = 8,
) -> EmbeddingIndex:
    """调用向量接口，为知识库切块构建本地向量索引。

    输入：
        chunk_records: 知识库文本块记录列表。
        client: DashScope 客户端。
        model_name: 向量模型名称。
        dimensions: 向量维度。
        index_path: 索引保存路径。
        batch_size: 批量向量化大小。
    输出：
        构建完成的向量索引对象。
    异常：
        当向量接口调用失败时，抛出 RuntimeError。
        当写文件失败时，抛出 OSError。
    """

    vectors: dict[str, list[float]] = {}
    for start_index in range(0, len(chunk_records), batch_size):
        batch_records = chunk_records[start_index : start_index + batch_size]
        batch_texts = [str(item.get("text", "")) for item in batch_records]
        batch_vectors = client.embed_texts(
            model=model_name,
            texts=batch_texts,
            dimensions=dimensions,
        )
        for record, vector in zip(batch_records, batch_vectors):
            vectors[str(record.get("chunk_id", ""))] = vector

    index = EmbeddingIndex(
        model_name=model_name,
        dimensions=dimensions,
        chunk_fingerprint=build_chunk_fingerprint(chunk_records),
        index_path=str(Path(index_path).expanduser().resolve()),
        vectors=vectors,
    )
    save_embedding_index(index)
    return index


def run_embedder_demo() -> None:
    """执行 embedder 模块的最小演示说明。

    输入：
        无。
    输出：
        无。函数会直接打印使用提示。
    异常：
        无。
    """

    print("Embedder Demo")
    print("请在已配置 DASHSCOPE_API_KEY 后，通过 app.py 中的 build_index 命令构建本地向量索引。")


if __name__ == "__main__":
    run_embedder_demo()

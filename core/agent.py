"""本模块作用：在整个智能体中负责串联论文加载、检索、问答与最小学术分析，形成可运行的本地论文助手闭环。"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import OUTPUT_DIR
from llm_dashscope import DashScopeClient
# 提示词模块：集中维护回答规则与兜底文案，避免把硬编码文本散落在业务逻辑里。
from core.prompts import (
    build_answer_suffix,
    build_answer_system_prompt,
    build_answer_task_prompt,
    build_empty_library_message,
    build_no_result_message,
)
from core.workflow import (
    WorkflowRunResult,
    WorkflowState,
    build_default_workflow_plan,
    export_workflow_result,
    format_workflow_run_result,
    mark_step_status,
)
from rag.embedder import (
    EmbeddingIndex,
    build_embedding_index as run_build_embedding_index,
    get_default_embedding_index_path,
    is_embedding_index_compatible,
    load_embedding_index,
)
# RAG 基础链路：文件发现 -> PDF 解析 -> 文本切块 -> 关键词检索。
from rag.loader import list_pdf_files
from rag.parser import ParsedDocument, parse_pdf_files
from rag.retriever import (
    RetrievedChunk,
    extract_search_terms,
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_hybrid,
)
from rag.splitter import build_knowledge_base_records
# 学术分析工具：负责单篇论文结构化提取与结果格式化。
from tools.analyze_tool import StructuredPaperAnalysis, analyze_single_paper, format_analysis_result
from tools.compare_tool import MultiPaperComparison, compare_papers as run_compare_papers, format_comparison_result
from tools.outline_tool import ReviewOutline, format_review_outline, generate_review_outline


@dataclass
class KnowledgeBaseState:
    """本数据结构作用：保存当前本地论文知识库的最小运行状态。"""

    raw_papers_dir: str  # 原始论文目录（绝对路径字符串）。
    pdf_files: list[str] = field(default_factory=list)  # 扫描到的 PDF 文件路径列表。
    documents: list[ParsedDocument] = field(default_factory=list)  # 成功解析后的文档对象列表。
    chunk_records: list[dict[str, object]] = field(default_factory=list)  # 可直接检索的扁平化切块记录。
    parse_errors: list[dict[str, str]] = field(default_factory=list)  # 解析失败清单（文件名/路径/错误信息）。


@dataclass
class AgentAnswer:
    """本数据结构作用：保存一次问答的最终输出，供命令行或后续界面展示。"""

    question: str  # 用户原始问题。
    model_answer: str  # 最终展示给用户的回答文本（规则生成或 LLM 增强后）。
    sources: list[dict[str, object]]  # 来源依据列表（文件、页码、片段、分数等）。
    retrieved_count: int  # 本次问答命中的片段数量。
    used_prompt: str  # 本次问答内部使用的提示词文本（便于调试与教学演示）。


@dataclass
class PaperAnalysisResponse:
    """本数据结构作用：保存一次单篇论文结构化分析结果，供主程序展示。"""

    target: str  # 用户指定的分析目标（序号/关键词/文件名）。
    status_message: str  # 状态提示（成功、回退、未命中等）。
    analysis: StructuredPaperAnalysis | None  # 结构化对象，失败时为 None。
    formatted_output: str  # 直接可打印的分析文本。


@dataclass
class PaperComparisonResponse:
    """本数据结构作用：保存一次多篇论文比较结果，供主程序展示。"""

    targets: list[str]
    status_message: str
    comparison: MultiPaperComparison | None
    formatted_output: str


@dataclass
class ReviewOutlineResponse:
    """本数据结构作用：保存一次综述提纲生成结果，供主程序展示。"""

    topic: str
    status_message: str
    outline: ReviewOutline | None
    formatted_output: str


@dataclass
class WorkflowResponse:
    """本数据结构作用：保存一次多步工作流执行结果，供主程序展示。"""

    topic: str
    status_message: str
    workflow_result: WorkflowRunResult | None
    formatted_output: str


@dataclass
class EmbeddingIndexResponse:
    """本数据结构作用：保存一次向量索引加载或构建结果，供主程序展示。"""

    status_message: str
    index_path: str
    vector_count: int
    loaded_from_cache: bool


class CityScholarAgent:
    """本类作用：封装最小论文助手闭环，统一管理建库、检索、问答与单篇论文分析逻辑。"""

    def __init__(
        self,
        raw_papers_dir: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 3,
    ) -> None:
        """初始化最小论文代理。

        输入：
            raw_papers_dir: 本地 PDF 论文目录。
            chunk_size: 单个文本块的最大字符数。
            chunk_overlap: 相邻文本块之间的重叠字符数。
            top_k: 单次检索最多返回多少条结果。
        输出：
            无。
        异常：
            当切块参数不合法时，相关异常会在后续建库阶段抛出。
        """

        # 目录统一转绝对路径，避免命令行启动位置不同导致找不到数据目录。
        self.raw_papers_dir = Path(raw_papers_dir).expanduser().resolve()
        # 切块参数影响检索粒度与上下文连续性。
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # top_k 决定单次问答最多使用多少个来源片段。
        self.top_k = top_k
        # 延迟构建知识库：首次问答/分析前自动建库，缩短对象初始化等待时间。
        self.knowledge_base: KnowledgeBaseState | None = None
        # 第四周新增：向量索引和向量查询运行时状态。
        self.embedding_index: EmbeddingIndex | None = None
        self.embedding_client: DashScopeClient | None = None
        self.embedding_model_name: str = ""
        self.embedding_dimensions: int = 0

    def build_knowledge_base(self) -> KnowledgeBaseState:
        """根据本地 PDF 构建最小知识库。

        输入：
            无。
        输出：
            当前知识库状态对象。
        异常：
            当 PDF 解析依赖缺失时，抛出 ImportError。
            当目录访问失败或切块参数不合法时，抛出对应异常。
        """

        pdf_paths = list_pdf_files(self.raw_papers_dir)
        # 解析阶段会把失败文件记录在 parse_errors 中，不影响其余文件继续入库。
        documents, parse_errors = parse_pdf_files(pdf_paths)
        chunk_records = build_knowledge_base_records(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.knowledge_base = KnowledgeBaseState(
            raw_papers_dir=str(self.raw_papers_dir),
            pdf_files=[str(path) for path in pdf_paths],
            documents=documents,
            chunk_records=chunk_records,
            parse_errors=parse_errors,
        )
        return self.knowledge_base

    def ensure_knowledge_base_ready(self) -> KnowledgeBaseState:
        """确保知识库已准备完成。

        输入：
            无。
        输出：
            当前知识库状态对象。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        if self.knowledge_base is None:
            return self.build_knowledge_base()
        return self.knowledge_base

    def list_available_papers(self) -> list[dict[str, object]]:
        """列出当前知识库中可分析的论文信息。

        输入：
            无。
        输出：
            论文信息列表，每项包含序号、文件名、文档编号与页数统计。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        paper_items: list[dict[str, object]] = []
        for index, document in enumerate(knowledge_base.documents, start=1):
            paper_items.append(
                {
                    "index": index,
                    "file_name": document.file_name,
                    "document_id": document.document_id,
                    "total_pages": document.total_pages,
                    "total_characters": document.total_characters,
                }
            )
        return paper_items

    def find_document(self, target: str | None = None) -> ParsedDocument | None:
        """根据序号、文件名或文档编号定位目标论文。

        输入：
            target: 论文序号、文件名、文档编号或其片段。
        输出：
            命中的论文对象；若未找到则返回 None。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return None

        if target is None or not target.strip():
            # 默认分析第一篇，保证命令 `analyze` 可直接运行。
            return knowledge_base.documents[0]

        normalized_target = target.strip().lower()
        if normalized_target.isdigit():
            document_index = int(normalized_target) - 1
            if 0 <= document_index < len(knowledge_base.documents):
                return knowledge_base.documents[document_index]
            return None

        # 先做精确匹配，再做模糊包含匹配，降低误命中概率。
        for document in knowledge_base.documents:
            if normalized_target == document.file_name.lower() or normalized_target == document.document_id.lower():
                return document

        for document in knowledge_base.documents:
            if normalized_target in document.file_name.lower() or normalized_target in document.document_id.lower():
                return document

        return None

    def find_documents(self, targets: list[str] | None = None, default_count: int = 2) -> list[ParsedDocument]:
        """根据多个目标定位多篇论文。

        输入：
            targets: 论文序号、文件名、文档编号或其片段列表。
            default_count: targets 为空时默认选取多少篇论文。
        输出：
            命中的论文对象列表。
        异常：
            当某个目标无法匹配论文时，抛出 ValueError。
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return []

        if not targets:
            return knowledge_base.documents[: min(default_count, len(knowledge_base.documents))]

        selected_documents: list[ParsedDocument] = []
        seen_ids: set[str] = set()
        for target in targets:
            document = self.find_document(target)
            if document is None:
                raise ValueError(f"未找到目标论文：{target}")
            if document.document_id in seen_ids:
                continue
            seen_ids.add(document.document_id)
            selected_documents.append(document)
        return selected_documents

    def answer(self, question: str) -> AgentAnswer:
        """根据用户问题生成最小回答结果。

        输入：
            question: 用户问题文本。
        输出：
            包含模型回答与来源依据的结果对象。
        异常：
            当问题为空时，抛出 ValueError。
            当知识库尚未构建且自动构建失败时，抛出对应异常。
        """

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("问题不能为空。")

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.chunk_records:
            return AgentAnswer(
                question=cleaned_question,
                model_answer=build_empty_library_message(),
                sources=[],
                retrieved_count=0,
                used_prompt=build_answer_system_prompt(),
            )

        retrieved_chunks = self.retrieve_chunks_for_question(cleaned_question, knowledge_base.chunk_records)

        context_blocks = build_context_blocks(retrieved_chunks)
        used_prompt = (
            build_answer_system_prompt()
            + "\n\n"
            + build_answer_task_prompt(cleaned_question, context_blocks)
        )

        if not retrieved_chunks:
            return AgentAnswer(
                question=cleaned_question,
                model_answer=build_no_result_message(cleaned_question),
                sources=[],
                retrieved_count=0,
                used_prompt=used_prompt,
            )

        model_answer = synthesize_answer(cleaned_question, retrieved_chunks)
        sources = build_source_entries(retrieved_chunks)
        return AgentAnswer(
            question=cleaned_question,
            model_answer=model_answer,
            sources=sources,
            retrieved_count=len(retrieved_chunks),
            used_prompt=used_prompt,
        )

    def retrieve_chunks_for_question(
        self,
        question: str,
        chunk_records: list[dict[str, object]],
    ) -> list[RetrievedChunk]:
        """根据当前可用能力选择关键词检索或混合检索。

        输入：
            question: 用户问题文本。
            chunk_records: 当前知识库切块记录列表。
        输出：
            检索结果列表。
        异常：
            当问题为空时，抛出 ValueError。
        """

        if (
            self.embedding_index is not None
            and self.embedding_client is not None
            and self.embedding_model_name
            and self.embedding_dimensions > 0
        ):
            try:
                query_vector = self.embedding_client.embed_texts(
                    model=self.embedding_model_name,
                    texts=[question],
                    dimensions=self.embedding_dimensions,
                )[0]
                return retrieve_relevant_chunks_hybrid(
                    question=question,
                    chunk_records=chunk_records,
                    embedding_vectors=self.embedding_index.vectors,
                    query_vector=query_vector,
                    top_k=self.top_k,
                )
            except Exception:
                # 向量检索失败时回退关键词检索，保证问答流程不中断。
                pass

        return retrieve_relevant_chunks(
            question=question,
            chunk_records=chunk_records,
            top_k=self.top_k,
        )

    def prepare_embedding_index(
        self,
        *,
        client: DashScopeClient | None,
        model_name: str,
        dimensions: int,
        processed_data_dir: str | Path,
        build_if_missing: bool = False,
        force_rebuild: bool = False,
    ) -> EmbeddingIndexResponse:
        """加载或构建第四周的本地向量索引。

        输入：
            client: DashScope 客户端；构建索引时必需。
            model_name: 向量模型名称。
            dimensions: 向量维度。
            processed_data_dir: 索引目录。
            build_if_missing: 若索引缺失是否自动构建。
            force_rebuild: 是否忽略旧索引并强制重建。
        输出：
            向量索引响应对象。
        异常：
            当知识库为空时，抛出 ValueError。
            当需要构建但 client 缺失时，抛出 ValueError。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.chunk_records:
            raise ValueError("当前没有可用于构建向量索引的切块记录。")

        index_path = get_default_embedding_index_path(processed_data_dir, model_name, dimensions)
        self.embedding_client = client
        self.embedding_model_name = model_name
        self.embedding_dimensions = dimensions

        if not force_rebuild:
            cached_index = load_embedding_index(index_path)
            if cached_index is not None and is_embedding_index_compatible(
                cached_index,
                knowledge_base.chunk_records,
                model_name,
                dimensions,
            ):
                self.embedding_index = cached_index
                return EmbeddingIndexResponse(
                    status_message="已加载本地向量索引。",
                    index_path=str(index_path),
                    vector_count=len(cached_index.vectors),
                    loaded_from_cache=True,
                )

        if not build_if_missing:
            self.embedding_index = None
            return EmbeddingIndexResponse(
                status_message="当前未找到可用向量索引，可执行 build_index 构建。",
                index_path=str(index_path),
                vector_count=0,
                loaded_from_cache=False,
            )

        if client is None:
            raise ValueError("构建向量索引需要可用的大模型客户端。")

        built_index = run_build_embedding_index(
            chunk_records=knowledge_base.chunk_records,
            client=client,
            model_name=model_name,
            dimensions=dimensions,
            index_path=index_path,
        )
        self.embedding_index = built_index
        return EmbeddingIndexResponse(
            status_message="已完成本地向量索引构建。",
            index_path=str(index_path),
            vector_count=len(built_index.vectors),
            loaded_from_cache=False,
        )

    def analyze_paper(self, target: str | None = None) -> PaperAnalysisResponse:
        """对指定论文执行单篇结构化学术分析。

        输入：
            target: 论文序号、文件名、文档编号或其片段；为空时默认分析第一篇。
        输出：
            包含结构化分析结果与展示文本的响应对象。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return PaperAnalysisResponse(
                target=target or "默认第一篇论文",
                status_message="当前没有可分析的论文。请先放入 PDF 并确认其能够被成功解析。",
                analysis=None,
                formatted_output="当前没有可分析的论文。",
            )

        document = self.find_document(target)
        if document is None:
            return PaperAnalysisResponse(
                target=target or "默认第一篇论文",
                status_message="未找到匹配的论文，请先使用 papers 查看可用论文列表。",
                analysis=None,
                formatted_output="未找到匹配的论文，请先使用 papers 查看可用论文列表。",
            )

        analysis = analyze_single_paper(
            text_or_segments=document.full_text,
            file_name=document.file_name,
            document_id=document.document_id,
        )
        return PaperAnalysisResponse(
            target=target or document.file_name,
            status_message=f"已完成对《{document.file_name}》的结构化提取。",
            analysis=analysis,
            formatted_output=format_analysis_result(analysis),
        )

    def compare_papers(self, targets: list[str] | None = None, topic_hint: str = "未指定比较主题") -> PaperComparisonResponse:
        """对多篇论文执行最小结构化比较。

        输入：
            targets: 论文序号、文件名、文档编号或其片段列表；为空时默认取前两篇。
            topic_hint: 可选的比较主题说明。
        输出：
            多篇论文比较响应对象。
        异常：
            当自动建库失败时，抛出对应异常。
            当目标论文无法匹配时，抛出 ValueError。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if len(knowledge_base.documents) < 2:
            return PaperComparisonResponse(
                targets=targets or [],
                status_message="当前可用论文数量不足 2 篇，暂时无法执行多篇比较。",
                comparison=None,
                formatted_output="当前可用论文数量不足 2 篇，暂时无法执行多篇比较。",
            )

        documents = self.find_documents(targets, default_count=2)
        if len(documents) < 2:
            return PaperComparisonResponse(
                targets=targets or [],
                status_message="多篇比较至少需要 2 篇论文，请重新指定目标。",
                comparison=None,
                formatted_output="多篇比较至少需要 2 篇论文，请重新指定目标。",
            )

        paper_inputs = [
            {
                "file_name": document.file_name,
                "document_id": document.document_id,
                "full_text": document.full_text,
            }
            for document in documents
        ]
        comparison = run_compare_papers(paper_inputs, topic_hint=topic_hint)
        return PaperComparisonResponse(
            targets=[document.file_name for document in documents],
            status_message=f"已完成 {len(documents)} 篇论文的最小比较。",
            comparison=comparison,
            formatted_output=format_comparison_result(comparison),
        )

    def generate_review_outline(
        self,
        topic: str,
        targets: list[str] | None = None,
    ) -> ReviewOutlineResponse:
        """根据若干论文生成最小综述提纲。

        输入：
            topic: 综述主题。
            targets: 论文序号、文件名、文档编号或其片段列表；为空时默认取前三篇。
        输出：
            综述提纲响应对象。
        异常：
            当自动建库失败时，抛出对应异常。
            当主题为空或目标论文无法匹配时，抛出 ValueError。
        """

        cleaned_topic = topic.strip()
        if not cleaned_topic:
            raise ValueError("综述主题不能为空。")

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return ReviewOutlineResponse(
                topic=cleaned_topic,
                status_message="当前没有可用于生成综述提纲的论文。",
                outline=None,
                formatted_output="当前没有可用于生成综述提纲的论文。",
            )

        documents = self.find_documents(targets, default_count=3)
        if len(documents) < 2:
            return ReviewOutlineResponse(
                topic=cleaned_topic,
                status_message="综述提纲生成至少需要 2 篇论文，请重新指定目标。",
                outline=None,
                formatted_output="综述提纲生成至少需要 2 篇论文，请重新指定目标。",
            )

        paper_inputs = [
            {
                "file_name": document.file_name,
                "document_id": document.document_id,
                "full_text": document.full_text,
            }
            for document in documents
        ]
        outline = generate_review_outline(cleaned_topic, paper_inputs)
        return ReviewOutlineResponse(
            topic=cleaned_topic,
            status_message=f"已基于 {len(documents)} 篇论文生成最小综述提纲。",
            outline=outline,
            formatted_output=format_review_outline(outline),
        )

    def run_review_workflow(
        self,
        topic: str,
        targets: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> WorkflowResponse:
        """执行第三周的最小多步科研工作流。

        输入：
            topic: 工作流主题。
            targets: 目标论文列表；为空时默认取前三篇。
            output_dir: 导出目录；为空时使用项目默认输出目录。
        输出：
            工作流响应对象。
        异常：
            当主题为空或目标论文无法匹配时，抛出 ValueError。
            当自动建库失败时，抛出对应异常。
        """

        cleaned_topic = topic.strip()
        if not cleaned_topic:
            raise ValueError("工作流主题不能为空。")

        knowledge_base = self.ensure_knowledge_base_ready()
        if len(knowledge_base.documents) < 2:
            return WorkflowResponse(
                topic=cleaned_topic,
                status_message="当前可用论文不足 2 篇，暂时无法运行多步工作流。",
                workflow_result=None,
                formatted_output="当前可用论文不足 2 篇，暂时无法运行多步工作流。",
            )

        plan = build_default_workflow_plan(topic=cleaned_topic, targets=targets)
        state = WorkflowState(topic=cleaned_topic, targets=targets or [])

        documents = self.find_documents(targets, default_count=3)
        if len(documents) < 2:
            return WorkflowResponse(
                topic=cleaned_topic,
                status_message="多步工作流至少需要 2 篇论文，请重新指定目标。",
                workflow_result=None,
                formatted_output="多步工作流至少需要 2 篇论文，请重新指定目标。",
            )

        mark_step_status(plan, "select_papers", "completed")
        state.selected_papers = [document.file_name for document in documents]
        state.step_logs.append(f"步骤 1：已选中 {len(documents)} 篇论文。")

        comparison_result = self.compare_papers(
            targets=[document.document_id for document in documents],
            topic_hint=cleaned_topic,
        )
        mark_step_status(plan, "compare_papers", "completed")
        state.comparison_text = comparison_result.formatted_output
        state.step_logs.append("步骤 2：已完成多篇论文比较。")

        outline_result = self.generate_review_outline(
            topic=cleaned_topic,
            targets=[document.document_id for document in documents],
        )
        mark_step_status(plan, "generate_outline", "completed")
        state.outline_text = outline_result.formatted_output
        state.step_logs.append("步骤 3：已生成综述提纲。")

        artifact = export_workflow_result(
            output_dir=str(Path(output_dir).expanduser().resolve()) if output_dir is not None else str(OUTPUT_DIR),
            topic=cleaned_topic,
            selected_papers=state.selected_papers,
            comparison_text=state.comparison_text,
            outline_text=state.outline_text,
            step_logs=state.step_logs,
        )
        mark_step_status(plan, "export_markdown", "completed")
        state.export_artifact = artifact
        state.step_logs.append(f"步骤 4：已导出 Markdown 报告到 {artifact.output_path}")

        workflow_result = WorkflowRunResult(
            status_message=f"已完成主题“{cleaned_topic}”的最小多步工作流。",
            plan=plan,
            state=state,
            formatted_output="",
        )
        workflow_result.formatted_output = format_workflow_run_result(workflow_result)

        return WorkflowResponse(
            topic=cleaned_topic,
            status_message=workflow_result.status_message,
            workflow_result=workflow_result,
            formatted_output=workflow_result.formatted_output,
        )


def split_text_into_sentences(text: str) -> list[str]:
    """将文本按中文与英文常见分隔符切分为句子。

    输入：
        text: 原始文本。
    输出：
        清洗后的句子列表。
    异常：
        无。
    """

    raw_sentences = re.split(r"[。！？!?；;\n]+", text)
    sentences: list[str] = []
    for sentence in raw_sentences:
        cleaned_sentence = re.sub(r"\s+", " ", sentence).strip()
        if cleaned_sentence:
            sentences.append(cleaned_sentence)
    return sentences


def truncate_text(text: str, max_length: int = 120) -> str:
    """将文本裁剪到适合回答展示的长度。

    输入：
        text: 原始文本。
        max_length: 最大保留字符数。
    输出：
        适合展示的短文本。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def score_supporting_sentence(
    question: str,
    sentence: str,
    retrieval_score: float,
) -> float:
    """计算候选支撑句与问题的相关程度。

    输入：
        question: 用户问题文本。
        sentence: 候选句子文本。
        retrieval_score: 句子所属文本块的检索分数。
    输出：
        用于排序的句子分数。
    异常：
        无。
    """

    question_terms = extract_search_terms(question)
    sentence_text = sentence.lower()
    # 以检索分数为基础分，再叠加关键词命中与句长等启发式特征。
    score = retrieval_score * 0.5

    if question.strip().lower() in sentence_text:
        score += 6.0

    for term in question_terms:
        if term in sentence_text:
            if len(term) >= 6:
                score += 2.2
            elif len(term) >= 4:
                score += 1.7
            else:
                score += 1.0

    if len(sentence.strip()) < 8:
        score -= 1.0

    return score


def select_supporting_sentences(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    max_sentences: int = 3,
) -> list[str]:
    """从召回片段中选出最能支撑回答的句子。

    输入：
        question: 用户问题文本。
        retrieved_chunks: 已召回的文本块列表。
        max_sentences: 最多保留多少句。
    输出：
        适合直接拼装为回答的句子列表。
    异常：
        无。
    """

    sentence_candidates: list[tuple[float, str]] = []
    for chunk in retrieved_chunks:
        for sentence in split_text_into_sentences(chunk.text):
            sentence_score = score_supporting_sentence(question, sentence, chunk.score)
            if sentence_score <= 0:
                continue
            sentence_candidates.append((sentence_score, truncate_text(sentence)))

    # 先按分数降序，再按文本稳定排序，保证结果可复现。
    sentence_candidates.sort(key=lambda item: (-item[0], item[1]))

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence in sentence_candidates:
        if sentence in seen_sentences:
            continue
        selected_sentences.append(sentence)
        seen_sentences.add(sentence)
        if len(selected_sentences) >= max_sentences:
            break

    return selected_sentences


def build_context_blocks(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """将召回结果转换为后续回答模块可使用的上下文块。

    输入：
        retrieved_chunks: 已召回的文本块列表。
    输出：
        带有论文名、页码和片段的文本块列表。
    异常：
        无。
    """

    context_blocks: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        # 统一上下文模板，便于后续 LLM 增强时直接复用。
        page_numbers = chunk.metadata.get("page_numbers", [])
        page_text = "、".join(str(number) for number in page_numbers) if page_numbers else "未知页码"
        file_name = str(chunk.metadata.get("file_name", "未知论文"))
        context_blocks.append(
            f"[来源 {index}] 论文：{file_name} | 页码：{page_text} | 片段：{chunk.snippet}"
        )
    return context_blocks


def synthesize_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """基于召回片段生成最小回答文本。

    输入：
        question: 用户问题文本。
        retrieved_chunks: 已召回的文本块列表。
    输出：
        适合直接展示给用户的回答文本。
    异常：
        无。
    """

    supporting_sentences = select_supporting_sentences(question, retrieved_chunks)
    answer_lines = ["根据当前召回的论文片段，可以整理出以下回答："]

    if supporting_sentences:
        for index, sentence in enumerate(supporting_sentences, start=1):
            answer_lines.append(f"{index}. {sentence}")
    else:
        # 当抽取不到稳定句子时，回退展示高分片段，保证可解释性。
        answer_lines.append("当前能够召回相关片段，但还不足以抽取出稳定的支撑句。")
        for index, chunk in enumerate(retrieved_chunks[:2], start=1):
            answer_lines.append(f"{index}. {truncate_text(chunk.snippet)}")

    answer_lines.append(build_answer_suffix())
    return "\n".join(answer_lines)


def build_source_entries(
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int = 3,
) -> list[dict[str, object]]:
    """将召回结果转换为可展示的来源依据列表。

    输入：
        retrieved_chunks: 已召回的文本块列表。
        max_sources: 最多保留多少条来源信息。
    输出：
        适合界面或命令行展示的来源依据列表。
    异常：
        无。
    """

    sources: list[dict[str, object]] = []
    for chunk in retrieved_chunks[:max_sources]:
        sources.append(
            {
                "file_name": str(chunk.metadata.get("file_name", "未知论文")),
                "source_path": str(chunk.metadata.get("source_path", "")),
                "page_numbers": chunk.metadata.get("page_numbers", []),
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "score": round(chunk.score, 2),
                "snippet": chunk.snippet,
                "matched_terms": chunk.matched_terms,
            }
        )
    return sources


def run_agent_demo() -> None:
    """执行 agent 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印问答与分析示例。
    异常：
        无。
    """

    demo_agent = CityScholarAgent(raw_papers_dir="data/raw_papers")
    demo_agent.knowledge_base = KnowledgeBaseState(
        raw_papers_dir="data/raw_papers",
        pdf_files=["data/raw_papers/demo_paper.pdf"],
        documents=[
            ParsedDocument(
                document_id="demo_paper",
                file_name="demo_paper.pdf",
                source_path="data/raw_papers/demo_paper.pdf",
                total_pages=2,
                total_characters=88,
                full_text=(
                    "本文以某沿海超大城市的社区更新项目为研究对象，关注公共服务设施可达性变化对居民满意度的影响。"
                    "研究采用问卷调查、POI 数据分析与多元回归方法。"
                    "数据来源包括社区问卷、城市开放 POI 数据和统计年鉴。"
                    "结果表明，设施步行可达性提升能够显著改善居民对社区更新的评价。"
                    "然而，样本主要集中于中心城区，外部可推广性仍然受限。"
                    "研究建议在城市治理与规划中，将十五分钟生活圈与道路安全整治协同推进。"
                ),
                pages=[],
            )
        ],
        chunk_records=[
            {
                "chunk_id": "demo_chunk_0001",
                "document_id": "demo_paper",
                "text": "城市更新项目中，公共服务设施的可达性提升与居民满意度存在明显正相关。",
                "metadata": {
                    "file_name": "demo_paper.pdf",
                    "source_path": "data/raw_papers/demo_paper.pdf",
                    "chunk_index": 0,
                    "page_numbers": [1],
                },
            },
            {
                "chunk_id": "demo_chunk_0002",
                "document_id": "demo_paper",
                "text": "步行十五分钟生活圈能够改善日常服务覆盖，但需要结合道路安全治理。",
                "metadata": {
                    "file_name": "demo_paper.pdf",
                    "source_path": "data/raw_papers/demo_paper.pdf",
                    "chunk_index": 1,
                    "page_numbers": [2],
                },
            },
        ],
        parse_errors=[],
    )

    answer_result = demo_agent.answer("城市更新如何影响公共服务可达性？")
    print("Agent Demo - 问答")
    print("模型回答：")
    print(answer_result.model_answer)
    print("来源依据：")
    for index, source in enumerate(answer_result.sources, start=1):
        print(f"{index}. {source['file_name']} | 页码：{source['page_numbers']} | 片段：{source['snippet']}")

    analysis_result = demo_agent.analyze_paper("1")
    print("\nAgent Demo - 结构化分析")
    print(analysis_result.formatted_output)


if __name__ == "__main__":
    run_agent_demo()


"""
本模块是 CityScholar-Agent 的命令行入口与主调度层。

主要职责：
1. 加载应用配置并初始化运行环境；
2. 构建知识库、向量索引与大模型客户端；
3. 提供命令路由、任务分发、对话历史管理；
4. 支持检索问答、论文分析、论文比较、综述提纲与工作流；
5. 在终端中输出可读的状态信息与结果。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

from config import get_app_config
from core.agent import (
    AgentAnswer,
    CityScholarAgent,
    EmbeddingIndexResponse,
    KnowledgeBaseState,
    PaperAnalysisResponse,
    PaperComparisonResponse,
    ReviewOutlineResponse,
    WorkflowResponse,
)
from llm_dashscope import DashScopeClient, parse_first_json_object
from tools.analyze_tool import StructuredPaperAnalysis, format_analysis_result

# ==============================
# 配置区：集中管理，减少硬编码
# ==============================

ALLOWED_ROUTE_INTENTS = {
    "question",
    "analyze",
    "compare",
    "outline",
    "workflow",
    "build_index",
    "rebuild_index",
    "reload_index",
    "rescan",
    "papers",
    "help",
    "history",
    "clear",
    "stats",
    "config",
    "export_history",
    "route",
    "last",
    "save_answer",
    "exit",
    "other",
}

COMMAND_ALIASES = {
    "help": {"help", "帮助"},
    "papers": {"papers", "list", "论文", "论文列表"},
    "build_index": {"build_index", "index"},
    "rebuild_index": {"rebuild_index"},
    "reload_index": {"reload_index", "reload", "重载索引"},
    "rescan": {"rescan", "scan", "重新扫描", "重扫"},
    "history": {"history", "历史", "对话历史"},
    "clear": {"clear", "clear_history", "清空", "清空历史"},
    "stats": {"stats", "状态", "统计"},
    "config": {"config", "配置"},
    "export_history": {"export_history", "导出历史", "export"},
    "route": {"route", "路由"},
    "last": {"last", "上次回答", "last_answer"},
    "save_answer": {"save_answer", "保存回答", "save"},
    "exit": {"exit", "quit", "q", "退出", "结束"},
}

COMMAND_HELP = [
    ("直接输入问题", "执行论文检索问答（若已构建向量索引，则自动走混合检索）"),
    ("build_index", "构建本地向量索引"),
    ("rebuild_index", "强制重建本地向量索引"),
    ("reload_index", "重新加载向量索引"),
    ("rescan", "重新扫描论文目录并重建知识库"),
    ("papers", "查看论文列表"),
    ("history", "查看最近对话记录"),
    ("clear", "清空对话历史"),
    ("stats", "查看当前知识库状态"),
    ("config", "查看当前配置"),
    ("export_history", "导出对话历史到文件"),
    ("route", "查看当前输入的路由结果"),
    ("last", "查看上一次模型回答"),
    ("save_answer", "保存上一次模型回答到文件"),
    ("analyze", "分析第一篇论文"),
    ("analyze 1", "分析第 1 篇论文"),
    ("analyze 关键词", "按文件名或文档编号模糊匹配分析"),
    ("compare", "比较前两篇论文"),
    ("compare 1,2", "比较指定论文"),
    ("outline 主题", "基于默认论文生成综述提纲"),
    ("outline 1,2,3 :: 主题", "基于指定论文生成综述提纲"),
    ("workflow 主题", "基于默认论文执行多步工作流"),
    ("workflow 1,2,3 :: 主题", "基于指定论文执行多步工作流"),
    ("help", "查看帮助"),
    ("exit", "退出程序"),
]

PROMPTS = {
    "answer_system": """你是 CityScholar-Agent 的学术问答模块。
你只能基于给定来源回答，不要编造来源中不存在的结论。
回答必须包含简洁结论，并尽量保留来源编号引用。""",

    "answer_user": """用户问题：{question}

来源片段：
{context}

请输出中文回答，结构为：
1) 直接回答
2) 关键依据（用 [来源1] 这样的编号表示）
3) 不确定性提示（如果有）""",

    "analysis_system": """你是城市研究领域的论文分析助手。
请仅根据给定论文内容提取结构化结果，不要编造。
输出必须是 JSON 对象。""",

    "analysis_user": """请对以下论文内容进行结构化提取，并严格输出 JSON。
字段必须包含：
research_question, research_object, methods, data_source, key_findings, limitations, implications, evidence_map
其中 evidence_map 是对象，键为上述七个字段名，值为字符串数组（每项是依据片段）。

论文文件名：{file_name}
文档编号：{document_id}

论文内容：
{paper_text}""",

    "router_json": """你是 CityScholar-Agent 的命令路由器。
请根据用户输入识别任务类型，并提取参数，严格输出 JSON 对象，不要输出其他内容。

可选 intent：
- question：普通检索问答
- analyze：论文分析
- compare：论文比较
- outline：综述提纲
- workflow：综述工作流
- build_index：构建向量索引
- rebuild_index：重建向量索引
- reload_index：重新加载向量索引
- rescan：重新扫描知识库
- papers：查看论文列表
- help：查看帮助
- history：查看对话历史
- clear：清空对话历史
- stats：查看状态
- config：查看配置
- export_history：导出对话历史
- route：查看当前路由
- last：查看上一次回答
- save_answer：保存上一次回答
- exit：退出
- other：其他

输出 JSON 格式：
{{
  "intent": "question|analyze|compare|outline|workflow|build_index|rebuild_index|reload_index|rescan|papers|help|history|clear|stats|config|export_history|route|last|save_answer|exit|other",
  "targets": ["可选，论文编号/关键词列表"],
  "topic": "可选，综述主题或任务主题，没有则为空字符串"
}}

规则：
1. 如果用户在问一个知识问题，intent = question
2. 如果用户要求分析某篇论文，intent = analyze
3. 如果用户要求比较多篇论文，intent = compare
4. 如果用户要求生成综述提纲，intent = outline
5. 如果用户要求执行综述工作流，intent = workflow
6. 如果用户只是查看论文列表，intent = papers
7. 如果用户要求构建/重建/重载索引，识别为 build_index / rebuild_index / reload_index
8. 如果用户要求重新扫描知识库，intent = rescan
9. 如果用户要求查看历史，intent = history
10. 如果用户要求清空历史，intent = clear
11. 如果用户要求查看状态，intent = stats
12. 如果用户要求查看配置，intent = config
13. 如果用户要求导出历史，intent = export_history
14. 如果用户要求查看当前路由，intent = route
15. 如果用户要求查看上一次回答，intent = last
16. 如果用户要求保存上一次回答，intent = save_answer
17. 如果用户要求帮助，intent = help
18. 如果用户要求退出，intent = exit
19. targets 只提取明确提到的论文编号、关键词或文件名线索
20. topic 只提取明确的主题
21. 只输出 JSON，不要解释

用户输入：
{user_input}
""",
}

RERANK_CONFIG = {
    "score_weight": 0.7,
    "match_weight": 0.3,
}

EVALUATION_CONFIG = {
    "min_answer_length": 10,
    "require_sources": True,
    "fallback_reason_ok": "来源充足，回答完整",
}

# ==============================
# 对话历史记忆模块
# ==============================

@dataclass
class ChatMessage:
    """对话消息记录。"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

class ChatHistory:
    """轻量级对话历史容器。"""

    def __init__(self, max_turns: int = 10):
        self.messages: list[ChatMessage] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        """添加一条消息。"""
        self.messages.append(ChatMessage(role=role, content=content))
        if len(self.messages) > self.max_turns:
            self.messages.pop(0)

    def to_list(self) -> list[dict[str, str]]:
        """转换为字典列表。"""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def latest(self, n: int = 6) -> list[dict[str, str]]:
        """获取最近 n 条消息。"""
        return [{"role": m.role, "content": m.content} for m in self.messages[-n:]]

    def clear(self) -> None:
        """清空历史。"""
        self.messages = []

# ==============================
# 路由结构
# ==============================

@dataclass
class RouteDecision:
    """命令路由结果。"""
    intent: str = "question"
    targets: list[str] = field(default_factory=list)
    topic: str = ""
    raw_text: str = ""

# ==============================
# 状态容器
# ==============================

class AgentState:
    """应用运行状态。"""

    def __init__(self):
        self.user_input = ""
        self.intent = "question"
        self.history = ChatHistory()
        self.retrieval_result = None
        self.reranked_result = None
        self.llm_answer = ""
        self.evaluation: dict[str, Any] = {}
        self.output = None
        self.warning: str | None = None
        self.route: RouteDecision | None = None

# ==============================
# 基础工具函数
# ==============================

def ensure_directories(paths: list[Path]) -> None:
    """确保目录存在。"""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def initialize_llm_client(config: dict[str, Path | str | int | bool]) -> DashScopeClient | None:
    """初始化 DashScope 客户端。"""
    api_key = str(config.get("dashscope_api_key", "")).strip()
    base_url = str(config.get("dashscope_base_url", "")).strip()
    timeout_sec = int(config.get("dashscope_timeout_sec", 45))
    if not api_key:
        return None
    try:
        return DashScopeClient(api_key=api_key, base_url=base_url, timeout_sec=timeout_sec)
    except Exception:
        return None

def format_page_numbers(page_numbers: list[int]) -> str:
    """格式化页码列表。"""
    if not page_numbers:
        return "未知页码"
    return "、".join(str(number) for number in page_numbers)

def show_cli_help() -> None:
    """打印命令帮助信息。"""
    print("\n命令说明：")
    for cmd, desc in COMMAND_HELP:
        print(f"- {cmd}：{desc}")

def show_parse_error_summary(parse_errors: list[dict[str, str]], max_items: int = 3) -> None:
    """打印 PDF 解析失败摘要。"""
    if not parse_errors:
        return
    print("以下 PDF 在建库时解析失败，已自动跳过：")
    for error_item in parse_errors[:max_items]:
        print(f"- 文件：{error_item['file_name']}")
        print(f"  原因：{error_item['error_message']}")

def show_startup_info(
    config: dict[str, Path | str | int | bool],
    knowledge_base: KnowledgeBaseState,
    llm_client: DashScopeClient | None,
    embedding_status: EmbeddingIndexResponse | None,
) -> None:
    """打印启动信息。"""
    llm_enabled = llm_client is not None
    answer_model = str(config.get("dashscope_answer_model", "qwen-plus"))
    analysis_model = str(config.get("dashscope_analysis_model", "qwen-max"))

    print("=" * 60)
    print(f"启动项目：{config['project_name']}")
    print(f"项目目录：{config['base_dir']}")
    print(f"论文目录：{config['raw_papers_dir']}")
    print(f"发现 PDF 数量：{len(knowledge_base.pdf_files)}")
    print(f"解析成功论文数：{len(knowledge_base.documents)}")
    print(f"可检索片段数：{len(knowledge_base.chunk_records)}")
    print(f"解析失败论文数：{len(knowledge_base.parse_errors)}")
    print(f"大模型增强：{'已启用' if llm_enabled else '未启用'}")
    if llm_enabled:
        print(f"问答模型：{answer_model}")
        print(f"分析模型：{analysis_model}")
        print(f"向量模型：{config['dashscope_embedding_model']}")
    if embedding_status is not None:
        print(f"向量索引：{embedding_status.status_message}")
        if embedding_status.vector_count > 0:
            print(f"向量数量：{embedding_status.vector_count}")
    print("当前模式：结构化路由 + 对话记忆 + 检索问答 + Rerank + 评估")
    show_cli_help()
    print("=" * 60)

# ==============================
# 路由模块：规则优先，LLM 补充
# ==============================

def normalize_route_json(data: dict[str, Any], raw_text: str) -> RouteDecision:
    """将 LLM 产出的路由 JSON 规范化。"""
    intent = str(data.get("intent", "question")).strip().lower()
    if intent not in ALLOWED_ROUTE_INTENTS:
        intent = "question"

    raw_targets = data.get("targets", [])
    targets: list[str] = []
    if isinstance(raw_targets, list):
        targets = [str(item).strip() for item in raw_targets if str(item).strip()]
    elif isinstance(raw_targets, str) and raw_targets.strip():
        targets = [raw_targets.strip()]

    topic = str(data.get("topic", "")).strip()

    return RouteDecision(
        intent=intent,
        targets=targets,
        topic=topic,
        raw_text=raw_text,
    )

def detect_rule_based_route(user_input: str) -> RouteDecision | None:
    """基于规则的路由识别。"""
    normalized = user_input.strip().lower()

    for command_name, aliases in COMMAND_ALIASES.items():
        if normalized in aliases:
            return RouteDecision(intent=command_name, raw_text=user_input)

    if normalized.startswith("analyze") or user_input.startswith("分析"):
        return RouteDecision(intent="analyze", raw_text=user_input)

    if normalized.startswith("compare") or user_input.startswith("对比"):
        return RouteDecision(intent="compare", raw_text=user_input)

    if normalized.startswith("outline") or user_input.startswith("提纲"):
        return RouteDecision(intent="outline", raw_text=user_input)

    if normalized.startswith("workflow") or user_input.startswith("流程"):
        return RouteDecision(intent="workflow", raw_text=user_input)

    return None

def route_with_llm(
    user_input: str,
    llm_client: DashScopeClient | None,
    model: str,
) -> RouteDecision:
    """使用 LLM 进行路由补充识别。"""
    if llm_client is None:
        return RouteDecision(intent="question", raw_text=user_input)

    try:
        prompt = PROMPTS["router_json"].format(user_input=user_input)
        raw_text = llm_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        json_data = parse_first_json_object(raw_text)
        if not isinstance(json_data, dict):
            return RouteDecision(intent="question", raw_text=user_input)
        return normalize_route_json(json_data, user_input)
    except Exception:
        return RouteDecision(intent="question", raw_text=user_input)

def resolve_route(
    user_input: str,
    llm_client: DashScopeClient | None,
    model: str,
) -> RouteDecision:
    """解析用户输入的任务路由。"""
    rule_route = detect_rule_based_route(user_input)
    if rule_route is not None:
        return rule_route
    return route_with_llm(user_input, llm_client, model)

# ==============================
# 检索结果重排模块
# ==============================

def rerank_sources(sources: list[dict]) -> list[dict]:
    """对来源进行简单重排。"""
    scored = []
    score_weight = float(RERANK_CONFIG["score_weight"])
    match_weight = float(RERANK_CONFIG["match_weight"])

    for s in sources:
        score = float(s.get("score", 0.0))
        matched_len = len(s.get("matched_terms", []))
        final = score * score_weight + matched_len * match_weight
        scored.append((final, s))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in scored]

# ==============================
# 答案评估模块
# ==============================

def evaluate_answer(answer: str, sources: list[dict]) -> dict[str, Any]:
    """评估回答是否满足基本要求。"""
    if EVALUATION_CONFIG["require_sources"] and not sources:
        return {"valid": False, "reason": "无来源依据"}

    if len(answer.strip()) < int(EVALUATION_CONFIG["min_answer_length"]):
        return {"valid": False, "reason": "回答过短"}

    return {"valid": True, "reason": EVALUATION_CONFIG["fallback_reason_ok"]}

# ==============================
# 回答增强
# ==============================

def build_answer_context(result: AgentAnswer) -> str:
    """为问答模型构造来源上下文。"""
    context_lines: list[str] = []
    for index, source in enumerate(result.sources, start=1):
        page_text = format_page_numbers(source.get("page_numbers", []))
        file_name = str(source.get("file_name", "未知论文"))
        snippet = str(source.get("snippet", ""))
        context_lines.append(
            f"[来源{index}] 论文：{file_name} | 页码：{page_text} | 片段：{snippet}"
        )
    return "\n".join(context_lines)

def enhance_answer_with_llm(
    result: AgentAnswer,
    client: DashScopeClient | None,
    model_name: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[AgentAnswer, str | None]:
    """使用大模型润色或增强回答。"""
    if client is None:
        return result, None
    if not result.sources:
        return result, None

    system_prompt = PROMPTS["answer_system"]
    user_prompt = PROMPTS["answer_user"].format(
        question=result.question,
        context=build_answer_context(result),
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        llm_answer = client.chat(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=900,
        )
    except Exception as exc:
        return result, f"问答大模型调用失败，已回退最小规则回答：{exc}"

    result.model_answer = llm_answer
    return result, None

# ==============================
# 分析增强
# ==============================

def build_analysis_input_for_llm(full_text: str, max_chars: int = 18000) -> str:
    """截取适合送入大模型的论文正文。"""
    text = full_text.strip()
    if not text:
        return text

    lowered = text.lower()
    reference_markers = ["\nreferences", "\nreference", "\nbibliography", "\nacknowledgments"]
    cut_positions: list[int] = []

    for marker in reference_markers:
        index = lowered.find(marker)
        if index != -1:
            cut_positions.append(index)

    if cut_positions:
        text = text[: min(cut_positions)].strip()

    if len(text) <= max_chars:
        return text

    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.35):]
    return f"{head}\n\n[...中间内容已省略...]\n\n{tail}"

def pick_text_field(data: dict[str, Any], key: str, fallback: str) -> str:
    """从 JSON 中提取字符串字段。"""
    value = data.get(key, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback

def normalize_evidence_map(data: dict[str, Any]) -> dict[str, list[str]]:
    """规范化 evidence_map 字段。"""
    field_keys = [
        "research_question",
        "research_object",
        "methods",
        "data_source",
        "key_findings",
        "limitations",
        "implications",
    ]

    evidence_map: dict[str, list[str]] = {}
    raw_map = data.get("evidence_map", {})
    if not isinstance(raw_map, dict):
        raw_map = {}

    for field_key in field_keys:
        raw_value = raw_map.get(field_key, [])
        if isinstance(raw_value, list):
            normalized_list = [str(item).strip() for item in raw_value if str(item).strip()]
        elif isinstance(raw_value, str):
            normalized_list = [raw_value.strip()] if raw_value.strip() else []
        else:
            normalized_list = []
        evidence_map[field_key] = normalized_list

    return evidence_map

def enhance_analysis_with_llm(
    agent: CityScholarAgent,
    target: str | None,
    client: DashScopeClient | None,
    model_name: str,
) -> tuple[PaperAnalysisResponse, str | None]:
    """使用大模型增强论文分析结果。"""
    base_result = agent.analyze_paper(target)
    if client is None or base_result.analysis is None:
        return base_result, None

    document = agent.find_document(target)
    if document is None:
        return base_result, None

    llm_input = build_analysis_input_for_llm(document.full_text)
    if not llm_input:
        return base_result, None

    system_prompt = PROMPTS["analysis_system"]
    user_prompt = PROMPTS["analysis_user"].format(
        file_name=document.file_name,
        document_id=document.document_id,
        paper_text=llm_input,
    )

    try:
        raw_text = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
    except Exception as exc:
        return base_result, f"学术分析大模型调用失败，已回退规则分析：{exc}"

    json_data = parse_first_json_object(raw_text)
    if json_data is None:
        return base_result, "学术分析大模型输出非 JSON，已回退规则分析。"

    fallback_analysis = base_result.analysis
    merged_analysis = StructuredPaperAnalysis(
        file_name=fallback_analysis.file_name,
        document_id=fallback_analysis.document_id,
        research_question=pick_text_field(json_data, "research_question", fallback_analysis.research_question),
        research_object=pick_text_field(json_data, "research_object", fallback_analysis.research_object),
        methods=pick_text_field(json_data, "methods", fallback_analysis.methods),
        data_source=pick_text_field(json_data, "data_source", fallback_analysis.data_source),
        key_findings=pick_text_field(json_data, "key_findings", fallback_analysis.key_findings),
        limitations=pick_text_field(json_data, "limitations", fallback_analysis.limitations),
        implications=pick_text_field(json_data, "implications", fallback_analysis.implications),
        evidence_map=normalize_evidence_map(json_data),
    )

    enhanced_result = PaperAnalysisResponse(
        target=base_result.target,
        status_message=f"{base_result.status_message}（已使用大模型增强：{model_name}）",
        analysis=merged_analysis,
        formatted_output=format_analysis_result(merged_analysis),
    )
    return enhanced_result, None

# ==============================
# 展示函数
# ==============================

def display_answer(result: AgentAnswer) -> None:
    """展示问答结果。"""
    print("\n模型回答：")
    print(result.model_answer)
    print("\n来源依据：")
    if not result.sources:
        print("当前没有可展示的来源依据。")
        return

    for index, source in enumerate(result.sources, start=1):
        print(f"{index}. 论文：{source.get('file_name', '未知论文')}")
        print(f"   页码：{format_page_numbers(source.get('page_numbers', []))}")
        print(f"   片段编号：{source.get('chunk_id', '未知')}")
        print(f"   匹配词：{', '.join(source.get('matched_terms', [])) if source.get('matched_terms') else '无'}")
        print(f"   相关分数：{source.get('score', 0.0)}")
        print(f"   片段内容：{source.get('snippet', '')}")
        print(f"   文件路径：{source.get('source_path', '')}")

def display_analysis(result: PaperAnalysisResponse) -> None:
    """展示论文分析结果。"""
    print("\n学术分析：")
    print(result.status_message)
    print(result.formatted_output)

def display_comparison(result: PaperComparisonResponse) -> None:
    """展示论文比较结果。"""
    print("\n多篇比较：")
    print(result.status_message)
    print(result.formatted_output)

def display_outline(result: ReviewOutlineResponse) -> None:
    """展示综述提纲。"""
    print("\n综述提纲：")
    print(result.status_message)
    print(result.formatted_output)

def display_workflow(result: WorkflowResponse) -> None:
    """展示工作流结果。"""
    print("\n多步工作流：")
    print(result.status_message)
    print(result.formatted_output)

def display_embedding_index_status(result: EmbeddingIndexResponse) -> None:
    """展示向量索引状态。"""
    print("\n向量索引：")
    print(result.status_message)
    print(f"索引路径：{result.index_path}")
    print(f"向量数量：{result.vector_count}")
    print(f"是否缓存加载：{'是' if result.loaded_from_cache else '否'}")

def display_paper_list(agent: CityScholarAgent) -> None:
    """展示当前可用论文列表。"""
    paper_items = agent.list_available_papers()
    print("\n当前可用论文：")
    if not paper_items:
        print("当前没有可用论文。")
        return
    for item in paper_items:
        print(
            f"{item['index']}. {item['file_name']} | 文档编号：{item['document_id']} | "
            f"页数：{item['total_pages']} | 字符数：{item['total_characters']}"
        )

def display_history(state: AgentState, n: int = 10) -> None:
    """展示最近对话历史。"""
    messages = state.history.latest(n)
    print(f"\n最近 {len(messages)} 条对话：")
    if not messages:
        print("当前没有历史记录。")
        return

    for i, msg in enumerate(messages, start=1):
        role = "用户" if msg["role"] == "user" else "助手"
        print(f"{i}. [{role}] {msg['content']}")

def clear_history(state: AgentState) -> None:
    """清空对话历史。"""
    state.history.clear()
    print("\n已清空对话历史。")

def display_stats(
    agent: CityScholarAgent,
    knowledge_base: KnowledgeBaseState,
    embedding_status: EmbeddingIndexResponse | None,
    llm_client: DashScopeClient | None,
) -> None:
    """展示当前运行状态。"""
    print("\n当前状态：")
    print(f"- PDF 数量：{len(knowledge_base.pdf_files)}")
    print(f"- 解析成功论文数：{len(knowledge_base.documents)}")
    print(f"- 可检索片段数：{len(knowledge_base.chunk_records)}")
    print(f"- 解析失败论文数：{len(knowledge_base.parse_errors)}")
    print(f"- 当前可用论文：{len(agent.list_available_papers())}")
    print(f"- 大模型增强：{'已启用' if llm_client is not None else '未启用'}")
    if embedding_status is not None:
        print(f"- 向量索引：{embedding_status.status_message}")
        print(f"- 索引路径：{embedding_status.index_path}")
        print(f"- 向量数量：{embedding_status.vector_count}")
        print(f"- 是否缓存加载：{'是' if embedding_status.loaded_from_cache else '否'}")

def display_config(config: dict[str, Path | str | int | bool], llm_client: DashScopeClient | None) -> None:
    """展示当前配置。"""
    print("\n当前配置：")
    print(f"- 项目名：{config.get('project_name')}")
    print(f"- 项目目录：{config.get('base_dir')}")
    print(f"- 论文目录：{config.get('raw_papers_dir')}")
    print(f"- 处理目录：{config.get('processed_data_dir')}")
    print(f"- 输出目录：{config.get('output_dir')}")
    print(f"- 问答模型：{config.get('dashscope_answer_model')}")
    print(f"- 分析模型：{config.get('dashscope_analysis_model')}")
    print(f"- 向量模型：{config.get('dashscope_embedding_model')}")
    print(f"- 大模型增强：{'已启用' if llm_client is not None else '未启用'}")
    print(f"- 向量维度：{config.get('dashscope_embedding_dimensions')}")
    print(f"- 超时时间：{config.get('dashscope_timeout_sec')}")

def display_route(route: RouteDecision) -> None:
    """展示路由结果。"""
    print("\n路由结果：")
    print(f"- intent: {route.intent}")
    print(f"- targets: {route.targets}")
    print(f"- topic: {route.topic}")
    print(f"- raw_text: {route.raw_text}")

def display_last_answer(state: AgentState) -> None:
    """展示上一次回答。"""
    if state.output is None:
        print("\n当前还没有回答记录。")
        return
    print("\n上一次模型回答：")
    print(state.output.model_answer)

def export_history(state: AgentState, output_dir: Path) -> Path:
    """导出对话历史。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with export_path.open("w", encoding="utf-8") as f:
        for msg in state.history.to_list():
            role = "用户" if msg["role"] == "user" else "助手"
            f.write(f"[{role}] {msg['content']}\n\n")

    return export_path

def save_last_answer(state: AgentState, output_dir: Path) -> Path | None:
    """保存上一次回答到文件。"""
    if state.output is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"last_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(state.output.model_answer)
    return path

# ==============================
# 旧解析函数：保留兜底兼容
# ==============================

def parse_analyze_target(user_input: str) -> str | None:
    """解析 analyze 命令中的目标。"""
    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[1].strip() or None

def parse_target_list(target_text: str) -> list[str]:
    """解析目标列表。"""
    cleaned_text = target_text.replace("，", ",").strip()
    if not cleaned_text:
        return []
    if "," in cleaned_text:
        return [item.strip() for item in cleaned_text.split(",") if item.strip()]
    return [item.strip() for item in cleaned_text.split() if item.strip()]

def parse_compare_targets(user_input: str) -> list[str] | None:
    """解析 compare 命令中的目标列表。"""
    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None
    targets = parse_target_list(parts[1])
    return targets or None

def parse_outline_request(user_input: str) -> tuple[list[str] | None, str]:
    """解析 outline 命令中的目标和主题。"""
    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None, ""
    payload = parts[1].strip()
    if "::" not in payload:
        return None, payload
    target_text, topic = payload.split("::", maxsplit=1)
    targets = parse_target_list(target_text)
    return (targets or None), topic.strip()

def parse_workflow_request(user_input: str) -> tuple[list[str] | None, str]:
    """解析 workflow 命令中的目标和主题。"""
    return parse_outline_request(user_input)

def rescan_knowledge_base(agent: CityScholarAgent) -> KnowledgeBaseState:
    """重新扫描知识库。"""
    return agent.build_knowledge_base()

# ==============================
# 执行器
# ==============================

def handle_build_index(
    agent: CityScholarAgent,
    llm_client: DashScopeClient | None,
    embedding_model: str,
    embedding_dimensions: int,
    processed_data_dir: Path,
    force_rebuild: bool = False,
) -> EmbeddingIndexResponse:
    """构建或重建向量索引。"""
    embedding_result = agent.prepare_embedding_index(
        client=llm_client,
        model_name=embedding_model,
        dimensions=embedding_dimensions,
        processed_data_dir=processed_data_dir,
        build_if_missing=True,
        force_rebuild=force_rebuild,
    )
    display_embedding_index_status(embedding_result)
    return embedding_result

def handle_analyze(
    agent: CityScholarAgent,
    route: RouteDecision,
    user_input: str,
    llm_client: DashScopeClient | None,
    analysis_model: str,
    state: AgentState,
) -> None:
    """执行论文分析。"""
    analyze_target = route.targets[0] if route.targets else parse_analyze_target(user_input)

    analysis_result, warning = enhance_analysis_with_llm(
        agent=agent,
        target=analyze_target,
        client=llm_client,
        model_name=analysis_model,
    )
    if warning:
        print(f"提示：{warning}")
    display_analysis(analysis_result)
    state.history.add("assistant", analysis_result.formatted_output)
    state.output = AgentAnswer(question=user_input, model_answer=analysis_result.formatted_output, sources=[])

def handle_compare(
    agent: CityScholarAgent,
    route: RouteDecision,
    user_input: str,
    state: AgentState,
) -> None:
    """执行论文比较。"""
    compare_targets = route.targets or parse_compare_targets(user_input)
    comparison_result = agent.compare_papers(compare_targets)
    display_comparison(comparison_result)
    state.history.add("assistant", comparison_result.formatted_output)

def handle_outline(
    agent: CityScholarAgent,
    route: RouteDecision,
    user_input: str,
    state: AgentState,
) -> None:
    """执行综述提纲生成。"""
    outline_targets = route.targets or None
    outline_topic = route.topic

    if not outline_topic:
        old_targets, old_topic = parse_outline_request(user_input)
        outline_targets = outline_targets or old_targets
        outline_topic = old_topic

    if not outline_topic:
        print("请输入综述主题，例如：outline 城市韧性研究综述")
        return

    outline_result = agent.generate_review_outline(
        topic=outline_topic,
        targets=outline_targets,
    )
    display_outline(outline_result)
    state.history.add("assistant", outline_result.formatted_output)

def handle_workflow(
    agent: CityScholarAgent,
    route: RouteDecision,
    user_input: str,
    state: AgentState,
) -> None:
    """执行多步综述工作流。"""
    workflow_targets = route.targets or None
    workflow_topic = route.topic

    if not workflow_topic:
        old_targets, old_topic = parse_workflow_request(user_input)
        workflow_targets = workflow_targets or old_targets
        workflow_topic = old_topic

    if not workflow_topic:
        print("请输入工作流主题，例如：workflow 城市韧性研究综述")
        return

    workflow_result = agent.run_review_workflow(
        topic=workflow_topic,
        targets=workflow_targets,
    )
    display_workflow(workflow_result)
    state.history.add("assistant", workflow_result.formatted_output)

def handle_question(
    agent: CityScholarAgent,
    user_input: str,
    llm_client: DashScopeClient | None,
    answer_model: str,
    state: AgentState,
) -> None:
    """执行普通检索问答。"""
    answer_result = agent.answer(user_input)
    answer_result.sources = rerank_sources(answer_result.sources or [])
    answer_result, warning = enhance_answer_with_llm(
        answer_result,
        llm_client,
        answer_model,
        history=state.history.latest(6),
    )
    eval_result = evaluate_answer(answer_result.model_answer, answer_result.sources)

    if warning:
        print(f"提示：{warning}")
    print(f"\n【智能体评估】{eval_result}")
    display_answer(answer_result)

    state.history.add("assistant", answer_result.model_answer)
    state.evaluation = eval_result
    state.output = answer_result
    state.warning = warning

# ==============================
# 主对话循环
# ==============================

def run_cli_chat(
    agent: CityScholarAgent,
    llm_client: DashScopeClient | None,
    answer_model: str,
    analysis_model: str,
    embedding_model: str,
    embedding_dimensions: int,
    processed_data_dir: Path,
    config: dict[str, Path | str | int | bool],
    knowledge_base: KnowledgeBaseState,
    embedding_status: EmbeddingIndexResponse | None,
) -> None:
    """运行命令行对话循环。"""
    state = AgentState()
    print("✅ 智能体已启动 | 结构化路由/对话记忆/检索路由/重排/评估已启用")

    while True:
        try:
            user_input = input("\n请输入你的问题或命令：").strip()
        except EOFError:
            print("\n检测到输入结束，程序已退出。")
            break
        except KeyboardInterrupt:
            print("\n检测到手动中断，程序已退出。")
            break

        if not user_input:
            print("请输入有效问题或命令，输入 help 查看说明。")
            continue

        route = resolve_route(user_input, llm_client, answer_model)
        state.user_input = user_input
        state.intent = route.intent
        state.route = route

        if route.intent == "exit":
            print("程序已退出，欢迎下次继续使用。")
            break

        if route.intent == "help":
            show_cli_help()
            continue

        if route.intent == "papers":
            try:
                display_paper_list(agent)
            except Exception as exc:
                print(f"读取论文列表失败：{exc}")
            continue

        if route.intent == "build_index":
            try:
                embedding_status = handle_build_index(
                    agent=agent,
                    llm_client=llm_client,
                    embedding_model=embedding_model,
                    embedding_dimensions=embedding_dimensions,
                    processed_data_dir=processed_data_dir,
                    force_rebuild=False,
                )
            except Exception as exc:
                print(f"向量索引构建失败：{exc}")
            continue

        if route.intent == "rebuild_index":
            try:
                embedding_status = handle_build_index(
                    agent=agent,
                    llm_client=llm_client,
                    embedding_model=embedding_model,
                    embedding_dimensions=embedding_dimensions,
                    processed_data_dir=processed_data_dir,
                    force_rebuild=True,
                )
            except Exception as exc:
                print(f"向量索引重建失败：{exc}")
            continue

        if route.intent == "reload_index":
            try:
                print("\n正在重载向量索引，请稍候...")
                embedding_status = agent.prepare_embedding_index(
                    client=llm_client,
                    model_name=embedding_model,
                    dimensions=embedding_dimensions,
                    processed_data_dir=processed_data_dir,
                    build_if_missing=True,
                    force_rebuild=False,
                )
                display_embedding_index_status(embedding_status)
            except Exception as exc:
                print(f"重载索引失败：{exc}")
            continue

        if route.intent == "rescan":
            try:
                print("\n正在重新扫描知识库，请稍候...")
                knowledge_base = rescan_knowledge_base(agent)
                print("重新扫描完成。")
                print(f"- PDF 数量：{len(knowledge_base.pdf_files)}")
                print(f"- 解析成功论文数：{len(knowledge_base.documents)}")
                print(f"- 可检索片段数：{len(knowledge_base.chunk_records)}")
                print(f"- 解析失败论文数：{len(knowledge_base.parse_errors)}")

                if llm_client is not None and knowledge_base.chunk_records:
                    embedding_status = agent.prepare_embedding_index(
                        client=llm_client,
                        model_name=embedding_model,
                        dimensions=embedding_dimensions,
                        processed_data_dir=processed_data_dir,
                        build_if_missing=False,
                        force_rebuild=False,
                    )
            except Exception as exc:
                print(f"重新扫描失败：{exc}")
            continue

        if route.intent == "history":
            display_history(state, n=10)
            continue

        if route.intent == "clear":
            clear_history(state)
            continue

        if route.intent == "stats":
            try:
                display_stats(agent, knowledge_base, embedding_status, llm_client)
            except Exception as exc:
                print(f"获取状态失败：{exc}")
            continue

        if route.intent == "config":
            try:
                display_config(config, llm_client)
            except Exception as exc:
                print(f"读取配置失败：{exc}")
            continue

        if route.intent == "export_history":
            try:
                export_path = export_history(state, Path(config["output_dir"]))
                print(f"\n对话历史已导出到：{export_path}")
            except Exception as exc:
                print(f"导出历史失败：{exc}")
            continue

        if route.intent == "route":
            display_route(route)
            continue

        if route.intent == "last":
            display_last_answer(state)
            continue

        if route.intent == "save_answer":
            try:
                saved_path = save_last_answer(state, Path(config["output_dir"]))
                if saved_path is None:
                    print("\n当前没有可保存的回答。")
                else:
                    print(f"\n上一次回答已保存到：{saved_path}")
            except Exception as exc:
                print(f"保存回答失败：{exc}")
            continue

        # 任务型对话统一记录历史
        state.history.add("user", user_input)

        if route.intent == "analyze":
            try:
                handle_analyze(
                    agent=agent,
                    route=route,
                    user_input=user_input,
                    llm_client=llm_client,
                    analysis_model=analysis_model,
                    state=state,
                )
            except Exception as exc:
                print(f"学术分析过程出现异常：{exc}")
            continue

        if route.intent == "compare":
            try:
                handle_compare(
                    agent=agent,
                    route=route,
                    user_input=user_input,
                    state=state,
                )
            except ValueError as exc:
                print(f"多篇比较输入有误：{exc}")
            except Exception as exc:
                print(f"多篇比较过程出现异常：{exc}")
            continue

        if route.intent == "outline":
            try:
                handle_outline(
                    agent=agent,
                    route=route,
                    user_input=user_input,
                    state=state,
                )
            except ValueError as exc:
                print(f"综述提纲输入有误：{exc}")
            except Exception as exc:
                print(f"综述提纲生成过程出现异常：{exc}")
            continue

        if route.intent == "workflow":
            try:
                handle_workflow(
                    agent=agent,
                    route=route,
                    user_input=user_input,
                    state=state,
                )
            except ValueError as exc:
                print(f"多步工作流输入有误：{exc}")
            except Exception as exc:
                print(f"多步工作流执行过程出现异常：{exc}")
            continue

        # 默认 question / other
        try:
            handle_question(
                agent=agent,
                user_input=user_input,
                llm_client=llm_client,
                answer_model=answer_model,
                state=state,
            )
        except Exception as exc:
            print(f"问答异常：{exc}")
            continue

# ==============================
# 程序入口
# ==============================

def main() -> None:
    """程序主入口。"""
    config = get_app_config()

    ensure_directories([
        config["data_dir"],
        config["raw_papers_dir"],
        config["processed_data_dir"],
        config["output_dir"],
    ])

    llm_client = initialize_llm_client(config)
    answer_model = str(config.get("dashscope_answer_model", "qwen-plus"))
    analysis_model = str(config.get("dashscope_analysis_model", "qwen-max"))
    embedding_model = str(config.get("dashscope_embedding_model", "text-embedding-v3"))
    embedding_dimensions = int(config.get("dashscope_embedding_dimensions", 128))

    agent = CityScholarAgent(raw_papers_dir=config["raw_papers_dir"])

    try:
        knowledge_base = agent.build_knowledge_base()
    except ImportError as exc:
        print(f"依赖缺失：{exc}")
        return
    except Exception as exc:
        print(f"知识库构建失败：{exc}")
        return

    embedding_status: EmbeddingIndexResponse | None = None
    if llm_client is not None and knowledge_base.chunk_records:
        try:
            embedding_status = agent.prepare_embedding_index(
                client=llm_client,
                model_name=embedding_model,
                dimensions=embedding_dimensions,
                processed_data_dir=config["processed_data_dir"],
                build_if_missing=False,
                force_rebuild=False,
            )
        except Exception as exc:
            embedding_status = EmbeddingIndexResponse(
                status_message=f"向量索引初始化失败：{exc}",
                index_path="",
                vector_count=0,
                loaded_from_cache=False,
            )

    show_startup_info(config, knowledge_base, llm_client, embedding_status)
    show_parse_error_summary(knowledge_base.parse_errors)

    if not knowledge_base.pdf_files:
        print("当前未在 data/raw_papers/ 发现 PDF，暂时无法进入交互。")
        print("请先放入论文文件后重新运行。")
        return

    if not knowledge_base.documents:
        print("当前没有可用论文被成功解析，暂时无法进行问答或分析。")
        return

    if not knowledge_base.chunk_records:
        print("提示：当前没有可检索片段，问答可能无法返回有效结果。")

    run_cli_chat(
        agent=agent,
        llm_client=llm_client,
        answer_model=answer_model,
        analysis_model=analysis_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        processed_data_dir=config["processed_data_dir"],
        config=config,
        knowledge_base=knowledge_base,
        embedding_status=embedding_status,
    )

if __name__ == "__main__":
    main()
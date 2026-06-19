"""本模块作用：在整个智能体中负责对多篇论文做最小结构化比较，帮助系统完成“多工具调用”中的论文对比任务。"""

import re
from collections import Counter
from dataclasses import dataclass, field

from tools.analyze_tool import StructuredPaperAnalysis, analyze_single_paper


COMPARE_STOP_WORDS = {
    "研究",
    "城市",
    "方法",
    "结果",
    "数据",
    "问题",
    "对象",
    "分析",
    "影响",
    "表明",
    "指出",
    "采用",
    "基于",
    "本文",
    "论文",
    "以及",
    "进行",
    "不同",
    "相关",
    "对于",
    "and",
    "the",
    "this",
    "that",
    "these",
    "those",
    "into",
    "from",
    "have",
    "been",
    "also",
    "more",
    "such",
    "with",
    "using",
    "based",
    "paper",
    "study",
    "urban",
    "are",
    "was",
    "were",
    "their",
    "them",
    "both",
    "between",
    "across",
    "through",
}


@dataclass
class ComparisonPaperSummary:
    """本数据结构作用：保存单篇论文在比较任务中的精简摘要。"""

    file_name: str
    document_id: str
    research_question: str
    research_object: str
    methods: str
    data_source: str
    key_findings: str
    limitations: str
    implications: str


@dataclass
class MultiPaperComparison:
    """本数据结构作用：保存多篇论文比较任务的结构化结果。"""

    topic_hint: str
    paper_summaries: list[ComparisonPaperSummary] = field(default_factory=list)
    common_themes: list[str] = field(default_factory=list)
    method_comparison: list[str] = field(default_factory=list)
    data_comparison: list[str] = field(default_factory=list)
    finding_comparison: list[str] = field(default_factory=list)
    integrated_implications: list[str] = field(default_factory=list)


def normalize_compare_text(text: str) -> str:
    """对比较任务中的文本做基础清洗。

    输入：
        text: 原始文本。
    输出：
        统一空白后的文本。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text


def truncate_compare_text(text: str, max_length: int = 120) -> str:
    """将比较结果中的长文本裁剪为适合展示的长度。

    输入：
        text: 原始文本。
        max_length: 最大保留字符数。
    输出：
        裁剪后的文本。
    异常：
        无。
    """

    cleaned_text = normalize_compare_text(text)
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def extract_compare_keywords(text: str) -> list[str]:
    """从比较文本中提取最小关键词。

    输入：
        text: 原始文本。
    输出：
        去重后的关键词列表。
    异常：
        无。
    """

    normalized_text = normalize_compare_text(text).lower()
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized_text)
    english_terms = re.findall(r"[a-z]{3,}", normalized_text)

    keywords: list[str] = []
    seen_terms: set[str] = set()
    for term in chinese_terms + english_terms:
        if term in COMPARE_STOP_WORDS or term in seen_terms:
            continue
        seen_terms.add(term)
        keywords.append(term)
    return keywords


def build_paper_summary(analysis: StructuredPaperAnalysis) -> ComparisonPaperSummary:
    """将单篇论文分析结果转换为多篇比较所需的摘要结构。

    输入：
        analysis: 单篇论文结构化分析结果。
    输出：
        适用于比较任务的单篇摘要对象。
    异常：
        无。
    """

    return ComparisonPaperSummary(
        file_name=analysis.file_name,
        document_id=analysis.document_id,
        research_question=analysis.research_question,
        research_object=analysis.research_object,
        methods=analysis.methods,
        data_source=analysis.data_source,
        key_findings=analysis.key_findings,
        limitations=analysis.limitations,
        implications=analysis.implications,
    )


def collect_common_themes(analyses: list[StructuredPaperAnalysis], max_items: int = 4) -> list[str]:
    """从多篇论文中提取共同主题。

    输入：
        analyses: 多篇论文的结构化分析结果。
        max_items: 最多返回多少条共同主题。
    输出：
        共同主题列表。
    异常：
        无。
    """

    counter: Counter[str] = Counter()
    for analysis in analyses:
        merged_text = " ".join(
            [
                analysis.research_question,
                analysis.research_object,
                analysis.key_findings,
                analysis.implications,
            ]
        )
        counter.update(extract_compare_keywords(merged_text))

    common_terms = [term for term, count in counter.items() if count >= 2 and len(term) >= 4][:max_items]
    if not common_terms:
        return ["当前几篇论文的共同主题不够集中，更适合按方法、对象和发现分别比较。"]
    return [f"多篇论文都涉及“{term}”相关议题。" for term in common_terms]


def build_method_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """生成多篇论文的方法对比结果。

    输入：
        analyses: 多篇论文的结构化分析结果。
    输出：
        方法对比文本列表。
    异常：
        无。
    """

    return [f"《{item.file_name}》：{truncate_compare_text(item.methods)}" for item in analyses]


def build_data_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """生成多篇论文的数据来源对比结果。

    输入：
        analyses: 多篇论文的结构化分析结果。
    输出：
        数据来源对比文本列表。
    异常：
        无。
    """

    return [f"《{item.file_name}》：{truncate_compare_text(item.data_source)}" for item in analyses]


def build_finding_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """生成多篇论文的主要发现对比结果。

    输入：
        analyses: 多篇论文的结构化分析结果。
    输出：
        主要发现对比文本列表。
    异常：
        无。
    """

    return [f"《{item.file_name}》：{truncate_compare_text(item.key_findings)}" for item in analyses]


def build_integrated_implications(analyses: list[StructuredPaperAnalysis], max_items: int = 4) -> list[str]:
    """汇总多篇论文对城市治理、规划或安全的综合启示。

    输入：
        analyses: 多篇论文的结构化分析结果。
        max_items: 最多返回多少条综合启示。
    输出：
        综合启示列表。
    异常：
        无。
    """

    implication_items: list[str] = []
    seen_items: set[str] = set()
    for analysis in analyses:
        candidate = truncate_compare_text(analysis.implications, max_length=150)
        if candidate in seen_items:
            continue
        seen_items.add(candidate)
        implication_items.append(f"《{analysis.file_name}》提示：{candidate}")
        if len(implication_items) >= max_items:
            break

    if implication_items:
        return implication_items
    return ["当前未能稳定归纳出综合启示，建议回看单篇分析中的“启示”字段。"]


def compare_papers(
    papers: list[dict[str, str]],
    topic_hint: str = "未指定比较主题",
) -> MultiPaperComparison:
    """对多篇论文执行最小结构化比较。

    输入：
        papers: 待比较论文列表，每项至少包含 file_name、document_id 和 full_text。
        topic_hint: 可选的比较主题提示。
    输出：
        多篇论文比较结果对象。
    异常：
        当论文数量不足两篇时，抛出 ValueError。
        当论文文本为空时，可能抛出 ValueError。
    """

    if len(papers) < 2:
        raise ValueError("多篇比较至少需要 2 篇论文。")

    analyses: list[StructuredPaperAnalysis] = []
    summaries: list[ComparisonPaperSummary] = []
    for paper in papers:
        analysis = analyze_single_paper(
            text_or_segments=paper["full_text"],
            file_name=paper["file_name"],
            document_id=paper["document_id"],
        )
        analyses.append(analysis)
        summaries.append(build_paper_summary(analysis))

    return MultiPaperComparison(
        topic_hint=topic_hint,
        paper_summaries=summaries,
        common_themes=collect_common_themes(analyses),
        method_comparison=build_method_comparison(analyses),
        data_comparison=build_data_comparison(analyses),
        finding_comparison=build_finding_comparison(analyses),
        integrated_implications=build_integrated_implications(analyses),
    )


def format_comparison_result(result: MultiPaperComparison) -> str:
    """将多篇论文比较结果整理为适合命令行展示的文本。

    输入：
        result: 多篇论文比较结果对象。
    输出：
        可直接打印的比较结果文本。
    异常：
        无。
    """

    lines = [
        "多篇论文比较结果：",
        f"比较主题：{result.topic_hint}",
        f"纳入论文数量：{len(result.paper_summaries)}",
        "纳入论文：",
    ]

    for index, summary in enumerate(result.paper_summaries, start=1):
        lines.append(f"{index}. 《{summary.file_name}》")
        lines.append(f"   研究问题：{truncate_compare_text(summary.research_question)}")
        lines.append(f"   研究对象：{truncate_compare_text(summary.research_object)}")
        lines.append(f"   方法：{truncate_compare_text(summary.methods)}")

    lines.append("共同主题：")
    for item in result.common_themes:
        lines.append(f"- {item}")

    lines.append("方法比较：")
    for item in result.method_comparison:
        lines.append(f"- {item}")

    lines.append("数据来源比较：")
    for item in result.data_comparison:
        lines.append(f"- {item}")

    lines.append("主要发现比较：")
    for item in result.finding_comparison:
        lines.append(f"- {item}")

    lines.append("综合启示：")
    for item in result.integrated_implications:
        lines.append(f"- {item}")

    return "\n".join(lines)


def run_compare_demo() -> None:
    """执行 compare_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印多篇比较结果。
    异常：
        无。
    """

    demo_papers = [
        {
            "file_name": "paper_a.pdf",
            "document_id": "paper_a",
            "full_text": (
                "本文关注城市韧性评估，研究对象为沿海城市群。"
                "研究采用指标评价与空间分析方法。"
                "数据来源包括统计年鉴和遥感数据。"
                "结果表明，基础设施韧性与治理协同能力显著相关。"
                "研究建议提升跨区域协同治理能力。"
            ),
        },
        {
            "file_name": "paper_b.pdf",
            "document_id": "paper_b",
            "full_text": (
                "研究聚焦城市安全治理与韧性提升，案例为内陆都市圈。"
                "方法上采用问卷调查与回归分析。"
                "数据来源包括问卷、POI 数据与统计资料。"
                "结果指出公共服务可达性会影响城市安全感知。"
                "论文建议把公共服务规划与安全治理协同推进。"
            ),
        },
    ]

    result = compare_papers(demo_papers, topic_hint="城市韧性与安全治理")
    print(format_comparison_result(result))


if __name__ == "__main__":
    run_compare_demo()

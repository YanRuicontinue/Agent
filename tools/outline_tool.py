"""本模块作用：在整个智能体中负责根据若干论文生成最小综述提纲，帮助系统完成“工具调用与模块化”阶段的提纲生成任务。"""

from dataclasses import dataclass, field

from tools.compare_tool import MultiPaperComparison, compare_papers


@dataclass
class ReviewOutlineSection:
    """本数据结构作用：保存综述提纲中的单个章节。"""

    title: str
    bullets: list[str] = field(default_factory=list)


@dataclass
class ReviewOutline:
    """本数据结构作用：保存综述提纲生成任务的结构化结果。"""

    topic: str
    source_papers: list[str] = field(default_factory=list)
    sections: list[ReviewOutlineSection] = field(default_factory=list)


def build_section(
    title: str,
    bullets: list[str],
    default_bullet: str,
    max_items: int = 4,
) -> ReviewOutlineSection:
    """构建综述提纲中的单个章节。

    输入：
        title: 章节标题。
        bullets: 候选要点列表。
        default_bullet: 当候选为空时使用的默认要点。
        max_items: 最多保留多少条要点。
    输出：
        单个综述章节对象。
    异常：
        无。
    """

    cleaned_items = [item.strip() for item in bullets if item and item.strip()]
    if not cleaned_items:
        cleaned_items = [default_bullet]
    return ReviewOutlineSection(title=title, bullets=cleaned_items[:max_items])


def build_outline_sections(topic: str, comparison: MultiPaperComparison) -> list[ReviewOutlineSection]:
    """根据多篇比较结果生成综述提纲章节。

    输入：
        topic: 综述主题。
        comparison: 多篇论文比较结果。
    输出：
        综述提纲章节列表。
    异常：
        无。
    """

    sections = [
        build_section(
            title="研究背景与问题提出",
            bullets=[
                f"围绕“{topic}”梳理研究缘起、应用场景与现实问题。",
                *comparison.common_themes,
            ],
            default_bullet="先界定研究主题的现实背景，再说明为什么需要综述。",
        ),
        build_section(
            title="研究对象与案例场景",
            bullets=[
                f"《{item.file_name}》关注：{item.research_object}"
                for item in comparison.paper_summaries
            ],
            default_bullet="概括不同论文关注的城市、区域、社区或系统类型。",
        ),
        build_section(
            title="常用方法与数据来源",
            bullets=comparison.method_comparison + comparison.data_comparison,
            default_bullet="整理已有研究常见的分析方法和数据来源。",
        ),
        build_section(
            title="主要发现与分歧",
            bullets=comparison.finding_comparison,
            default_bullet="比较不同论文在结论上的一致点与差异点。",
        ),
        build_section(
            title="对城市治理、规划或安全的启示",
            bullets=comparison.integrated_implications,
            default_bullet="总结研究对治理实践的可操作启示。",
        ),
        build_section(
            title="局限与后续研究方向",
            bullets=[
                f"《{item.file_name}》的局限：{item.limitations}"
                for item in comparison.paper_summaries
            ],
            default_bullet="可从样本、方法、数据和外部适用性等角度总结局限。",
        ),
    ]
    return sections


def generate_review_outline(topic: str, papers: list[dict[str, str]]) -> ReviewOutline:
    """根据若干论文生成最小综述提纲。

    输入：
        topic: 综述主题。
        papers: 参与提纲生成的论文列表，每项至少包含 file_name、document_id 和 full_text。
    输出：
        综述提纲对象。
    异常：
        当 topic 为空时，抛出 ValueError。
        当 papers 数量不足两篇时，抛出 ValueError。
    """

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        raise ValueError("综述主题不能为空。")
    if len(papers) < 2:
        raise ValueError("生成综述提纲时至少需要 2 篇论文。")

    comparison = compare_papers(papers=papers, topic_hint=cleaned_topic)

    return ReviewOutline(
        topic=cleaned_topic,
        source_papers=[item.file_name for item in comparison.paper_summaries],
        sections=build_outline_sections(cleaned_topic, comparison),
    )


def format_review_outline(result: ReviewOutline) -> str:
    """将综述提纲整理为适合命令行展示的文本。

    输入：
        result: 综述提纲对象。
    输出：
        可直接打印的提纲文本。
    异常：
        无。
    """

    lines = [
        "综述提纲：",
        f"主题：{result.topic}",
        f"参考论文：{', '.join(result.source_papers)}",
    ]

    for index, section in enumerate(result.sections, start=1):
        lines.append(f"{index}. {section.title}")
        for bullet in section.bullets:
            lines.append(f"- {bullet}")

    return "\n".join(lines)


def run_outline_demo() -> None:
    """执行 outline_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印综述提纲。
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

    result = generate_review_outline("城市韧性与安全治理研究综述", demo_papers)
    print(format_review_outline(result))


if __name__ == "__main__":
    run_outline_demo()

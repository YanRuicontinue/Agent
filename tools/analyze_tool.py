"""本模块作用：在整个智能体中负责对单篇论文全文或若干片段进行最小结构化学术分析，为教学讲解与论文理解提供统一结果。"""

import re
from dataclasses import dataclass, field


SECTION_CUTOFF_MARKERS = [
    "\nreferences",
    "\nreference",
    "\nbibliography",
    "\nacknowledg",
    "\nappendix",
]

SECTION_START_MARKERS = {
    "abstract": ["\nabstract", "\nabstract\n", "\nabstract "],
    "introduction": ["\n1. introduction", "\nintroduction"],
    "conclusion": ["\nconclusion", "\n6. conclusion", "\n7. conclusion", "\n5. conclusion"],
}


FIELD_RULES = {
    "research_question": {
        "label": "研究问题",
        "keywords": ["研究问题", "问题", "目的", "目标", "旨在", "关注", "探讨", "分析"],
        "default": "文本中未明确识别出研究问题，建议查看摘要、引言或研究设计部分。",
    },
    "research_object": {
        "label": "研究对象",
        "keywords": ["研究对象", "对象", "样本", "案例", "城市", "社区", "居民", "区域", "街区"],
        "default": "文本中未明确识别出研究对象，建议补充研究区、样本或案例描述。",
    },
    "methods": {
        "label": "方法",
        "keywords": ["方法", "采用", "构建", "模型", "访谈", "问卷", "实证", "回归", "评价", "分析方法"],
        "default": "文本中未明确识别出研究方法，建议查看方法或研究设计部分。",
    },
    "data_source": {
        "label": "数据来源",
        "keywords": ["数据", "数据来源", "样本", "年鉴", "问卷", "访谈", "遥感", "POI", "统计", "文本数据"],
        "default": "文本中未明确识别出数据来源，建议查看数据说明或样本来源部分。",
    },
    "key_findings": {
        "label": "主要结论",
        "keywords": ["结论", "发现", "结果", "表明", "说明", "影响", "显著", "提升", "降低"],
        "default": "文本中未明确识别出主要结论，建议查看结果与结论部分。",
    },
    "limitations": {
        "label": "局限性",
        "keywords": ["局限", "不足", "受限", "限制", "未来", "仍需", "但", "然而"],
        "default": "文本中未明确识别出局限性，可后续结合作者讨论部分人工补充。",
    },
    "implications": {
        "label": "对城市治理/规划/安全的启示",
        "keywords": ["启示", "建议", "治理", "规划", "安全", "政策", "管理", "优化", "决策"],
        "default": "文本中未明确识别出针对城市治理、规划或安全的启示，建议结合结论做进一步人工总结。",
    },
}

FIELD_CUE_PATTERNS = {
    "research_question": r"this paper|this study|we conduct|we review|aim|objective|purpose|focus|关注|旨在|探讨|研究",
    "research_object": r"focus on|focusing on|application of|case|sample|study area|研究对象|案例|样本|城市群|urban studies|urban resilience",
    "methods": r"method|systematic review|review|analyz|using|with the aid of|采用|方法|模型|回归|问卷|访谈",
    "data_source": r"article|articles|dataset|data|statistics|survey|poi|遥感|年鉴|数据来源|样本|233 articles",
    "key_findings": r"findings|results|show|suggest|highlight|表明|发现|结果|说明|指出",
    "limitations": r"limitation|however|future research|concern|challenge|不足|局限|未来|然而|问题",
    "implications": r"insight|implication|provide|support|guide|suggest|启示|建议|治理|规划|政策",
}


@dataclass
class StructuredPaperAnalysis:
    """本数据结构作用：保存单篇论文结构化提取结果，便于命令行展示或后续模块复用。"""

    file_name: str
    document_id: str
    research_question: str
    research_object: str
    methods: str
    data_source: str
    key_findings: str
    limitations: str
    implications: str
    evidence_map: dict[str, list[str]] = field(default_factory=dict)


def normalize_analysis_text(text: str) -> str:
    """对输入文本做基础清洗。

    输入：
        text: 原始全文或片段文本。
    输出：
        适合进一步分析的清洗后文本。
    异常：
        无。
    """

    cleaned_text = text.replace("\r", "\n")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    return cleaned_text.strip()


def remove_reference_sections(text: str) -> str:
    """删除正文后部明显属于参考文献或附录的内容。

    输入：
        text: 已做基础清洗的文本。
    输出：
        删除参考文献等后部区域后的文本。
    异常：
        无。
    """

    lowered_text = text.lower()
    cut_positions: list[int] = []
    for marker in SECTION_CUTOFF_MARKERS:
        position = lowered_text.find(marker)
        if position >= 0:
            cut_positions.append(position)

    if not cut_positions:
        return text
    return text[: min(cut_positions)].strip()


def find_first_marker_position(text: str, markers: list[str]) -> int:
    """查找一组标记在文本中的最早出现位置。

    输入：
        text: 原始文本。
        markers: 标记列表。
    输出：
        最早位置；若不存在则返回 -1。
    异常：
        无。
    """

    lowered_text = text.lower()
    positions = [lowered_text.find(marker) for marker in markers if lowered_text.find(marker) >= 0]
    if not positions:
        return -1
    return min(positions)


def extract_section_text(
    text: str,
    start_markers: list[str],
    end_markers: list[str],
    max_chars: int,
) -> str:
    """从全文中提取指定章节的近似文本窗口。

    输入：
        text: 原始全文。
        start_markers: 起始标记列表。
        end_markers: 结束标记列表。
        max_chars: 最多保留字符数。
    输出：
        提取到的章节文本；未找到时返回空字符串。
    异常：
        无。
    """

    start_position = find_first_marker_position(text, start_markers)
    if start_position < 0:
        return ""

    sliced_text = text[start_position:]
    end_position = find_first_marker_position(sliced_text, end_markers)
    if end_position > 0:
        sliced_text = sliced_text[:end_position]
    return sliced_text[:max_chars].strip()


def combine_input_text(text_or_segments: str | list[str]) -> str:
    """将单段全文或多段片段合并为统一分析文本。

    输入：
        text_or_segments: 单个字符串或字符串列表。
    输出：
        合并后的完整文本。
    异常：
        当输入既不是字符串也不是字符串列表时，抛出 TypeError。
        当输入为空时，抛出 ValueError。
    """

    if isinstance(text_or_segments, str):
        merged_text = text_or_segments
    elif isinstance(text_or_segments, list):
        merged_text = "\n".join(str(item).strip() for item in text_or_segments if str(item).strip())
    else:
        raise TypeError("text_or_segments 必须是字符串或字符串列表。")

    merged_text = normalize_analysis_text(merged_text)
    merged_text = remove_reference_sections(merged_text)
    if not merged_text:
        raise ValueError("待分析文本不能为空。")
    return merged_text


def split_into_sentences(text: str) -> list[str]:
    """将分析文本切分为句子列表。

    输入：
        text: 已清洗的文本。
    输出：
        去除空白后的句子列表。
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


def is_reference_like_sentence(sentence: str) -> bool:
    """判断句子是否更像参考文献、作者信息或出版信息噪声。

    输入：
        sentence: 候选句子文本。
    输出：
        若更像噪声句则返回 True，否则返回 False。
    异常：
        无。
    """

    normalized_sentence = sentence.strip()
    lowered_sentence = normalized_sentence.lower()

    if len(normalized_sentence) < 8:
        return True
    if lowered_sentence.startswith(("doi", "http", "www", "copyright")):
        return True
    if re.search(r"\bvol\b|\bno\b|\bpp\b|\bjournal\b|\buniversity\b|\bpress\b", lowered_sentence):
        return True
    if " et al" in lowered_sentence:
        return True
    if re.search(r"\b\d{4}\b", normalized_sentence):
        # 年份大量出现且缺少任务词时，通常更像引用信息。
        cue_count = len(
            re.findall(
                r"研究|方法|数据|结果|结论|分析|样本|案例|模型|survey|method|data|result|finding|study",
                lowered_sentence,
            )
        )
        if cue_count == 0 and re.search(r"[A-Z][a-z]+", normalized_sentence):
            return True
    if normalized_sentence.count(",") >= 3 and len(re.findall(r"\b[A-Z][a-z]+", normalized_sentence)) >= 2:
        return True
    if re.fullmatch(r"[&\sA-Za-z0-9(),.\-]+", normalized_sentence) and len(re.findall(r"\b\d{4}\b", normalized_sentence)) >= 1:
        return True
    return False


def build_analysis_sentence_pool(sentences: list[str]) -> list[str]:
    """构建用于规则分析的候选句池，过滤明显噪声。

    输入：
        sentences: 原始句子列表。
    输出：
        适合结构化提取的候选句列表。
    异常：
        无。
    """

    candidate_sentences: list[str] = []
    for sentence in sentences:
        if is_reference_like_sentence(sentence):
            continue
        candidate_sentences.append(sentence)
    return candidate_sentences


def filter_field_candidate_sentences(sentences: list[str], field_name: str) -> list[str]:
    """按字段提示词筛选更贴近目标的候选句。

    输入：
        sentences: 候选句列表。
        field_name: 字段名。
    输出：
        与该字段更相关的句子列表。
    异常：
        无。
    """

    pattern = FIELD_CUE_PATTERNS.get(field_name, "")
    if not pattern:
        return sentences

    matched_sentences = [sentence for sentence in sentences if re.search(pattern, sentence, flags=re.IGNORECASE)]
    if matched_sentences:
        return matched_sentences
    return sentences


def build_field_sentence_pool(merged_text: str, field_name: str) -> list[str]:
    """按字段构建更合适的候选句池。

    输入：
        merged_text: 已清洗且裁掉参考文献后的正文。
        field_name: 当前字段名。
    输出：
        面向该字段的候选句列表。
    异常：
        无。
    """

    abstract_text = extract_section_text(
        merged_text,
        SECTION_START_MARKERS["abstract"],
        SECTION_START_MARKERS["introduction"],
        max_chars=5000,
    )
    conclusion_text = extract_section_text(
        merged_text,
        SECTION_START_MARKERS["conclusion"],
        SECTION_CUTOFF_MARKERS,
        max_chars=4000,
    )

    # 头部窗口通常含摘要、研究问题、方法与主要发现。
    head_window = merged_text[:5000]
    # 尾部窗口通常更容易出现局限、启示与结论。
    tail_window = merged_text[-5000:]

    if field_name in {"research_question", "research_object", "methods", "data_source", "key_findings"}:
        candidate_text = "\n".join(item for item in [abstract_text, head_window] if item.strip())
    else:
        candidate_text = "\n".join(item for item in [conclusion_text, tail_window, abstract_text] if item.strip())

    sentences = split_into_sentences(candidate_text)
    filtered_sentences = build_analysis_sentence_pool(sentences)
    filtered_sentences = filter_field_candidate_sentences(filtered_sentences, field_name)
    if filtered_sentences:
        return filtered_sentences
    return sentences


def truncate_sentence(text: str, max_length: int = 120) -> str:
    """将过长句子裁剪到适合展示的长度。

    输入：
        text: 原始句子。
        max_length: 最大保留字符数。
    输出：
        裁剪后的句子。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def deduplicate_sentences(sentences: list[str]) -> list[str]:
    """按原始顺序去除重复句子。

    输入：
        sentences: 原始句子列表。
    输出：
        去重后的句子列表。
    异常：
        无。
    """

    unique_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for sentence in sentences:
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        unique_sentences.append(sentence)
    return unique_sentences


def score_sentence(sentence: str, keywords: list[str]) -> float:
    """根据关键词为句子打基础分。

    输入：
        sentence: 候选句子文本。
        keywords: 当前字段的关键词列表。
    输出：
        句子的相关性分数。
    异常：
        无。
    """

    normalized_sentence = sentence.lower()
    score = 0.0

    for keyword in keywords:
        if keyword.lower() in normalized_sentence:
            if len(keyword) >= 4:
                score += 2.0
            else:
                score += 1.2

    if re.search(r"aim|objective|purpose|focus|关注|旨在|探讨", normalized_sentence):
        score += 0.7
    if re.search(r"method|model|survey|regression|interview|questionnaire|采用|方法|模型|问卷|回归|访谈", normalized_sentence):
        score += 0.7
    if re.search(r"data|dataset|statistics|poi|遥感|年鉴|数据来源|样本", normalized_sentence):
        score += 0.7
    if re.search(r"result|finding|conclusion|表明|结果|发现|结论|说明", normalized_sentence):
        score += 0.7
    if re.search(r"limitation|future|however|局限|不足|未来|然而", normalized_sentence):
        score += 0.5
    if re.search(r"implication|policy|planning|governance|启示|建议|治理|规划|政策", normalized_sentence):
        score += 0.7

    if sentence.startswith("本文") or sentence.startswith("研究"):
        score += 0.8
    if re.search(r"study|paper|article|this study|this paper", normalized_sentence):
        score += 0.6
    if 12 <= len(sentence) <= 80:
        score += 0.5
    if len(sentence) > 150:
        score -= 0.6
    if is_reference_like_sentence(sentence):
        score -= 5.0

    return score


def select_field_evidence(
    sentences: list[str],
    keywords: list[str],
    top_k: int = 1,
) -> list[str]:
    """为某个分析字段选择最相关的依据句子。

    输入：
        sentences: 论文句子列表。
        keywords: 当前字段的关键词列表。
        top_k: 最多返回多少条依据句子。
    输出：
        去重并排序后的依据句子列表。
    异常：
        无。
    """

    scored_sentences: list[tuple[float, str]] = []
    for sentence in sentences:
        # 规则打分只做“最小可用”筛选，不追求复杂语义理解。
        score = score_sentence(sentence, keywords)
        if score <= 0:
            continue
        scored_sentences.append((score, truncate_sentence(sentence)))

    scored_sentences.sort(key=lambda item: (-item[0], item[1]))

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence in scored_sentences:
        if sentence in seen_sentences:
            continue
        selected_sentences.append(sentence)
        seen_sentences.add(sentence)
        if len(selected_sentences) >= top_k:
            break

    return selected_sentences


def build_field_summary(evidence_sentences: list[str], default_text: str) -> str:
    """将依据句子整理为字段摘要文本。

    输入：
        evidence_sentences: 当前字段的依据句子列表。
        default_text: 没有识别结果时的默认提示。
    输出：
        结构化字段摘要。
    异常：
        无。
    """

    if not evidence_sentences:
        return default_text
    return "；".join(deduplicate_sentences(evidence_sentences))


def analyze_single_paper(
    text_or_segments: str | list[str],
    file_name: str = "未知论文",
    document_id: str = "unknown_document",
) -> StructuredPaperAnalysis:
    """对单篇论文全文或若干片段执行结构化提取。

    输入：
        text_or_segments: 论文全文字符串，或若干论文片段组成的列表。
        file_name: 论文文件名。
        document_id: 论文文档编号。
    输出：
        单篇论文结构化分析结果对象。
    异常：
        当输入文本为空或类型错误时，抛出对应异常。
    """

    merged_text = combine_input_text(text_or_segments)
    sentences = split_into_sentences(merged_text)
    if not sentences:
        raise ValueError("文本过短，无法切分出有效句子进行分析。")

    evidence_map: dict[str, list[str]] = {}
    field_values: dict[str, str] = {}

    # 按统一字段顺序提取，确保输出结构稳定、便于课堂讲解与自动评测。
    for field_name, rule in FIELD_RULES.items():
        candidate_sentences = build_field_sentence_pool(merged_text, field_name)
        if not candidate_sentences:
            candidate_sentences = build_analysis_sentence_pool(sentences) or sentences
        evidence_sentences = select_field_evidence(candidate_sentences, rule["keywords"])
        evidence_map[field_name] = evidence_sentences
        field_values[field_name] = build_field_summary(evidence_sentences, rule["default"])

    return StructuredPaperAnalysis(
        file_name=file_name,
        document_id=document_id,
        research_question=field_values["research_question"],
        research_object=field_values["research_object"],
        methods=field_values["methods"],
        data_source=field_values["data_source"],
        key_findings=field_values["key_findings"],
        limitations=field_values["limitations"],
        implications=field_values["implications"],
        evidence_map=evidence_map,
    )


def format_analysis_result(result: StructuredPaperAnalysis) -> str:
    """将结构化分析结果整理为适合命令行展示的文本。

    输入：
        result: 单篇论文结构化分析结果对象。
    输出：
        带标题与分项字段的展示文本。
    异常：
        无。
    """

    lines = [
        "结构化学术分析结果：",
        f"论文名称：{result.file_name}",
        f"文档编号：{result.document_id}",
        f"1. 研究问题：{result.research_question}",
        f"2. 研究对象：{result.research_object}",
        f"3. 方法：{result.methods}",
        f"4. 数据来源：{result.data_source}",
        f"5. 主要结论：{result.key_findings}",
        f"6. 局限性：{result.limitations}",
        f"7. 对城市治理/规划/安全的启示：{result.implications}",
        "依据片段：",
    ]

    for field_name, rule in FIELD_RULES.items():
        # 输出字段值的同时附上依据片段，保障“可解释回答”。
        evidence_sentences = result.evidence_map.get(field_name, [])
        evidence_text = "；".join(evidence_sentences) if evidence_sentences else "未识别到明确依据片段。"
        lines.append(f"- {rule['label']}：{evidence_text}")

    return "\n".join(lines)


def run_analysis_demo() -> None:
    """执行分析工具的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印结构化分析结果。
    异常：
        无。
    """

    demo_text = (
        "本文以某沿海超大城市的社区更新项目为研究对象，关注公共服务设施可达性变化对居民满意度的影响。"
        "研究采用问卷调查、POI 数据分析与多元回归方法。"
        "数据来源包括 2023 年社区问卷、城市开放 POI 数据和统计年鉴。"
        "结果表明，设施步行可达性提升能够显著改善居民对社区更新的评价。"
        "然而，样本主要集中于中心城区，外部可推广性仍然受限。"
        "研究建议在城市治理与规划中，将十五分钟生活圈与道路安全整治协同推进。"
    )

    result = analyze_single_paper(
        text_or_segments=demo_text,
        file_name="demo_paper.pdf",
        document_id="demo_paper",
    )
    print(format_analysis_result(result))


if __name__ == "__main__":
    run_analysis_demo()



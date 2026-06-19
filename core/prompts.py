"""本模块作用：集中管理问答闭环中的提示语与兜底文本，为后续接入真实大模型保留统一接口。"""


def build_answer_system_prompt() -> str:
    """返回问答模块的基础回答原则。

    输入：
        无。
    输出：
        描述回答风格与约束的提示文本。
    异常：
        无。
    """

    return (
        "你是 CityScholar-Agent 的最小问答模块。"
        "请优先依据召回到的论文片段作答，不要编造未在来源中出现的结论。"
        "回答应尽量简洁、清晰，并提醒用户结合来源依据进一步核对。"
    )


def build_answer_task_prompt(question: str, context_blocks: list[str]) -> str:
    """根据问题和召回片段拼装回答任务提示文本。

    输入：
        question: 用户问题文本。
        context_blocks: 已格式化的来源片段列表。
    输出：
        适合作为后续模型输入的任务提示文本。
    异常：
        无。
    """

    context_text = "\n\n".join(context_blocks) if context_blocks else "当前没有可用来源片段。"
    return (
        f"用户问题：{question}\n\n"
        f"请仅基于以下来源片段整理回答：\n{context_text}\n\n"
        "输出时优先总结和问题最相关的信息，并保持对来源的尊重。"
    )


def build_empty_library_message() -> str:
    """返回论文库为空时的兜底提示。

    输入：
        无。
    输出：
        用于提示用户当前没有可检索论文的说明文本。
    异常：
        无。
    """

    return (
        "当前论文库中还没有可用内容，因此暂时无法完成检索与回答。"
        "请先将 PDF 放入 data/raw_papers/，并确认论文能够被正常解析。"
    )


def build_no_result_message(question: str) -> str:
    """返回检索不到相关结果时的兜底提示。

    输入：
        question: 用户问题文本。
    输出：
        用于提示用户调整问题或补充论文的说明文本。
    异常：
        无。
    """

    return (
        f"当前没有在本地论文片段中检索到与“{question}”足够相关的内容。"
        "你可以尝试换一种问法、减少问题长度，或补充更相关的论文后再试。"
    )


def build_answer_suffix() -> str:
    """返回回答结尾提示语。

    输入：
        无。
    输出：
        用于提醒用户结合来源核对的补充文本。
    异常：
        无。
    """

    return "以上内容是基于当前召回片段做出的最小整理，建议继续结合下方来源依据核对。"

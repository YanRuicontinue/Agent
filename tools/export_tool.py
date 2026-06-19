"""本模块作用：在整个智能体中负责将多步流程的中间结果导出为 Markdown 文件，支撑第三周的最小工作流闭环。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportArtifact:
    """本数据结构作用：保存一次导出任务的结果信息。"""

    output_path: str
    title: str
    content_length: int


def sanitize_file_name(name: str, fallback_name: str = "workflow_report") -> str:
    """将任意标题清洗为适合文件名使用的字符串。

    输入：
        name: 原始标题文本。
        fallback_name: 当标题为空时使用的默认文件名。
    输出：
        清洗后的文件名主体。
    异常：
        无。
    """

    cleaned_name = re.sub(r"[^\w\u4e00-\u9fff\- ]+", "_", name.strip())
    cleaned_name = re.sub(r"\s+", "_", cleaned_name)
    cleaned_name = cleaned_name.strip("_")
    return cleaned_name or fallback_name


def ensure_output_dir(output_dir: str | Path) -> Path:
    """确保导出目录存在。

    输入：
        output_dir: 导出目录路径。
    输出：
        已确保存在的目录对象。
    异常：
        当目录创建失败时，抛出 OSError。
    """

    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_workflow_markdown(
    topic: str,
    selected_papers: list[str],
    comparison_text: str,
    outline_text: str,
    step_logs: list[str],
) -> str:
    """将第三周工作流结果整理为 Markdown 文本。

    输入：
        topic: 工作流主题。
        selected_papers: 纳入论文列表。
        comparison_text: 多篇比较展示文本。
        outline_text: 综述提纲展示文本。
        step_logs: 工作流步骤日志。
    输出：
        Markdown 格式文本。
    异常：
        无。
    """

    lines = [
        f"# {topic}",
        "",
        "## 工作流摘要",
        "",
        f"- 主题：{topic}",
        f"- 纳入论文数量：{len(selected_papers)}",
        "",
        "## 纳入论文",
        "",
    ]

    for paper in selected_papers:
        lines.append(f"- {paper}")

    lines.extend(
        [
            "",
            "## 工作流步骤日志",
            "",
        ]
    )
    for step_log in step_logs:
        lines.append(f"- {step_log}")

    lines.extend(
        [
            "",
            "## 多篇论文比较",
            "",
            "```text",
            comparison_text,
            "```",
            "",
            "## 综述提纲",
            "",
            "```text",
            outline_text,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def export_markdown_report(
    output_dir: str | Path,
    title: str,
    markdown_text: str,
) -> ExportArtifact:
    """将 Markdown 内容写入输出目录。

    输入：
        output_dir: 输出目录。
        title: 导出标题。
        markdown_text: 待写入的 Markdown 内容。
    输出：
        导出结果对象。
    异常：
        当文件写入失败时，抛出 OSError。
    """

    target_dir = ensure_output_dir(output_dir)
    file_name = sanitize_file_name(title) + ".md"
    output_path = target_dir / file_name
    output_path.write_text(markdown_text, encoding="utf-8")
    return ExportArtifact(
        output_path=str(output_path),
        title=title,
        content_length=len(markdown_text),
    )


def run_export_demo() -> None:
    """执行 export_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印导出结果。
    异常：
        无。
    """

    markdown_text = build_workflow_markdown(
        topic="城市韧性研究综述",
        selected_papers=["paper_a.pdf", "paper_b.pdf"],
        comparison_text="这里是多篇比较结果。",
        outline_text="这里是综述提纲。",
        step_logs=[
            "步骤 1：已选中 2 篇论文。",
            "步骤 2：已完成多篇比较。",
            "步骤 3：已生成综述提纲。",
        ],
    )
    artifact = export_markdown_report("outputs", "城市韧性研究综述", markdown_text)
    print(f"导出成功：{artifact.output_path}")


if __name__ == "__main__":
    run_export_demo()

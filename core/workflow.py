"""本模块作用：在整个智能体中负责第三周的最小流程编排，提供 Planner/State 风格的数据结构与顺序执行能力。"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.export_tool import ExportArtifact, build_workflow_markdown, export_markdown_report


@dataclass
class WorkflowStep:
    """本数据结构作用：保存工作流中的单个步骤信息。"""

    name: str
    description: str
    status: str = "pending"


@dataclass
class WorkflowPlan:
    """本数据结构作用：保存工作流规划结果。"""

    topic: str
    targets: list[str] = field(default_factory=list)
    steps: list[WorkflowStep] = field(default_factory=list)


@dataclass
class WorkflowState:
    """本数据结构作用：保存工作流执行过程中的状态与产物。"""

    topic: str
    targets: list[str] = field(default_factory=list)
    selected_papers: list[str] = field(default_factory=list)
    step_logs: list[str] = field(default_factory=list)
    comparison_text: str = ""
    outline_text: str = ""
    export_artifact: ExportArtifact | None = None


@dataclass
class WorkflowRunResult:
    """本数据结构作用：保存一次完整工作流运行结果。"""

    status_message: str
    plan: WorkflowPlan
    state: WorkflowState
    formatted_output: str


def build_default_workflow_plan(topic: str, targets: list[str] | None = None) -> WorkflowPlan:
    """构建第三周最小工作流的默认计划。

    输入：
        topic: 工作流主题。
        targets: 目标论文列表。
    输出：
        工作流计划对象。
    异常：
        无。
    """

    return WorkflowPlan(
        topic=topic,
        targets=targets or [],
        steps=[
            WorkflowStep(name="select_papers", description="根据目标定位纳入论文"),
            WorkflowStep(name="compare_papers", description="执行多篇论文比较"),
            WorkflowStep(name="generate_outline", description="根据比较结果生成综述提纲"),
            WorkflowStep(name="export_markdown", description="将结果导出为 Markdown 文件"),
        ],
    )


def mark_step_status(plan: WorkflowPlan, step_name: str, status: str) -> None:
    """更新工作流计划中的步骤状态。

    输入：
        plan: 工作流计划对象。
        step_name: 步骤名称。
        status: 新状态。
    输出：
        无。
    异常：
        无。
    """

    for step in plan.steps:
        if step.name == step_name:
            step.status = status
            return


def format_workflow_plan(plan: WorkflowPlan) -> str:
    """将工作流计划整理为适合展示的文本。

    输入：
        plan: 工作流计划对象。
    输出：
        计划展示文本。
    异常：
        无。
    """

    lines = [
        "工作流计划：",
        f"主题：{plan.topic}",
    ]
    if plan.targets:
        lines.append(f"目标论文：{', '.join(plan.targets)}")
    else:
        lines.append("目标论文：未显式指定，将使用默认论文集合。")

    for index, step in enumerate(plan.steps, start=1):
        lines.append(f"{index}. {step.name} | {step.status} | {step.description}")
    return "\n".join(lines)


def format_workflow_run_result(result: WorkflowRunResult) -> str:
    """将工作流运行结果整理为适合命令行展示的文本。

    输入：
        result: 工作流运行结果对象。
    输出：
        可直接打印的展示文本。
    异常：
        无。
    """

    lines = [
        "多步工作流结果：",
        result.status_message,
        "",
        format_workflow_plan(result.plan),
        "",
        "工作流日志：",
    ]
    for log in result.state.step_logs:
        lines.append(f"- {log}")

    if result.state.export_artifact is not None:
        lines.extend(
            [
                "",
                f"导出文件：{result.state.export_artifact.output_path}",
                f"导出内容长度：{result.state.export_artifact.content_length}",
            ]
        )
    return "\n".join(lines)


def export_workflow_result(
    output_dir: str,
    topic: str,
    selected_papers: list[str],
    comparison_text: str,
    outline_text: str,
    step_logs: list[str],
) -> ExportArtifact:
    """将工作流结果导出为 Markdown 报告。

    输入：
        output_dir: 输出目录。
        topic: 工作流主题。
        selected_papers: 纳入论文列表。
        comparison_text: 比较结果文本。
        outline_text: 提纲结果文本。
        step_logs: 执行日志列表。
    输出：
        导出结果对象。
    异常：
        当写文件失败时，抛出 OSError。
    """

    markdown_text = build_workflow_markdown(
        topic=topic,
        selected_papers=selected_papers,
        comparison_text=comparison_text,
        outline_text=outline_text,
        step_logs=step_logs,
    )
    return export_markdown_report(output_dir=output_dir, title=topic, markdown_text=markdown_text)

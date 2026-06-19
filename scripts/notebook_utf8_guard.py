from __future__ import annotations

"""本模块作用：校验课程 Notebook 的 UTF-8 与中文完整性，并在校验通过后同步导出 Markdown 讲义。"""

import argparse
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
PREFIXES = ("00_", "01_", "02_", "03_", "04_")


def find_target_ipynb() -> list[Path]:
    """查找需要检查的课程 Notebook 文件。"""

    files: list[Path] = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        # 仅处理课程主文件，避免误扫临时 Notebook。
        if path.name.startswith(PREFIXES) and not path.name.isascii():
            files.append(path)
    return files


def has_cjk(text: str) -> bool:
    """判断文本中是否包含中日韩统一表意文字。"""

    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def validate_file(path: Path) -> tuple[bool, str]:
    """执行单文件编码与内容健康检查。"""

    text = path.read_text(encoding="utf-8", errors="replace")
    question_count = text.count("?")
    cjk_exists = has_cjk(text)

    if not cjk_exists:
        return False, f"{path.name}: no CJK characters found"
    if question_count > 30:
        # 问号异常偏多通常意味着编码损坏导致中文丢失。
        return False, f"{path.name}: suspicious question mark count = {question_count}"
    return True, f"{path.name}: ok"


def sync_md_from_ipynb(path: Path) -> Path:
    """将 Notebook 中的 Markdown 单元拼接导出为 .md 文件。"""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    sections: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", [])).rstrip()
        if text:
            sections.append(text)

    md_text = "\n\n---\n\n".join(sections) + "\n"
    md_path = path.with_suffix(".md")
    md_path.write_text(md_text, encoding="utf-8")
    return md_path


def main() -> int:
    """脚本主入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sync-md", action="store_true")
    args = parser.parse_args()

    targets = find_target_ipynb()
    if not targets:
        print("No target notebooks found")
        return 1

    failed = False
    for path in targets:
        ok, msg = validate_file(path)
        print(msg)
        if not ok:
            failed = True

    if args.sync_md and not failed:
        for path in targets:
            md_path = sync_md_from_ipynb(path)
            print(f"synced: {md_path.name}")

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

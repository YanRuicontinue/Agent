"""本模块作用：集中管理 CityScholar-Agent 的基础配置，为整个智能体提供统一目录、模型与运行参数。"""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_NAME = "CityScholar-Agent"
DATA_DIR = BASE_DIR / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()

# 按场景分配模型：问答优先性价比，结构化分析优先准确性。
DASHSCOPE_ANSWER_MODEL = os.getenv("DASHSCOPE_ANSWER_MODEL", "qwen-plus").strip()
DASHSCOPE_ANALYSIS_MODEL = os.getenv("DASHSCOPE_ANALYSIS_MODEL", "qwen-max").strip()
DASHSCOPE_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3").strip()


def _read_int_env(name: str, default_value: int) -> int:
    """读取整型环境变量，并在异常时回退默认值。"""

    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        return int(raw_value)
    except ValueError:
        return default_value


DASHSCOPE_TIMEOUT_SEC = _read_int_env("DASHSCOPE_TIMEOUT_SEC", 45)
DASHSCOPE_EMBEDDING_DIMENSIONS = _read_int_env("DASHSCOPE_EMBEDDING_DIMENSIONS", 128)


def get_app_config() -> dict[str, Path | str | int | bool]:
    """返回应用当前使用的基础配置。

    输入：
        无。
    输出：
        包含项目目录、模型参数和 API 开关的配置字典。
    异常：
        无。
    """

    return {
        "project_name": PROJECT_NAME,
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "raw_papers_dir": RAW_PAPERS_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "dashscope_api_key": DASHSCOPE_API_KEY,
        "dashscope_base_url": DASHSCOPE_BASE_URL,
        "dashscope_answer_model": DASHSCOPE_ANSWER_MODEL,
        "dashscope_analysis_model": DASHSCOPE_ANALYSIS_MODEL,
        "dashscope_embedding_model": DASHSCOPE_EMBEDDING_MODEL,
        "dashscope_embedding_dimensions": DASHSCOPE_EMBEDDING_DIMENSIONS,
        "dashscope_timeout_sec": DASHSCOPE_TIMEOUT_SEC,
        "llm_enabled": bool(DASHSCOPE_API_KEY),
    }

本模块作用：说明 CityScholar-Agent 的项目目标、当前最小功能、目录结构与运行方式。

# CityScholar-Agent

## 项目目标

CityScholar-Agent 是一个面向城市治理、城市规划与学术研究辅助场景的智能体项目。
当前阶段先搭建最小可运行骨架，为后续接入论文读取、文本解析、检索问答与基础分析能力做准备。

## 当前最小功能

- 提供清晰、可扩展的项目目录结构。
- 提供统一的基础配置入口。
- 提供可直接运行的主流程入口框架。
- 预留原始数据、处理结果和输出目录。

## 目录结构

```text
CityScholar-Agent/
├─ README.md
├─ requirements.txt
├─ config.py
├─ app.py
├─ data/
│  ├─ raw/
│  └─ processed/
├─ outputs/
└─ src/
   ├─ loaders/
   ├─ pipelines/
   └─ services/
```

## 运行步骤

1. 进入项目目录。
2. 可选：创建并激活虚拟环境。
3. 执行 `pip install -r requirements.txt`。
4. 执行 `python app.py`。
5. 程序会自动检查并创建运行所需目录，然后输出当前最小骨架的启动信息。

## 说明

- 当前版本只提供项目骨架，不包含真实的论文解析、检索和问答实现。
- 后续可以优先在 `src/loaders/`、`src/pipelines/`、`src/services/` 中逐步补充具体模块。

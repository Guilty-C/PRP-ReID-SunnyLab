# 基于 Prompt 提示工程的行人重识别 (ReID)

本项目探索如何利用大语言模型 (LLM) 的提示工程 (Prompt Engineering) 与多模态模型结合，提升行人重识别 (Person Re-Identification, ReID) 的检索性能。  
通过设计高质量的提示词，生成细粒度的行人描述与结构化属性，使文本特征与图像特征在共同空间对齐，从而提升跨模态检索的准确率与可解释性。

---

## 📂 项目结构

```text
reid-prompt/
├─ configs/        # 环境与流水线配置
│  ├─ env.yaml         # 环境配置
│  ├─ pipeline.yaml    # 流水线配置
│  └─ prompt_evaluation.yaml # 提示词评估配置
├─ data/           # 数据集目录（需手动下载）
│  └─ market1501/  # Market-1501 数据集
│     ├─ query/
│     ├─ bounding_box_test/
│     └─ bounding_box_train/
├─ docs/           # 文档与实验记录
│  ├─ BATCH_GENERATION_GUIDE.md  # 批量生成指南
│  ├─ batch_caption_guide.md     # 批量图像注释指南
│  └─ WINDOWS_SETUP_GUIDE.md     # Windows环境设置指南
├─ outputs/        # 模型输出（描述、属性、特征、结果）
│  ├─ captions_alt.csv  # 备选输出文件
│  ├─ features/         # 特征文件
│  └─ results/          # 检索结果
├─ prompts/        # 提示词模板
│  ├─ base.txt      # 基础提示词
│  ├─ compare.txt   # 比较提示词
│  ├─ json.txt      # JSON格式提示词
│  └─ salient.txt   # 显著性提示词
├─ scripts/        # 启动脚本
│  ├─ batch_generate_captions.cmd  # Windows批量生成脚本
│  ├─ batch_generate_captions.sh   # Linux/macOS批量生成脚本
│  ├─ quick_start.sh  # 快速开始脚本(Linux/macOS)
│  └─ quick_start_windows.cmd  # 快速开始脚本(Windows)
├─ src/            # 核心代码
│  ├─ encode_clip.py              # CLIP编码
│  ├─ eval_clip_results.py        # CLIP结果评估
│  ├─ eval_metrics.py             # 评估指标计算
│  ├─ eval_prompt_effectiveness.py # 提示词效果评估
│  ├─ gen_caption.py              # 单批生成描述
│  ├─ gen_caption_batch.py        # 批量生成描述
│  ├─ parse_attrs.py              # 属性解析
│  ├─ prepare_data.py             # 数据准备
│  ├─ rerank_optional.py          # 重排序（可选）
│  ├─ rerun_failed.py             # 失败任务重跑
│  └─ retrieve.py                 # 检索功能
├─ requirements.txt  # 项目依赖
└─ README.md         # 项目说明文档
```

---

## 📊 数据集说明

* **Market-1501**（主数据集）：包含 1,501 个行人身份，32,668 张检测框图像。
* 后续可扩展：DukeMTMC-reID、CUHK03、MSMT17、Occluded/Partial ReID 等，用于鲁棒性评测。

数据集放置方式（以 Market-1501 为例）：

```text
data/market1501/
├─ query/
├─ bounding_box_test/
└─ bounding_box_train/
```

---

## ⚙️ 环境依赖安装

推荐使用 **Python 3.10+** 与虚拟环境 (venv/conda)。

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# 升级 pip
pip install -U pip

# 安装依赖
pip install -r requirements.txt
```

当前项目依赖列表（`requirements.txt`）：

```txt
torch
torchvision
openai
requests
tqdm
numpy
pandas
transformers
# faiss-gpu 已移除，当前版本encode_clip.py使用随机向量作为占位符
```

---

## 🔑 API 密钥设置

本项目使用 VimsAI 平台上的 Qwen 模型进行图像描述生成，需要设置 API 密钥：

**Windows (cmd)**: 
```cmd
setx OPENAI_API_KEY "your_api_key_here"
```

**Windows (PowerShell)**: 
```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your_api_key_here", "User")
```

**Linux/macOS**: 
```bash
export OPENAI_API_KEY=your_api_key_here
# 永久设置（可选）
echo 'export OPENAI_API_KEY=your_api_key_here' >> ~/.bashrc
```

---

## 🚀 快速开始（小样冒烟测试）

### 通过脚本快速开始

**Windows**: 
```cmd
scripts\quick_start_windows.cmd
```

**Linux/macOS**: 
```bash
bash scripts/quick_start.sh
```

### 手动运行单个脚本

1. **准备数据**：下载并解压 Market-1501 数据集到 `data/market1501/`。
2. **准备提示词**：使用 `prompts/` 目录下的提示词模板或创建自定义提示词。
3. **生成描述**：

   ```bash
   # 单批生成（前20张图片）
   python src/gen_caption.py --index index_small.json --prompt_file prompts/base.txt --out outputs/captions/captions_20.csv
   ```
4. **查看输出**：生成的描述将保存在指定的输出文件中。

---

## 📋 核心功能

### 1. 图像描述生成

#### 单批生成（`gen_caption.py`）
- 处理前20张图片
- 支持JSON、JSONL、TXT格式的索引文件
- 输出CSV格式的描述结果

使用方式：
```bash
python src/gen_caption.py --index <index_file> --prompt_file <prompt_file> --out <output_file> [--verbose]
```

#### 批量生成（`gen_caption_batch.py`）
- 支持大规模图像处理
- 多线程并行处理
- 分批生成与保存
- 兼容多种索引文件格式
- Token计数功能

使用方式：
```bash
python src/gen_caption_batch.py <index_file> --prompt_file <prompt_file> --out <output_file> [--batch_size <size>] [--num_workers <num>] [--verbose]
```

### 2. 批量生成脚本

项目提供了便捷的批量生成脚本，支持Windows和Linux/macOS环境：

**Windows**: 
```cmd
scripts\batch_generate_captions.cmd
```

**Linux/macOS**: 
```bash
bash scripts/batch_generate_captions.sh
```

批量生成流程：
1. 调用 `prepare_data.py` 收集图像路径并创建索引文件
2. 检查API环境配置
3. 设置批处理参数（默认每批20张图片）
4. 调用 `gen_caption_batch.py` 执行批量生成
5. 结果保存在 `outputs/captions/batch` 目录

### 3. 数据准备

`prepare_data.py` 用于收集图像路径并创建索引文件，支持自定义数据集路径和收集数量。

### 4. 特征编码与检索

`encode_clip.py` 用于将图像和文本描述编码为特征向量，`retrieve.py` 实现基于这些特征的检索功能。

### 5. 评估功能

`eval_metrics.py` 计算检索性能指标（如Rank-1、mAP），`eval_prompt_effectiveness.py` 评估不同提示词的效果。

---

## 📈 评测指标

* **Rank-1**：Top-1 准确率
* **mAP**：平均准确率
* 可视化：CMC曲线、检索案例展示

---

## ✨ 项目目标

* 设计并验证多样化提示词模板
* 生成高质量行人描述与结构化属性
* 与多模态编码结合，实现文本→图像检索
* 在标准指标 (Rank-1, mAP) 上对比基线并分析改进效果
* 提供一套基于Prompt的可解释ReID实验框架

---

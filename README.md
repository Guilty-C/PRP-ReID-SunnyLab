# 基于 Prompt 提示工程的行人重识别 (ReID)

本项目探索如何利用大语言模型 (LLM) 的提示工程 (Prompt Engineering) 与多模态模型 (如 CLIP) 结合，提升行人重识别 (Person Re-Identification, ReID) 的检索性能。  
通过设计高质量的提示词，生成细粒度的行人描述与结构化属性，使文本特征与图像特征在共同空间对齐，从而提升跨模态检索的准确率与可解释性。

---

## 📂 项目结构

```text
reid-prompt/
├─ configs/        # 环境与流水线配置
├─ data/           # 数据集目录（需手动下载）
│  ├─ market1501/  # Market-1501 数据集
│  │  ├─ query/
│  │  ├─ bounding_box_test/
│  │  └─ bounding_box_train/
├─ docs/           # 文档与实验记录
├─ outputs/        # 模型输出（描述、属性、特征、结果）
├─ prompts/        # 提示词模板
├─ scripts/        # 启动脚本
├─ src/            # 核心代码
│  ├─ prepare_data.py
│  ├─ gen_caption.py
│  ├─ parse_attrs.py
│  ├─ encode_clip.py
│  ├─ retrieve.py
│  └─ eval_metrics.py
└─ requirements.txt
````

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

`requirements.txt` 建议包含：

```txt
torch
torchvision
faiss-gpu
openai
transformers
tqdm
numpy
pandas
requests
```

---

## 🚀 快速开始（小样冒烟测试）

1. **准备数据**：下载并解压 Market-1501 数据集到 `data/market1501/`。
2. **准备提示词**：

   * `prompts/base.txt`

     ```text
     请详细描述图像中的行人，只描述人物可见外观。
     包括性别、上衣、下装、鞋子、是否背包。若无法判断则写“未知”，不要猜测。
     不要描述背景。
     ```
   * `prompts/json.txt`

     ```text
     请用 JSON 格式输出该行人的属性，包括：
     gender, top_color, top_type, bottom_color, bottom_type, shoes_color, bag (true/false/未知)。
     如果无法确定，值填“未知”。
     ```
3. **运行快速脚本**：

   ```bash
   bash scripts/quick_start.sh
   ```
4. **查看输出**：

   * 描述结果：`outputs/captions/...`
   * 属性结果：`outputs/attrs/...`
   * 特征文件：`outputs/feats/...`
   * 评测指标与报告：`outputs/runs/...`

---

## 📈 评测指标

* **Rank-1**：Top-1 准确率
* **mAP**：平均准确率
* 可视化：CMC 曲线、检索案例展示

---

## ✨ 项目目标

* 设计并验证多样化提示词模板
* 生成高质量行人描述与结构化属性
* 与 CLIP 跨模态编码结合，实现文本→图像检索
* 在标准指标 (Rank-1, mAP) 上对比基线并分析改进效果
* 提供一套基于 Prompt 的可解释 ReID 实验框架

---

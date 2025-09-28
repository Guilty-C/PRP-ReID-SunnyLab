# 提示词效果评估实验指南

本文档详细介绍如何在行人重识别（ReID）项目中进行提示词效果评估实验，帮助您找到最优的提示词模板。

## 实验总览

实验流程主要分为以下几个阶段：
1. 数据准备：收集行人图像并生成索引
2. 描述生成：使用不同提示词为图像生成描述
3. 特征编码：使用CLIP模型编码文本描述
4. 检索匹配：基于编码特征进行检索
5. 性能评估：计算标准ReID指标和提示词效果指标

## 环境配置

首先，确保您的环境已正确配置：

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **设置API密钥**
   在环境变量中设置OpenAI兼容API的密钥（本文使用阿里云DashScope）：
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="您的API密钥"
   ```

3. **准备数据目录**
   ```bash
   mkdir -p ./data/market1501
   mkdir -p ./outputs/captions
   mkdir -p ./outputs/attrs
   mkdir -p ./outputs/feats
   mkdir -p ./outputs/runs
   mkdir -p ./experiments
   ```

## 实验步骤

### 1. 数据准备

首先，需要准备行人图像数据集并生成索引文件：

```bash
# 创建示例数据集索引（仅索引前10张图像用于测试）
python src/prepare_data.py --data_root ./data/market1501 --out_index ./outputs/runs/index_small.json
```

如果您有自己的数据集，可以将图像放在`./data/market1501`目录下，保持以下结构：
```
data/market1501/
├── bounding_box_test/  # 测试图像
└── query/              # 查询图像
```

### 2. 提示词模板准备

在`./prompts`目录下创建多个不同的提示词模板文件，用于比较效果。例如：

```bash
# 查看当前可用的提示词模板
ls ./prompts

# 创建新的提示词模板
notepad ./prompts/detailed.txt
notepad ./prompts/concise.txt
```

示例提示词内容（详细版本）：
```
请详细描述图像中的人物，重点关注以下方面：
1. 服装：上衣类型、颜色、图案；下装类型、颜色、图案
2. 配饰：帽子、围巾、手套、眼镜等
3. 携带物品：包、手机等
4. 显著特征：身材、发型等
描述应中立客观，不涉及身份识别信息。
```

### 3. 批量生成描述

使用不同的提示词模板为图像生成描述文本，并保存为CSV格式：

```bash
# 使用基础提示词生成描述
python src/gen_caption_batch.py --index ./outputs/runs/index_small.json --prompt_file ./prompts/base.txt --out ./outputs/captions/base

# 使用JSON格式提示词生成描述
python src/gen_caption_batch.py --index ./outputs/runs/index_small.json --prompt_file ./prompts/json.txt --out ./outputs/captions/json

# 使用自定义提示词生成描述
python src/gen_caption_batch.py --index ./outputs/runs/index_small.json --prompt_file ./prompts/my_prompt.txt --out ./outputs/captions/custom
```

### 4. 合并描述文件

为了进行提示词效果对比，需要将不同提示词生成的描述合并到一个CSV文件中：

```bash
# 创建合并后的描述文件
python -c "
import pandas as pd
import glob
import os

# 读取所有CSV文件
dfs = []
for prompt_name in ['base', 'json', 'custom']:
    csv_path = f'./outputs/captions/{prompt_name}/captions_all.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['prompt'] = prompt_name  # 添加提示词标识
        dfs.append(df)

# 合并并保存
merged_df = pd.concat(dfs)
merged_df.to_csv('./outputs/captions/comparison.csv', index=False)
print(f'已合并{len(merged_df)}条描述记录')
"
```

### 5. 运行提示词效果评估

使用提示词效果评估工具来比较不同提示词的性能：

```bash
# 使用默认配置运行评估
python src/eval_prompt_effectiveness.py --descriptions_file ./outputs/captions/comparison.csv --images_dir ./data/market1501

# 使用自定义配置运行评估
python src/eval_prompt_effectiveness.py --config configs/my_experiment_config.yaml --descriptions_file ./outputs/captions/comparison.csv --images_dir ./data/market1501 --output_file ./outputs/eval_results.json

# 调整权重参数运行评估
python src/eval_prompt_effectiveness.py --descriptions_file ./outputs/captions/comparison.csv --images_dir ./data/market1501 --w_a 0.6 --w_d 0.4
```

### 6. 分析评估结果

评估完成后，可以通过以下方式分析结果：

```bash
# 直接查看JSON结果文件
python -c "import json; import pprint; with open('./outputs/eval_results.json', 'r', encoding='utf-8') as f: results = json.load(f); pprint.pprint(results['summary'])"

# 使用测试脚本中的分析功能
python -c "from scripts.test_prompt_evaluation import analyze_results; analyze_results('./outputs/eval_results.json')"
```

### 7. 基于最优提示词进行ReID实验

根据评估结果选择最优的提示词，然后进行完整的行人重识别实验：

```bash
# 1. 使用最优提示词生成描述
python src/gen_caption_batch.py --index ./outputs/runs/index_small.json --prompt_file ./prompts/best_prompt.txt --out ./outputs/captions/best

# 2. 解析属性
python src/parse_attrs.py --captions ./outputs/captions/best/captions_all.csv --out_dir ./outputs/attrs/best

# 3. 编码CLIP特征
python src/encode_clip.py --captions ./outputs/captions/best/captions_all.csv --out_feats ./outputs/feats/text_best.npy

# 4. 检索匹配
python src/retrieve.py --feats ./outputs/feats/text_best.npy --topk 5

# 5. 评估检索性能
python src/eval_metrics.py --results ./outputs/retrieve_results.json
```

## 实验设计建议

### 基础实验设计

#### 1. 提示词详细程度对比实验

**目的**：比较不同详细程度的提示词对生成描述质量的影响

**实验设置**：
- 准备3-5个提示词模板，从非常简洁到非常详细
- 保持其他参数（如模型、图像集）一致
- 使用默认权重设置（w_a=0.6, w_d=0.4）

**评估指标**：
- 综合得分（final_score）
- 平均token消耗
- 准确性分数（accuracy_score）
- 详细性分数（detail_score）

**示例提示词**：
- 简洁版："描述图像中的人穿着。"
- 标准版："请描述图像中的人物，包括服装、颜色和携带物品。"
- 详细版："详细描述人物的服装（上衣、下装、鞋子）、颜色、图案、配饰（帽子、眼镜等）和携带物品。"

#### 2. 权重参数敏感性实验

**目的**：探索不同权重设置对提示词评估结果的影响

**实验设置**：
- 固定使用3-5个不同的提示词模板
- 尝试多种权重组合：(0.9,0.1), (0.7,0.3), (0.5,0.5), (0.3,0.7), (0.1,0.9)
- 对每种权重组合重复评估3次取平均

**分析方法**：
- 绘制权重变化对提示词排名的影响曲线
- 确定在不同应用场景下的最优权重设置
- 例如：资源受限场景可能更注重低token消耗（可降低详细性权重）

### 进阶实验设计

#### 1. 属性类别优化实验

**目的**：优化属性类别定义以提高评估准确性

**实验设置**：
- 创建多个配置文件，每个文件定义不同的属性类别和关键词
- 例如：一组专注于服装细节，一组包含更多配饰和特征描述

**配置示例**：
```yaml
# 配置1：基础属性
attributes:
  gender: ["男", "女", "man", "woman"]
  top: ["T恤", "衬衫", "夹克"]
  bottom: ["裤", "裙", "裤子", "裙子"]
  shoes: ["鞋", "鞋子"]
  bag: ["包", "背包"]

# 配置2：扩展属性
attributes:
  gender: ["男", "女", "man", "woman"]
  top: ["T恤", "衬衫", "夹克", "毛衣", "卫衣"]
  bottom: ["裤", "裙", "裤子", "裙子", "短裤"]
  shoes: ["鞋", "鞋子", "运动鞋", "靴子", "高跟鞋"]
  bag: ["包", "背包", "手提包", "公文包"]
  accessories: ["眼镜", "帽子", "围巾", "手套"]
  colors: ["红", "蓝", "绿", "黄", "黑", "白", "灰"]
```

#### 2. 模型性能对比实验

**目的**：比较不同CLIP模型和LLM模型对评估结果的影响

**实验设置**：
- 尝试不同的CLIP模型：`openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, `openai/clip-vit-base-patch16`
- 尝试不同的LLM模型生成描述：Qwen, GPT-4o, Claude等

**性能指标**：
- 模型推理速度
- 内存占用
- 评估结果的准确性和一致性

#### 3. 跨数据集泛化实验

**目的**：测试提示词在不同数据集上的泛化能力

**实验设置**：
- 准备2-3个不同的行人数据集（如Market-1501、CUHK03、DukeMTMC-reID）
- 使用同一组提示词在所有数据集上生成描述并进行评估
- 分析提示词在不同数据集上的表现稳定性

#### 4. 长期稳定性分析

**目的**：评估提示词效果的长期稳定性

**实验设置**：
- 在不同时间点（如每天一次，持续一周）使用相同的提示词和图像生成描述
- 分析不同时间点生成的描述质量变化
- 特别关注LLM API可能存在的更新或变化对结果的影响

## 实验技巧与注意事项

### 实用技巧

1. **资源优化**
   - 对于大规模实验，使用`--batch_size`参数控制批处理大小，避免内存溢出
   - 在GPU资源有限的情况下，可使用`--device cpu`切换到CPU模式
   - 对于多次重复实验，可使用`--cache_dir`参数缓存模型以节省时间

2. **数据管理**
   - 使用清晰的命名约定为实验结果文件命名，例如`eval_promptA_clipB_weights06-04.json`
   - 为每个实验创建单独的目录，便于结果跟踪和管理
   - 记录实验日志，包括参数设置、环境配置和关键发现

3. **结果可视化**
   - 使用pandas和matplotlib创建实验结果可视化图表
   - 绘制不同提示词的性能雷达图、得分对比柱状图等
   - 对于权重敏感性实验，可以创建热图展示不同权重组合下的结果变化

### 注意事项

1. **API调用限制**
   - 注意LLM API的调用频率限制和配额
   - 对于大规模实验，考虑使用异步调用或分批处理策略
   - 实现错误重试机制，处理可能的API调用失败

2. **数据质量控制**
   - 确保图像数据质量良好，避免模糊或低分辨率图像影响评估结果
   - 检查生成的描述文本，排除明显错误或不相关的描述
   - 对于异常值，可考虑在分析时进行适当处理

3. **结果解释**
   - 综合考虑多个指标，而不仅仅依赖单一得分
   - 理解评估结果的局限性，结合实际应用场景进行解释
   - 必要时进行人工验证，确保自动评估结果的可靠性

4. **环境一致性**
   - 保持实验环境的一致性，避免不同实验间的环境差异影响结果
   - 记录使用的软件版本、模型版本和硬件配置
   - 考虑使用Docker容器或conda环境确保实验可重复性

通过系统设计和执行实验，您将能够深入了解不同提示词对行人重识别性能的影响，并找到最适合您应用场景的最优提示词。

## 实验示例

下面提供一个完整的实验案例，演示如何比较不同提示词模板在行人重识别任务中的表现。

### 示例实验：提示词详细程度对比

#### 实验目的
比较不同详细程度的提示词模板对生成描述质量和行人重识别性能的影响，找到平衡准确性和token消耗的最优提示词。

#### 实验准备

1. **创建实验目录**
   ```bash
   mkdir -p experiments/prompt_comparison
   mkdir -p experiments/prompt_comparison/configs
   mkdir -p experiments/prompt_comparison/results
   ```

2. **准备实验数据**
   ```bash
   # 下载Market-1501数据集的小型示例版本（如果有）
   # 或者使用自己的数据集，放入./data/market1501目录
   
   # 创建数据集索引
   python src/prepare_data.py --data_root ./data/market1501 --out_index ./experiments/prompt_comparison/index.json
   ```

3. **准备提示词模板**
   创建3个不同详细程度的提示词模板：
   
   ```bash
   # 简洁提示词
   notepad ./experiments/prompt_comparison/configs/prompt_simple.txt
   # 内容："描述图像中的人穿着。"
   
   # 标准提示词
   notepad ./experiments/prompt_comparison/configs/prompt_standard.txt
   # 内容："请描述图像中的人物，包括服装类型、颜色和显著特征。描述应简洁明了，不超过3句话。"
   
   # 详细提示词
   notepad ./experiments/prompt_comparison/configs/prompt_detailed.txt
   # 内容："详细描述图像中人物的外观特征，包括：
   # 1. 服装：上衣类型、颜色、图案；下装类型、颜色、图案
   # 2. 配饰：帽子、围巾、手套、眼镜、手表等
   # 3. 携带物品：包、手机等
   # 4. 显著特征：发型、身材等
   # 描述应客观准确，不涉及身份识别信息。"
   ```

4. **创建评估配置文件**
   ```bash
   notepad ./experiments/prompt_comparison/configs/eval_config.yaml
   ```
   
   配置内容：
   ```yaml
   weights:
     accuracy: 0.6
     detail: 0.4
   
   attributes:
     gender: ["男", "女", "man", "woman"]
     top: ["T恤", "衬衫", "夹克", "毛衣", "卫衣", "上衣"]
     bottom: ["裤", "裙", "裤子", "裙子", "短裤"]
     shoes: ["鞋", "鞋子", "运动鞋", "靴子"]
     bag: ["包", "背包", "手提包"]
     accessories: ["眼镜", "帽子", "围巾", "手套", "手表"]
     colors: ["红", "蓝", "绿", "黄", "黑", "白", "灰", "紫", "粉"]
   
   output:
     sort_by: "final_score"
     sort_order: "descending"
   
   clip:
     model_name: "openai/clip-vit-base-patch32"
     device: "auto"
     use_half_precision: true
   
   tokenizer:
     model_name: "gpt2"
   
   performance:
     batch_size: 32
     cache_enabled: true
     cache_dir: "./cache"
   ```

#### 实验执行

1. **生成描述文本**
   ```bash
   # 设置API密钥
   $env:OPENAI_API_KEY="您的API密钥"
   
   # 使用简洁提示词生成描述
   python src/gen_caption_batch.py --index ./experiments/prompt_comparison/index.json --prompt_file ./experiments/prompt_comparison/configs/prompt_simple.txt --out ./experiments/prompt_comparison/results/simple
   
   # 使用标准提示词生成描述
   python src/gen_caption_batch.py --index ./experiments/prompt_comparison/index.json --prompt_file ./experiments/prompt_comparison/configs/prompt_standard.txt --out ./experiments/prompt_comparison/results/standard
   
   # 使用详细提示词生成描述
   python src/gen_caption_batch.py --index ./experiments/prompt_comparison/index.json --prompt_file ./experiments/prompt_comparison/configs/prompt_detailed.txt --out ./experiments/prompt_comparison/results/detailed
   ```

2. **合并描述文件**
   ```bash
   python -c "
   import pandas as pd
   import os
   
   # 读取所有提示词生成的描述
   prompt_types = ['simple', 'standard', 'detailed']
   dfs = []
   
   for prompt_type in prompt_types:
       csv_path = f'./experiments/prompt_comparison/results/{prompt_type}/captions_all.csv'
       if os.path.exists(csv_path):
           df = pd.read_csv(csv_path)
           # 提取并添加提示词内容
           with open(f'./experiments/prompt_comparison/configs/prompt_{prompt_type}.txt', 'r', encoding='utf-8') as f:
               prompt_content = f.read().strip()
           df['prompt'] = prompt_content
           df['prompt_type'] = prompt_type  # 添加类型标签
           dfs.append(df)
   
   # 合并并保存
   if dfs:
       merged_df = pd.concat(dfs)
       merged_df.to_csv('./experiments/prompt_comparison/results/comparison.csv', index=False)
       print(f'已合并{len(merged_df)}条描述记录')
   else:
       print('未找到任何描述文件')
   "
   ```

3. **运行提示词效果评估**
   ```bash
   python src/eval_prompt_effectiveness.py \
     --config ./experiments/prompt_comparison/configs/eval_config.yaml \
     --descriptions_file ./experiments/prompt_comparison/results/comparison.csv \
     --images_dir ./data/market1501 \
     --output_file ./experiments/prompt_comparison/results/eval_results.json
   ```

4. **使用最优提示词进行完整ReID实验**
   ```bash
   # 假设标准提示词表现最佳
   # 1. 重新生成描述（如果需要）
   python src/gen_caption_batch.py --index ./experiments/prompt_comparison/index.json --prompt_file ./experiments/prompt_comparison/configs/prompt_standard.txt --out ./experiments/prompt_comparison/results/best
   
   # 2. 解析属性
   python src/parse_attrs.py --captions ./experiments/prompt_comparison/results/best/captions_all.csv --out_dir ./experiments/prompt_comparison/results/attrs
   
   # 3. 编码CLIP特征（当前版本使用随机向量，后续可替换为真实实现）
   python src/encode_clip.py --captions ./experiments/prompt_comparison/results/best/captions_all.csv --out_feats ./experiments/prompt_comparison/results/text_feats.npy
   
   # 4. 检索匹配（注意：当前版本的retrieve.py使用captions参数而不是feats参数）
   python src/retrieve.py --captions ./experiments/prompt_comparison/results/best/captions_all.csv --topk 10 --out ./experiments/prompt_comparison/results/retrieve_results.json
   
   # 5. 评估检索性能
   python src/eval_metrics.py --results ./experiments/prompt_comparison/results/retrieve_results.json
   ```

#### 结果分析

1. **查看提示词评估结果**
   ```bash
   # 查看评估总结
   python -c "import json; with open('./experiments/prompt_comparison/results/eval_results.json', 'r', encoding='utf-8') as f: results = json.load(f); print('\n提示词评估结果总结：\n'); for prompt in results['prompt_evaluations'][:3]: print(f'提示词类型: {prompt.get('prompt_type', '未知')}\n综合得分: {prompt['average_scores']['final_score']:.4f}\n准确性得分: {prompt['average_scores']['accuracy_score']:.4f}\n详细性得分: {prompt['average_scores']['detail_score']:.4f}\n平均Token消耗: {prompt['average_scores']['token_count']:.2f}\n---')"
   ```

2. **创建可视化分析脚本**
   ```bash
   notepad ./experiments/prompt_comparison/analyze_results.py
   ```
   
   脚本内容：
   ```python
   import json
   import matplotlib.pyplot as plt
   import pandas as pd
   
   # 加载评估结果
   with open('./experiments/prompt_comparison/results/eval_results.json', 'r', encoding='utf-8') as f:
       results = json.load(f)
   
   # 提取提示词评估数据
   prompt_data = []
   for prompt in results['prompt_evaluations']:
       data = {
           'type': prompt.get('prompt_type', 'unknown'),
           'final_score': prompt['average_scores']['final_score'],
           'accuracy': prompt['average_scores']['accuracy_score'],
           'detail': prompt['average_scores']['detail_score'],
           'tokens': prompt['average_scores']['token_count']
       }
       prompt_data.append(data)
   
   df = pd.DataFrame(prompt_data)
   
   # 绘制对比图表
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # 综合得分对比
   df.plot(kind='bar', x='type', y='final_score', ax=axes[0, 0], legend=False)
   axes[0, 0].set_title('综合得分对比')
   axes[0, 0].set_ylabel('得分')
   
   # 准确性和详细性对比
   df.plot(kind='bar', x='type', y=['accuracy', 'detail'], ax=axes[0, 1])
   axes[0, 1].set_title('准确性和详细性对比')
   axes[0, 1].set_ylabel('得分')
   
   # Token消耗对比
   df.plot(kind='bar', x='type', y='tokens', ax=axes[1, 0], legend=False)
   axes[1, 0].set_title('平均Token消耗')
   axes[1, 0].set_ylabel('Token数量')
   
   # 雷达图
   from math import pi
   categories = ['final_score', 'accuracy', 'detail', 'tokens']
   N = len(categories)
   
   # 归一化数据用于雷达图
   normalized_df = df.copy()
   for col in categories:
       if col != 'type':
           min_val = normalized_df[col].min()
           max_val = normalized_df[col].max()
           if max_val - min_val > 0:
               normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
   
   angles = [n / float(N) * 2 * pi for n in range(N)]
   angles += angles[:1]  # 闭合雷达图
   
   axes[1, 1].set_theta_offset(pi / 2)
   axes[1, 1].set_theta_direction(-1)
   
   for i, row in normalized_df.iterrows():
       values = row[categories].tolist()
       values += values[:1]
       axes[1, 1].plot(angles, values, linewidth=2, linestyle='solid', label=row['type'])
       axes[1, 1].fill(angles, values, alpha=0.1)
   
   axes[1, 1].set_title('提示词性能雷达图')
   axes[1, 1].set_thetagrids([n / float(N) * 360 for n in range(N)], categories)
   axes[1, 1].legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
   
   plt.tight_layout()
   plt.savefig('./experiments/prompt_comparison/results/evaluation_plots.png', dpi=300)
   print('可视化图表已保存到 evaluation_plots.png')
   
   # 显示表格数据
   print('\n提示词评估结果表：')
   print(df.sort_values('final_score', ascending=False).to_string(index=False))
   ```
   
   运行分析脚本：
   ```bash
   python ./experiments/prompt_comparison/analyze_results.py
   ```

3. **生成实验报告**
   根据分析结果，按照以下结构撰写实验报告：
   - 实验目的和背景
   - 实验设置和方法
   - 实验结果与分析
   - 结论和建议
   - 未来改进方向

## 实验自动化脚本

为了简化实验流程，可以创建一个自动化脚本：

```bash
notepad ./scripts/run_prompt_experiment.ps1
```

脚本内容：
```powershell
# 设置实验参数
$EXPERIMENT_NAME = "prompt_comparison_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$DATA_ROOT = "./data/market1501"
$EXPERIMENT_DIR = "./experiments/$EXPERIMENT_NAME"
$CONFIG_FILE = "./configs/prompt_evaluation.yaml"

# 创建目录
New-Item -ItemType Directory -Path $EXPERIMENT_DIR -Force
New-Item -ItemType Directory -Path "$EXPERIMENT_DIR/results" -Force

# 准备数据
Write-Host "[1/6] 准备数据集索引..."
python src/prepare_data.py --data_root $DATA_ROOT --out_index "$EXPERIMENT_DIR/index.json"

# 准备提示词（假设使用现有的提示词模板）
Write-Host "[2/6] 复制提示词模板..."
Copy-Item -Path "./prompts/*.txt" -Destination "$EXPERIMENT_DIR/" -Force

# 生成描述
Write-Host "[3/6] 生成描述文本..."
$PROMPT_FILES = Get-ChildItem -Path "$EXPERIMENT_DIR/*.txt" -Name
foreach ($PROMPT_FILE in $PROMPT_FILES) {
    $PROMPT_NAME = [System.IO.Path]::GetFileNameWithoutExtension($PROMPT_FILE)
    Write-Host "  处理提示词: $PROMPT_NAME"
    python src/gen_caption_batch.py --index "$EXPERIMENT_DIR/index.json" --prompt_file "$EXPERIMENT_DIR/$PROMPT_FILE" --out "$EXPERIMENT_DIR/results/$PROMPT_NAME"
}

# 合并描述文件
Write-Host "[4/6] 合并描述文件..."
python -c "
import pandas as pd
import os
experiment_dir = '$EXPERIMENT_DIR'.replace('\', '/')
prompt_files = os.listdir(experiment_dir)
prompt_files = [f for f in prompt_files if f.endswith('.txt')]
dfs = []
for prompt_file in prompt_files:
    prompt_name = os.path.splitext(prompt_file)[0]
    csv_path = f'{experiment_dir}/results/{prompt_name}/captions_all.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        with open(f'{experiment_dir}/{prompt_file}', 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        df['prompt'] = prompt_content
        df['prompt_type'] = prompt_name
        dfs.append(df)
if dfs:
    merged_df = pd.concat(dfs)
    merged_df.to_csv(f'{experiment_dir}/results/comparison.csv', index=False)
    print(f'已合并{len(merged_df)}条描述记录')
"

# 运行评估
Write-Host "[5/6] 运行提示词评估..."
python src/eval_prompt_effectiveness.py --config $CONFIG_FILE --descriptions_file "$EXPERIMENT_DIR/results/comparison.csv" --images_dir $DATA_ROOT --output_file "$EXPERIMENT_DIR/results/eval_results.json"

# 分析结果
Write-Host "[6/6] 分析评估结果..."
python -c "
import json
with open('$EXPERIMENT_DIR/results/eval_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)
print('\n提示词评估结果总结：\n')
for i, prompt in enumerate(results['prompt_evaluations'][:3], 1):
    print(f'第{i}名提示词类型: {prompt.get('prompt_type', '未知')}\n综合得分: {prompt['average_scores']['final_score']:.4f}\n准确性得分: {prompt['average_scores']['accuracy_score']:.4f}\n详细性得分: {prompt['average_scores']['detail_score']:.4f}\n平均Token消耗: {prompt['average_scores']['token_count']:.2f}\n---')
print(f'\n完整结果已保存至: $EXPERIMENT_DIR/results/eval_results.json')
"

Write-Host "\n提示词评估完成！所有结果已保存至: $EXPERIMENT_DIR\n" 

# 基于最优提示词执行完整ReID实验
Write-Host "[7/7] 基于最优提示词执行完整ReID实验..."

# 找到最佳提示词
$BEST_PROMPT = python -c "
import json
with open('$EXPERIMENT_DIR/results/eval_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)
if results['prompt_evaluations']:
    best = results['prompt_evaluations'][0]
    print(best.get('prompt_type', 'standard'))
else:
    print('standard')
"

# 重新使用最佳提示词生成描述（如需）
Write-Host "  使用最佳提示词类型: $BEST_PROMPT"
if (Test-Path -Path "./prompts/prompt_$BEST_PROMPT.txt") {
    Write-Host "  找到预设的最佳提示词文件: ./prompts/prompt_$BEST_PROMPT.txt"
    $BEST_PROMPT_FILE = "./prompts/prompt_$BEST_PROMPT.txt"
} else {
    # 从评估结果中获取最佳提示词内容
    $BEST_PROMPT_CONTENT = python -c "
import json
with open('$EXPERIMENT_DIR/results/eval_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)
if results['prompt_evaluations']:
    best = results['prompt_evaluations'][0]
    print(best['prompt'])
"
    
    # 保存到临时文件
    $BEST_PROMPT_FILE = "$EXPERIMENT_DIR/prompt_best.txt"
    Set-Content -Path $BEST_PROMPT_FILE -Value $BEST_PROMPT_CONTENT -Encoding UTF8
    Write-Host "  已保存最佳提示词到: $BEST_PROMPT_FILE"
}

# 1. 使用最佳提示词生成描述
python src/gen_caption_batch.py --index "$EXPERIMENT_DIR/index.json" --prompt_file $BEST_PROMPT_FILE --out "$EXPERIMENT_DIR/results/best"

# 2. 解析属性（可选）
python src/parse_attrs.py --captions "$EXPERIMENT_DIR/results/best/captions_all.csv" --out_dir "$EXPERIMENT_DIR/results/attrs"

# 3. 编码CLIP特征（当前版本使用随机向量）
python src/encode_clip.py --captions "$EXPERIMENT_DIR/results/best/captions_all.csv" --out_feats "$EXPERIMENT_DIR/results/text_feats.npy"

# 4. 检索匹配
python src/retrieve.py --captions "$EXPERIMENT_DIR/results/best/captions_all.csv" --topk 10 --out "$EXPERIMENT_DIR/results/retrieve_results.json"

# 5. 评估检索性能
python src/eval_metrics.py --results "$EXPERIMENT_DIR/results/retrieve_results.json"

Write-Host "\n完整实验完成！所有结果已保存至: $EXPERIMENT_DIR"
```

运行自动化脚本：
```powershell
./scripts/run_prompt_experiment.ps1
```

## 总结

通过本指南，您应该能够：
1. 理解行人重识别项目中提示词效果评估的基本原理和流程
2. 设计和执行有效的提示词对比实验
3. 分析实验结果并选择最优提示词
4. 将最优提示词应用于完整的行人重识别流程

通过系统地优化提示词，您可以显著提高行人重识别系统的性能，同时平衡模型的计算成本和描述质量。

祝您实验顺利！
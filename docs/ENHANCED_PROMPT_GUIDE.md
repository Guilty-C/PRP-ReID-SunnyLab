# 增强版JSON格式Prompt使用指南

## 为什么需要增强版Prompt？

在Person ReID任务中，我们发现原始的简单描述性prompt存在以下问题：

1. **描述过于笼统**：只关注性别、基本服装颜色等，导致很多不同个体生成相似描述
2. **区分度不足**：Market1501数据集中有很多穿着相似的人，简单描述无法有效区分
3. **评估结果差**：由于caption区分度不足，CLIP无法正确匹配，导致Rank-1和mAP为0

## 增强版JSON格式Prompt的优势

新创建的`enhanced_json.txt`prompt具有以下优势：

1. **结构化输出**：以JSON格式确保每个属性都能正确提取
2. **更丰富的特征**：包含更多区分性特征，如发型、体型、姿势等
3. **更高的区分度**：增加了ReID任务中不同个体的可区分性
4. **便于后续处理**：结构化格式便于后续的解析和特征提取

## 使用方法

### 1. 基本用法

你可以直接在`run_pipeline.py`中使用`--prompt_file`参数指定增强版prompt：

```bash
# 使用增强版JSON格式prompt运行pipeline
python src/run_pipeline.py --prompt_file prompts/enhanced_json.txt --num 50
```

### 2. 实验对比

建议进行对比实验，比较不同prompt的效果：

```bash
# 使用原始prompt运行（对照组）
python src/run_pipeline.py --prompt_file prompts/base.txt --num 50

# 使用增强版JSON格式prompt运行（实验组）
python src/run_pipeline.py --prompt_file prompts/enhanced_json.txt --num 50
```

### 3. 测试单个prompt

可以使用提供的测试脚本快速验证prompt效果：

```bash
python scripts/test_enhanced_prompt.py
```

## 增强版Prompt的结构

增强版prompt要求输出包含以下字段的JSON对象：

```json
{
  "gender": "male",             # 性别
  "top": "light blue short-sleeved shirt",  # 上衣类型和颜色
  "bottom": "black pants",      # 下装类型和颜色
  "shoes": "white sneakers",    # 鞋子类型和颜色
  "accessories": ["backpack"],  # 配饰列表
  "hairstyle": "short hair",    # 发型描述
  "body_shape": "slim",         # 体型描述
  "pose": "hands in pockets",   # 姿势描述
  "distinctive_features": []     # 其他独特特征
}
```

## 预期效果

使用增强版JSON格式prompt后，我们期望看到：

1. 生成的caption包含更丰富的区分性特征
2. 不同个体的描述更加多样化
3. CLIP能够更好地区分不同个体
4. Rank-1准确率和mAP指标有所提升

## 注意事项

1. 确保环境变量`OPENAI_API_KEY`已正确设置
2. 对于大规模实验，可适当调整`--num_workers`和`--batch_size`参数
3. 所有结果会自动保存在`outputs/expXXX`目录下，可以方便地进行对比分析
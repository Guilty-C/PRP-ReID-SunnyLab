# 批量图像注释功能使用指南

## 概述

批量图像注释功能允许您一次性为大量图像生成详细描述，通过批处理方式提高效率。这个功能特别适合处理大型图像数据集，如Market-1501行人重识别数据集。

## 功能特点

- 支持自定义每批处理的图像数量
- 自动创建输出目录结构，方便管理生成的注释
- 同时生成每个批次的CSV文件和总的CSV文件，便于后续处理
- 包含API密钥检查和错误处理机制
- 支持UTF-8编码，确保中文等非ASCII字符正常显示

## 快速开始

### 使用批处理脚本（推荐）

1. 确保已设置`OPENAI_API_KEY`环境变量：
   ```cmd
   set OPENAI_API_KEY=your_api_key_here
   ```

2. 运行批量注释脚本：
   ```cmd
   cd d:\PRP SunnyLab\reid-prompt
   scripts\batch_generate_captions.cmd
   ```

3. 根据提示输入每批处理的图片数量（默认值：20）

4. 等待处理完成，生成的注释文件将保存在`outputs\captions\batch`目录下

### 直接调用Python脚本

如果需要更灵活的配置，可以直接调用`gen_caption_batch.py`脚本：

```cmd
cd d:\PRP SunnyLab\reid-prompt
python src\gen_caption_batch.py --index .\outputs\runs\index_batch.json --out outputs\captions\batch --prompt_file .\prompts\base.txt --batch_size 20
```

## 命令行参数说明

`gen_caption_batch.py`支持以下命令行参数：

- `--index`：图像ID索引文件路径（必需）
- `--out`：输出目录路径（必需）
- `--prompt_file`：提示词文件路径（必需）
- `--batch_size`：每批处理的图像数量（可选，默认值：20）

## 输出文件说明

执行成功后，将在指定的输出目录中生成以下文件：

1. **captions_all.csv**：包含所有处理过的图像的注释信息
2. **captions_part_*.csv**：按照批次生成的注释文件，每个文件包含一批图像的注释信息

每个CSV文件包含以下字段：
- `image_id`：图像的唯一标识符
- `prompt`：用于生成注释的提示词
- `caption`：模型生成的图像描述
- `timestamp`：生成时间戳

## 工作流程详解

1. **数据准备**：调用`prepare_data.py`生成图像ID索引文件
2. **环境检查**：确认Python环境和OpenAI API密钥已正确配置
3. **参数设置**：用户输入每批处理的图像数量
4. **批量处理**：按照指定的批次大小，批量调用模型生成图像描述
5. **结果保存**：将生成的注释保存到CSV文件中

## 权限错误处理

如果在保存CSV文件时遇到权限错误，脚本会尝试：

1. 首先尝试写入主路径 `outputs\captions\batch\captions_all.csv`
2. 如果失败，则尝试写入备选路径 `outputs\captions_all_alt.csv`
3. 如果备选路径写入也失败，则会生成前10条示例数据并保存到备选路径

## 常见问题排查

1. **API密钥错误**：
   - 确保已正确设置`OPENAI_API_KEY`环境变量
   - 可以通过`scripts\setup_api_and_run.cmd`来设置API密钥

2. **权限错误**：
   - 确保当前用户对输出目录有写权限
   - 以管理员身份运行命令提示符

3. **Python环境问题**：
   - 确保已安装所有必要的依赖：`pip install -r requirements.txt`
   - 确保Python版本符合要求（建议Python 3.8+）

## 性能优化建议

- 根据您的系统性能和API限制调整`batch_size`参数
- 对于非常大的数据集，可以考虑分多次处理
- 在网络条件不佳的情况下，适当减小批次大小以避免超时错误

## 示例用途

批量生成的图像描述可以用于：
- 为行人重识别模型提供文本模态的训练数据
- 构建多模态检索系统
- 生成数据集的元数据描述
- 为图像分析和可视化提供辅助信息

## 注意事项

- 生成大量图像描述可能会产生较高的API调用费用，请关注您的OpenAI账户使用情况
- 处理大量图像可能需要较长时间，请耐心等待
- 如遇处理中断，可以重新运行脚本，已生成的注释不会被覆盖
# 批量图像描述生成功能指南

本指南详细介绍如何使用Reid-Prompt项目中的批量图像描述生成功能，包括在不同环境下的运行方法、配置参数和常见问题解决方案。

## 功能概述

批量图像描述生成功能可以为多个图像自动生成详细的文本描述，适用于大规模图像处理场景。该功能支持两种脚本格式：

- `batch_generate_captions.cmd` - 适用于Windows命令提示符
- `batch_generate_captions.sh` - 适用于Git Bash、Linux或macOS

## 在Windows环境下运行

### 方法一：使用Git Bash

1. 确保已安装[Git for Windows](https://git-scm.com/download/win)
2. 打开Git Bash终端
3. 导航到项目根目录
4. 运行以下命令：

```bash
chmod +x scripts/batch_generate_captions.sh
./scripts/batch_generate_captions.sh
```

### 方法二：通过主设置脚本

1. 运行项目主设置脚本：

```cmd
scripts/setup_api_and_run.cmd
```

2. 在菜单中选择选项 `[5] 批量生成图片描述（批处理模式）`

## 在Linux/macOS环境下运行

1. 打开终端
2. 导航到项目根目录
3. 运行以下命令：

```bash
chmod +x scripts/batch_generate_captions.sh
./scripts/batch_generate_captions.sh
```

## 工作流程详解

批量生成图像描述的完整流程包括以下四个主要步骤：

### 1. 数据准备

脚本首先调用`prepare_data.py`收集图像路径并创建索引文件。默认情况下，它会从`data/market1501`目录下的`bounding_box_test`和`query`子目录中各收集前10个图像，总共生成20个图像的索引。

### 2. API环境检查

脚本会检查`OPENAI_API_KEY`环境变量是否已设置。如果未设置，将提示用户如何设置API密钥。

### 3. 批处理参数设置

系统会询问用户每批处理的图片数量，默认为20张。

### 4. 批量生成

脚本调用`gen_caption_batch.py`执行批量生成，将图像描述保存到CSV文件中。

## 输出文件说明

生成的结果将保存在`outputs/captions/batch`目录下：

- `captions_all.csv` - 包含所有批次生成的描述
- `captions_part1.csv`, `captions_part2.csv`等 - 按批次保存的描述

## 常见问题解决方案

### 1. 脚本无法执行（权限问题）

在Git Bash或Linux/macOS环境下，确保脚本具有执行权限：

```bash
chmod +x scripts/batch_generate_captions.sh
```

### 2. 导入错误（如`os模块未找到`）

确保您使用的Python环境包含所有必要的依赖。可以使用以下命令安装项目依赖：

```bash
pip install -r requirements.txt
```

### 3. API密钥问题

如果遇到API密钥相关错误，请确保已正确设置`OPENAI_API_KEY`环境变量：

- 在Git Bash中：
  ```bash
export OPENAI_API_KEY=your_api_key_here
  ```

- 在Windows命令提示符中：
  ```cmd
set OPENAI_API_KEY=your_api_key_here
  ```

### 4. 生成数量不足

如果脚本没有生成所有20个图像描述，可能是由于以下原因：

- API调用失败或超时
- 数据路径中实际可用的图像数量不足
- 脚本执行过程中出现错误

请检查脚本输出的错误信息，根据提示进行修复。

### 5. 文件写入权限错误

如果遇到文件写入权限错误，脚本会尝试使用备选路径`outputs/captions_all_alt.csv`保存结果。如果仍有问题，请确保您对输出目录有写权限。

## 自定义配置

您可以通过修改以下参数来自定义批量生成过程：

1. 在`batch_generate_captions.sh`脚本中修改数据根目录：
   ```bash
   DATA_ROOT="$(pwd)/data/market1501"
   ```

2. 修改索引文件路径：
   ```bash
   INDEX_FILE="$(pwd)/outputs/runs/index_batch.json"
   ```

3. 修改输出目录：
   ```bash
   OUTPUT_DIR="$(pwd)/outputs/captions/batch"
   ```

4. 修改使用的提示文件：
   ```bash
   PROMPT_FILE="$(pwd)/prompts/base.txt"
   ```

## 高级用法

如果需要处理大量图像，可以考虑调整批处理大小参数以优化性能和稳定性。较小的批处理大小可以减少API调用超时的风险，但会增加总体处理时间。

## 注意事项

1. 确保您的API密钥有足够的配额来处理所有图像
2. 批量处理大量图像可能需要较长时间，请耐心等待
3. 在网络不稳定的环境下，可能会出现API调用失败的情况
4. 生成的描述质量取决于使用的提示模板和模型能力

## 联系支持

如果您在使用过程中遇到任何问题，请检查项目文档或联系技术支持。
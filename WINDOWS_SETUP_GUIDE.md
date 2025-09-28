# Windows环境下使用真实API生成图片描述指南

本指南将帮助您在Windows环境下正确设置API并使用真实数据生成图片描述，而不是使用测试数据。

## 目录
- [前提条件](#前提条件)
- [快速开始：使用API设置助手](#快速开始使用api设置助手)
- [手动设置API密钥](#手动设置api密钥)
- [API连接测试](#api连接测试)
- [运行项目](#运行项目)
- [常见问题排查](#常见问题排查)

## 前提条件

在开始之前，请确保您已安装以下软件：

1. Python 3.7或更高版本
2. Git Bash（Windows版Git）
3. 有效的API密钥（用于调用qwen-plus模型）

## 快速开始：使用API设置助手

我们提供了一个便捷的批处理工具，可以帮助您设置API密钥、测试连接并运行项目。

1. 双击运行 `scripts\setup_api_and_run.cmd` 文件
2. 如果您尚未设置`OPENAI_API_KEY`环境变量，系统会提示您输入API密钥
3. 从菜单中选择您想要执行的操作：
   - 测试API连接
   - 生成图片描述（运行完整流程）
   - 生成图片描述并跳过JSONL转换
   - 仅生成CSV文件
   - 退出

这个工具会自动帮您处理环境变量设置和运行命令，非常适合初学者使用。

## 手动设置API密钥

如果您希望手动设置API密钥，可以按照以下方法操作：

### 临时设置（仅当前会话有效）

#### 在Git Bash中：
```bash
export OPENAI_API_KEY='your-api-key-here'
```

#### 在PowerShell中：
```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

### 永久设置

1. 右键点击"此电脑" -> "属性" -> "高级系统设置" -> "环境变量"
2. 在"用户变量"或"系统变量"中点击"新建"
3. 变量名：`OPENAI_API_KEY`
4. 变量值：您的API密钥
5. 点击"确定"保存设置
6. 重启所有打开的命令提示符或终端窗口，使设置生效

## API连接测试

在使用真实API之前，建议先测试连接是否正常工作。我们提供了专门的测试脚本：

1. 确保已设置`OPENAI_API_KEY`环境变量
2. 打开命令提示符或PowerShell
3. 运行以下命令：
   ```bash
   python scripts/test_api_connection.py
   ```

这个脚本会：
- 检查API密钥是否已设置
- 尝试连接到API服务器
- 发送一个简单的测试请求
- 显示响应结果或错误信息

## 运行项目

### 完整流程（生成描述+JSONL转换+后续处理）

在Git Bash中运行：
```bash
scripts/quick_start.sh
```

### 跳过JSONL转换

如果您只想生成图片描述（CSV文件），而不需要进行JSONL转换和后续处理，可以使用以下命令：

在Git Bash中运行：
```bash
SKIP_JSONL=true scripts/quick_start.sh
```

### 仅生成CSV文件

如果您只需要生成图片描述的CSV文件，可以运行简化版脚本：

在命令提示符或PowerShell中运行：
```cmd
scripts\simple_generate_csv.cmd
```

## 常见问题排查

### 权限错误

如果遇到类似`PermissionError: [Errno 13] Permission denied`的错误：

1. 尝试以管理员身份运行Git Bash或命令提示符
2. 检查输出目录（`outputs\captions`）的文件权限设置
3. 确保没有其他程序正在锁定相关文件

### API连接失败

如果API连接失败，请检查：

1. API密钥是否正确设置
2. 网络连接是否正常
3. 是否能够访问 `https://dashscope.aliyuncs.com`
4. 防火墙设置是否阻止了连接

### JSONL转换错误

如果JSONL转换步骤出现问题：

1. 检查CSV文件是否存在且格式正确
2. 使用`SKIP_JSONL=true`选项跳过JSONL转换
3. 查看错误信息以确定具体问题

## 输出文件位置

- 图片描述CSV文件：`outputs\captions\captions_20.csv`
- JSONL文件（如果生成）：`outputs\captions\captions.jsonl`
- 检索结果：`outputs\results\retrieval_results.json`

## 联系我们

如果您在使用过程中遇到任何问题，请参考项目文档或联系项目维护人员获取帮助。
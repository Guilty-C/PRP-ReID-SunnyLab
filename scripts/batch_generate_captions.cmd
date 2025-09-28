@echo off
chcp 65001 >nul

:: 设置UTF-8编码
echo [INFO] 设置UTF-8编码

:: 简化版本不使用彩色输出，确保在所有Windows环境下兼容

:: 检查Python环境
python --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 未找到Python环境，请确保Python已正确安装并添加到系统PATH中
    pause
    exit /b 1
)

:: 创建必要的输出目录
echo.
echo [INFO] 创建必要的输出目录...
if not exist outputs\captions mkdir outputs\captions
if not exist outputs\runs mkdir outputs\runs

echo [INFO] 已创建必要的输出目录

:: 步骤1：准备数据
echo.
echo [1/4] 准备数据...
python src\prepare_data.py --data_root .\data\market1501 --out_index .\outputs\runs\index_batch.json
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 数据准备失败，请检查错误信息
    pause
    exit /b 1
)
echo [1/4] 数据准备完成

:: 步骤2：检查API密钥
echo.
echo [2/4] 检查API环境...
if "%OPENAI_API_KEY%" == "" (
    echo [WARNING] OPENAI_API_KEY 环境变量未设置!
    echo [INFO] 请先设置API密钥，使用以下命令之一:
    echo [INFO] 1. 临时设置: set OPENAI_API_KEY=your_api_key_here
    echo [INFO] 2. 使用API设置助手: scripts\setup_api_and_run.cmd
    pause
    exit /b 1
) else (
    echo [INFO] OPENAI_API_KEY 环境变量已设置
)
echo [2/4] API环境检查完成

:: 步骤3：询问批处理参数
echo.
echo [3/4] 设置批处理参数...
set /p "BATCH_SIZE=请输入每批处理的图片数量 [默认: 20]: "
if "%BATCH_SIZE%" == "" set "BATCH_SIZE=20"

echo [INFO] 每批处理 %BATCH_SIZE% 张图片
set "OUTPUT_DIR=outputs\captions\batch"
mkdir %OUTPUT_DIR% >nul 2>nul
echo [INFO] 输出目录: %OUTPUT_DIR%
echo [3/4] 批处理参数设置完成

:: 步骤4：执行批量生成
echo.
echo [4/4] 开始批量生成图像描述...
echo [INFO] 调用 gen_caption_batch.py 进行批量处理

title 批量生成图像描述 - 正在运行...
python src\gen_caption_batch.py --index .\outputs\runs\index_batch.json --out %OUTPUT_DIR% --prompt_file .\prompts\base.txt --batch_size %BATCH_SIZE%

:: 检查执行结果
if %ERRORLEVEL% equ 0 (
    echo.
echo [INFO] 批量生成图像描述成功完成!
    echo [INFO] 生成的文件:
    echo [INFO] - 总的CSV文件: %OUTPUT_DIR%\captions_all.csv
    echo [INFO] - 批次CSV文件: %OUTPUT_DIR%\captions_part*.csv
) else (
    echo.
echo [ERROR] 批量生成图像描述过程中发生错误
    echo [INFO] 请检查上面的错误信息
)

echo.
echo [INFO] 批量注释功能执行结束
pause
exit /b %ERRORLEVEL%
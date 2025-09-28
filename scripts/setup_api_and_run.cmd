@echo off
REM 设置UTF-8编码
chcp 65001 > nul

REM 颜色定义
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

cls
echo %CYAN%========================================================================%RESET%
echo %CYAN%                         Reid-Prompt API设置助手%RESET%
echo %CYAN%========================================================================%RESET%
echo.

REM 检查是否已安装Python依赖
pip show colorama > nul
if %errorlevel% neq 0 (
    echo %YELLOW%[!] 缺少必要的依赖包colorama，正在安装...%RESET%
    pip install colorama
    if %errorlevel% neq 0 (
        echo %RED%[✗] 依赖包安装失败，请手动运行: pip install colorama%RESET%
        pause
        exit /b 1
    )
)

REM 检查是否已设置OPENAI_API_KEY
if not defined OPENAI_API_KEY (
    echo %RED%[✗] OPENAI_API_KEY 环境变量未设置%RESET%
    echo.
    set /p "api_key=请输入您的API密钥: "
    
    REM 临时设置环境变量（仅在当前会话有效）
    set "OPENAI_API_KEY=%api_key%"
    echo %GREEN%[✓] OPENAI_API_KEY 环境变量已临时设置%RESET%
    echo %YELLOW%[!] 注意：此设置仅在当前命令窗口有效，关闭窗口后需要重新设置%RESET%
    echo %YELLOW%[!] 如需永久设置，请通过系统环境变量设置%RESET%
    echo.
) else (
    echo %GREEN%[✓] OPENAI_API_KEY 环境变量已设置%RESET%
    echo.
)

:menu
cls
echo %CYAN%========================================================================%RESET%
echo %CYAN%                         Reid-Prompt API设置助手%RESET%
echo %CYAN%========================================================================%RESET%
echo.
echo %BLUE%[1] 测试API连接%RESET%
echo %BLUE%[2] 生成图片描述（运行quick_start.sh）%RESET%
echo %BLUE%[3] 生成图片描述并跳过JSONL转换%RESET%
echo %BLUE%[4] 仅生成CSV文件%RESET%
echo %BLUE%[5] 批量生成图片描述（批处理模式）%RESET%
echo %BLUE%[6] 退出%RESET%
echo.

set /p "choice=请选择操作 [1-6]: "

if "%choice%"=="1" (
    echo.
    echo %BLUE%[i] 正在运行API连接测试...%RESET%
    echo.
    python scripts/test_api_connection.py
    echo.
    pause
    goto menu
) else if "%choice%"=="2" (
    echo.
    echo %BLUE%[i] 正在运行quick_start.sh脚本，这将生成图片描述和JSONL文件...%RESET%
    echo.
    REM 启动Git Bash并运行quick_start.sh
    start "Git Bash" "%ProgramFiles%\Git\git-bash.exe" -c "scripts/quick_start.sh; read -p '按Enter键退出...'"
    goto menu
) else if "%choice%"=="3" (
    echo.
    echo %BLUE%[i] 正在运行quick_start.sh脚本，跳过JSONL转换...%RESET%
    echo.
    REM 启动Git Bash并运行带SKIP_JSONL参数的quick_start.sh
    start "Git Bash" "%ProgramFiles%\Git\git-bash.exe" -c "SKIP_JSONL=true scripts/quick_start.sh; read -p '按Enter键退出...'"
    goto menu
) else if "%choice%"=="4" (
    echo.
    echo %BLUE%[i] 正在运行simple_generate_csv.cmd脚本，仅生成CSV文件...%RESET%
    echo.
    call scripts\simple_generate_csv.cmd
    echo.
    pause
    goto menu
) else if "%choice%"=="5" (
    echo.
    echo %BLUE%[i] 正在运行batch_generate_captions.cmd脚本，批量生成图片描述...%RESET%
    echo %YELLOW%[!] 此功能可以一次处理大量图像，支持自定义每批处理数量%RESET%
    echo.
    call scripts\batch_generate_captions.cmd
    echo.
    pause
    goto menu
) else if "%choice%"=="6" (
    echo %CYAN%[i] 感谢使用Reid-Prompt API设置助手，再见！%RESET%
    pause
    exit /b 0
) else (
    echo %RED%[✗] 无效的选择，请重新输入！%RESET%
    pause
    goto menu
)
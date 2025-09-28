@echo off
chcp 65001 >nul

:: 在Windows上运行bash脚本的辅助工具

cls
echo ===============================================================
echo          Reid-Prompt - Bash脚本运行助手

echo ===============================================================
echo.
echo [说明] 这个工具可以帮助你在Windows环境下运行项目中的bash脚本。
echo [前提] 你的电脑上必须已经安装了Git（包含Git Bash）。
echo.

:: 检查是否已安装Git
errorlevel 0
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 未找到Git！请先安装Git，下载地址：https://git-scm.com/download/win
echo.
    echo [提示] 安装时请确保勾选了"Git Bash Here"选项。
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%a in ('git --version') do set GIT_VERSION=%%a
echo [INFO] Git已安装: %GIT_VERSION%
echo.
)

:: 询问要运行的bash脚本
set /p "BASH_SCRIPT=请输入要运行的bash脚本路径（例如：scripts/quick_start.sh）: "

:: 检查脚本是否存在
if not exist "%BASH_SCRIPT%" (
    echo [ERROR] 找不到脚本文件: %BASH_SCRIPT%
pause
    exit /b 1
)

:: 询问是否有额外参数
set /p "SCRIPT_ARGS=请输入脚本参数（可选，例如：SKIP_JSONL=true）: "

echo.
echo [INFO] 正在启动Git Bash并运行脚本...
echo [INFO] 脚本: %BASH_SCRIPT%
if not "%SCRIPT_ARGS%" == "" (
echo [INFO] 参数: %SCRIPT_ARGS%
)
echo.

echo [提示] 一个新的Git Bash窗口将打开并运行脚本。
echo [提示] 完成后，请在Git Bash窗口中按Enter键退出。
echo.
pause

:: 启动Git Bash并运行脚本
if "%SCRIPT_ARGS%" == "" (
    start "Git Bash - Reid-Prompt" "%ProgramFiles%\Git\git-bash.exe" -c "%BASH_SCRIPT%; read -p '按Enter键退出...'"
) else (
    start "Git Bash - Reid-Prompt" "%ProgramFiles%\Git\git-bash.exe" -c "%SCRIPT_ARGS% %BASH_SCRIPT%; read -p '按Enter键退出...'"
)

echo [INFO] Git Bash窗口已启动。
pause
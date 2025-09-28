#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from openai import OpenAI
from colorama import init, Fore, Style

# 初始化colorama以支持Windows环境下的彩色输出
init(autoreset=True)

def print_success(message):
    """打印成功信息"""
    print(f"{Fore.GREEN}[✓] {message}{Style.RESET_ALL}")

def print_error(message):
    """打印错误信息"""
    print(f"{Fore.RED}[✗] {message}{Style.RESET_ALL}")

def print_warning(message):
    """打印警告信息"""
    print(f"{Fore.YELLOW}[!] {message}{Style.RESET_ALL}")

def print_info(message):
    """打印普通信息"""
    print(f"{Fore.BLUE}[i] {message}{Style.RESET_ALL}")

def check_environment_variables():
    """检查必要的环境变量是否设置"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print_error("OPENAI_API_KEY 环境变量未设置！")
        print_warning("在Windows环境下设置环境变量的方法：")
        print_warning("1. 临时设置（仅当前会话有效）：")
        print_warning("   - 在Git Bash中：export OPENAI_API_KEY='your-api-key'")
        print_warning("   - 在PowerShell中：$env:OPENAI_API_KEY='your-api-key'")
        print_warning("2. 永久设置：")
        print_warning("   - 右键点击'此电脑' -> 属性 -> 高级系统设置 -> 环境变量")
        print_warning("   - 在'用户变量'或'系统变量'中点击'新建'")
        print_warning("   - 变量名：OPENAI_API_KEY，变量值：your-api-key")
        return False
    
    print_success(f"OPENAI_API_KEY 环境变量已设置（前6位：{api_key[:6]}...）")
    return True

def test_api_connection():
    """测试API连接是否正常"""
    print_info("开始测试API连接...")
    
    try:
        # 初始化OpenAI客户端
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        print_info(f"使用的base_url: https://dashscope.aliyuncs.com/compatible-mode/v1")
        print_info(f"测试模型: qwen-plus")
        
        # 发送测试请求
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个测试助手，只需简单回答问题"},
                {"role": "user", "content": "你好，能简单介绍一下自己吗？"}
            ],
            timeout=30
        )
        
        # 检查响应
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message.content
            print_success(f"API调用成功！")
            print_info(f"响应内容: {message}")
            return True
        else:
            print_error("API调用返回格式异常")
            print_error(f"完整响应: {response}")
            return False
            
    except Exception as e:
        print_error(f"API调用失败: {str(e)}")
        print_warning("可能的解决方案：")
        print_warning("1. 检查API密钥是否正确")
        print_warning("2. 确保您的网络能够访问 https://dashscope.aliyuncs.com")
        print_warning("3. 检查防火墙设置，确保没有阻止连接")
        print_warning("4. 尝试重启您的终端或命令提示符")
        return False

def main():
    """主函数"""
    print("="*60)
    print(f"{Fore.CYAN}Reid-Prompt API连接测试工具{Style.RESET_ALL}")
    print("="*60)
    
    # 检查环境变量
    if not check_environment_variables():
        print_warning("请先设置OPENAI_API_KEY环境变量，然后重新运行此脚本。")
        sys.exit(1)
    
    # 测试API连接
    if test_api_connection():
        print_success("\nAPI连接测试通过！您可以使用真实的API生成图片描述了。")
        print_info("运行生成图片描述的命令：")
        print_info("   - 在Git Bash中: scripts/quick_start.sh")
        print_info("   - 跳过JSONL转换: SKIP_JSONL=true scripts/quick_start.sh")
    else:
        print_error("\nAPI连接测试失败，请根据上面的错误信息进行排查。")
    
    print("="*60)

if __name__ == "__main__":
    main()
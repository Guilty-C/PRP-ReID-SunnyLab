#!/usr/bin/env python
# test_enhanced_prompt.py - 测试增强版JSON格式prompt

import os
import sys
import json
from openai import OpenAI

# 确保脚本可以在任何目录下运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_prompt(image_id="test_image"):
    """测试增强版JSON格式prompt"""
    # 读取增强版prompt
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "enhanced_json.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
        print(f"[INFO] 已加载增强版JSON格式prompt")
    except Exception as e:
        print(f"[ERROR] 无法读取prompt文件: {e}")
        return

    # 创建用户提示
    user_prompt = f"{prompt_text}\n图像ID: {image_id}"
    
    # 初始化OpenAI客户端
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://usa.vimsai.com/v1"
        )
        print(f"[INFO] OpenAI客户端初始化成功")
    except Exception as e:
        print(f"[ERROR] 初始化OpenAI客户端失败: {e}")
        return

    # 调用模型
    try:
        print(f"[INFO] 正在调用模型生成结构化描述...")
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个善于详细描述人物外观特征的助手，能准确提取多种区分性特征"},
                {"role": "user", "content": user_prompt}
            ],
            timeout=30
        )
        
        # 解析JSON响应
        caption = resp.choices[0].message.content
        try:
            # 尝试解析JSON
            structured_data = json.loads(caption)
            print(f"[SUCCESS] 生成结构化描述成功！")
            print("\n结构化输出:\n" + json.dumps(structured_data, ensure_ascii=False, indent=2))
            return structured_data
        except json.JSONDecodeError:
            print(f"[WARNING] 响应不是有效的JSON格式:")
            print(caption)
            return caption
    except Exception as e:
        print(f"[ERROR] 调用模型失败: {e}")
        return None

if __name__ == "__main__":
    # 运行测试
    test_enhanced_prompt()
    
    print("\n=== 增强版JSON格式prompt的优势 ===")
    print("1. 提供更详细的区分性特征（发型、体型、姿势等）")
    print("2. 结构化JSON格式确保每个属性都能正确提取")
    print("3. 增加ReID任务中不同个体的区分度")
    print("4. 便于后续解析和特征提取")
    print("\n建议在run_pipeline.py中使用此prompt进行对比实验！")
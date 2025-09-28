import os
import sys
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置中文显示
try:
    # 尝试加载中文字体
    font = ImageFont.truetype("simhei.ttf", 20)
except:
    # 如果没有找到中文字体，使用默认字体
    font = ImageFont.load_default()
    print("[WARNING] 无法加载中文字体，可能会显示乱码")

def create_sample_images(output_dir, num_images=5):
    """
    创建一些样本行人图像用于测试
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    # 定义一些行人描述特征
    person_features = [
        {"gender": "男", "top": "红色T恤", "bottom": "蓝色牛仔裤", "shoes": "白色运动鞋", "bag": "黑色背包"},
        {"gender": "女", "top": "白色衬衫", "bottom": "黑色裙子", "shoes": "黑色高跟鞋", "bag": "棕色手提包"},
        {"gender": "男", "top": "灰色夹克", "bottom": "卡其色长裤", "shoes": "棕色皮鞋", "bag": "无"},
        {"gender": "女", "top": "粉色毛衣", "bottom": "牛仔裤", "shoes": "白色运动鞋", "bag": "粉色双肩包"},
        {"gender": "男", "top": "黑色短袖", "bottom": "灰色短裤", "shoes": "蓝色拖鞋", "bag": "无"}
    ]
    
    for i in range(num_images):
        # 创建一个简单的图像
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制一个简化的人形
        draw.rectangle([50, 30, 170, 200], fill='lightgray')  # 身体
        draw.ellipse([70, 20, 150, 70], fill='lightblue')  # 头部
        
        # 获取当前行人特征
        features = person_features[i % len(person_features)]
        
        # 在图像上添加描述文本
        text = f"{features['gender']}\n{features['top']}\n{features['bottom']}\n{features['shoes']}\n{features['bag']}"
        text_lines = text.split('\n')
        
        y_position = 205
        for line in text_lines:
            draw.text((10, y_position), line, font=font, fill='black')
            y_position += 20
        
        # 保存图像
        image_id = f"person_{i+1}"
        image_path = os.path.join(output_dir, f"{image_id}.jpg")
        img.save(image_path)
        image_paths.append((image_id, image_path, features))
        
        print(f"[INFO] 创建样本图像: {image_path}")
    
    return image_paths

def generate_sample_descriptions(image_paths, output_file, prompts):
    """
    生成样本描述文件（CSV格式）
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 打开CSV文件并写入表头
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write("image_id,prompt,caption,timestamp\n")
        
        # 为每个图像生成多个描述
        for image_id, _, features in image_paths:
            for i, prompt in enumerate(prompts):
                # 根据提示词和图像特征生成不同详细程度的描述
                if i == 0:  # 详细描述
                    caption = f"{features['gender']}，上身穿{features['top']}，下身穿{features['bottom']}，脚穿{features['shoes']}，"
                    caption += f"{'背着' if '无' not in features['bag'] else '没有背'}{features['bag']}。"
                    caption += "整体着装休闲舒适。"
                elif i == 1:  # 中等详细描述
                    caption = f"{features['gender']}，{features['top']}，{features['bottom']}。"
                else:  # 简单描述
                    caption = f"一个{features['gender']}性。"
                
                # 写入CSV行，注意处理包含逗号的文本
                f.write(f"{image_id},\"{prompt}\",\"{caption}\",2023-01-01T00:00:00\n")
                
    print(f"[INFO] 生成样本描述文件: {output_file}")

def run_evaluation(descriptions_file, images_dir, output_file):
    """
    运行提示词评估算法
    """
    from src.eval_prompt_effectiveness import main as run_eval
    
    # 创建临时配置文件
    temp_config = os.path.join(os.path.dirname(output_file), "temp_config.yaml")
    
    # 创建配置内容
    config_content = {
        "clip": {
            "model_name": "openai/clip-vit-base-patch32",
            "device": "auto",
            "use_half_precision": True
        },
        "tokenizer": {
            "model_name": "gpt2",
            "fallback_method": "space_split"
        },
        "weights": {
            "accuracy": 0.6,
            "detail": 0.4
        },
        "data_paths": {
            "descriptions_file": descriptions_file,
            "images_dir": images_dir,
            "output_file": output_file
        },
        "output": {
            "save_individual_results": True,
            "pretty_print": True,
            "sort_by": "final_score",
            "sort_order": "descending"
        }
    }
    
    # 写入临时配置文件
    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f, allow_unicode=True)
    
    # 构建命令行参数
    sys.argv = ["eval_prompt_effectiveness.py", "--config", temp_config]
    
    # 运行评估
    try:
        import sys
        from io import StringIO
        
        # 重定向标准输出以便捕获
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        # 运行评估函数
        run_eval()
        
        # 恢复标准输出
        sys.stdout = old_stdout
        
        # 删除临时配置文件
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        print(f"[INFO] 评估完成，结果已保存至: {output_file}")
        return True
    except Exception as e:
        print(f"[ERROR] 评估过程中出错: {e}")
        # 清理临时文件
        if os.path.exists(temp_config):
            os.remove(temp_config)
        return False

def analyze_results(results_file):
    """
    分析评估结果
    """
    if not os.path.exists(results_file):
        print(f"[ERROR] 结果文件不存在: {results_file}")
        return
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print("\n[INFO] 评估结果分析：")
    print(f"共评估了 {len(results)} 个提示词")
    
    if results:
        best_prompt = results[0]
        print(f"\n最佳提示词：")
        print(f"提示词内容: {best_prompt['prompt'][:100]}{'...' if len(best_prompt['prompt']) > 100 else ''}")
        print(f"平均综合得分: {best_prompt['average_scores']['final_score']:.6f}")
        print(f"平均准确性: {best_prompt['average_scores']['accuracy']:.4f}")
        print(f"平均详细性: {best_prompt['average_scores']['detail']:.4f}")
        print(f"平均token数: {best_prompt['average_scores']['tokens']:.2f}")
        
        # 输出所有提示词的排名
        print("\n所有提示词排名：")
        for i, result in enumerate(results):
            print(f"排名 {i+1}: 得分={result['average_scores']['final_score']:.6f}, token数={result['average_scores']['tokens']:.2f}")
            print(f"  提示词: {result['prompt'][:80]}{'...' if len(result['prompt']) > 80 else ''}")
    

def main():
    # 定义测试配置
    test_dir = "./outputs/test_eval"
    images_dir = os.path.join(test_dir, "images")
    descriptions_file = os.path.join(test_dir, "descriptions.csv")
    results_file = os.path.join(test_dir, "prompt_evaluation_results.json")
    
    # 定义测试用的提示词
    test_prompts = [
        "请详细描述图像中的行人，包括性别、上衣、下装、鞋子、是否背包等外观特征。描述应尽可能详细准确。",
        "简要描述图像中的行人，包括主要着装特征。",
        "描述图像中的人物。"
    ]
    
    print("[INFO] 开始测试提示词评估算法...")
    
    # 1. 创建样本图像
    print("\n[步骤1] 创建样本图像...")
    image_paths = create_sample_images(images_dir, num_images=5)
    
    # 2. 生成样本描述
    print("\n[步骤2] 生成样本描述...")
    generate_sample_descriptions(image_paths, descriptions_file, test_prompts)
    
    # 3. 运行评估
    print("\n[步骤3] 运行提示词评估...")
    # 注意：由于我们是模拟环境，这里不实际调用评估函数，而是创建模拟结果
    # 如果要真实运行，请取消下面的注释
    
    # success = run_evaluation(descriptions_file, images_dir, results_file)
    # 
    # if success:
    #     # 4. 分析结果
    #     print("\n[步骤4] 分析评估结果...")
    #     analyze_results(results_file)
    
    # 创建模拟结果以便演示
    create_mock_results(results_file, test_prompts)
    analyze_results(results_file)
    
    print("\n[INFO] 测试完成！")
    print(f"测试数据保存在: {test_dir}")
    print("要真实运行评估，请修改脚本中的相关部分，取消对run_evaluation的注释。")
    

def create_mock_results(results_file, prompts):
    """
    创建模拟评估结果，用于演示
    """
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # 模拟结果数据
    mock_results = []
    
    # 为每个提示词创建模拟结果
    for i, prompt in enumerate(prompts):
        # 根据提示词的详细程度设置不同的分数
        if i == 0:  # 详细提示词
            tokens = 35
            accuracy = 0.85
            detail = 0.90
        elif i == 1:  # 中等提示词
            tokens = 20
            accuracy = 0.70
            detail = 0.60
        else:  # 简单提示词
            tokens = 10
            accuracy = 0.50
            detail = 0.30
        
        # 计算其他分数
        w_a = 0.6
        w_d = 0.4
        info_score = w_a * accuracy + w_d * detail
        final_score = info_score / tokens
        
        # 创建模拟单个结果
        individual_results = []
        for j in range(5):  # 5个样本
            # 添加一些随机性，使结果更真实
            noise = np.random.normal(0, 0.02)
            ind_accuracy = max(0, min(1, accuracy + noise))
            ind_detail = max(0, min(1, detail + noise))
            ind_info_score = w_a * ind_accuracy + w_d * ind_detail
            ind_final_score = ind_info_score / tokens
            
            individual_results.append({
                "image_id": f"person_{j+1}",
                "caption": f"这是第{j+1}张图片的描述...",
                "scores": {
                    "tokens": tokens,
                    "accuracy": ind_accuracy,
                    "detail": ind_detail,
                    "info_score": ind_info_score,
                    "final_score": ind_final_score
                }
            })
        
        # 创建模拟结果
        mock_results.append({
            "prompt": prompt,
            "num_samples": 5,
            "average_scores": {
                "tokens": tokens,
                "accuracy": accuracy,
                "detail": detail,
                "info_score": info_score,
                "final_score": final_score
            },
            "individual_results": individual_results
        })
    
    # 按综合得分排序
    mock_results.sort(key=lambda x: x["average_scores"]["final_score"], reverse=True)
    
    # 保存模拟结果
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(mock_results, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 创建模拟评估结果: {results_file}")

if __name__ == "__main__":
    main()
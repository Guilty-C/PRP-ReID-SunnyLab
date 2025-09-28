import argparse
import re
import numpy as np
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer
from PIL import Image
import torch
import os
import json
import yaml
import argparse
from tqdm import tqdm

# ========== 加载配置文件 ==========
def load_config(config_file):
    """
    加载YAML配置文件
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[INFO] 已加载配置文件: {config_file}")
        return config
    except Exception as e:
        print(f"[ERROR] 加载配置文件失败: {e}")
        return {}

# ========== 初始化 CLIP 模型 ==========
def init_clip_model(config):
    """
    根据配置初始化CLIP模型和处理器
    """
    model_name = config.get('clip', {}).get('model_name', 'openai/clip-vit-base-patch32')
    device_setting = config.get('clip', {}).get('device', 'auto')
    use_half_precision = config.get('clip', {}).get('use_half_precision', True)
    
    # 确定设备
    if device_setting == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_setting
    
    try:
        clip_model = CLIPModel.from_pretrained(model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # 使用半精度
        if use_half_precision and device == 'cuda':
            clip_model = clip_model.half()
            print(f"[INFO] CLIP模型已转换为半精度")
        
        print(f"[INFO] CLIP模型已加载到设备: {device}")
        return clip_model, clip_processor, device
    except Exception as e:
        print(f"[ERROR] 加载CLIP模型失败: {e}")
        return None, None, device

# ========== Token 数量计算函数 ==========
def count_tokens(prompt, tokenizer, config):
    """
    计算给定提示词的token数量
    """
    fallback_method = config.get('tokenizer', {}).get('fallback_method', 'space_split')
    
    try:
        if tokenizer:
            return len(tokenizer.encode(prompt))
        else:
            # 根据配置选择备选分词方法
            if fallback_method == 'space_split':
                return len(prompt.split())
            else:
                # 默认使用字符数
                return len(prompt)
    except Exception as e:
        print(f"[ERROR] 计算token数量失败: {e}")
        return 0

# ========== 评估准确性 ==========
def evaluate_accuracy(description, image, clip_model, clip_processor, device):
    """
    使用CLIP计算文本描述和图像的相似度
    返回值范围0~1，越高表示描述与图像越匹配
    """
    try:
        inputs = clip_processor(text=[description], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.softmax(dim=1)[0][0].item()
        return score
    except Exception as e:
        print(f"[ERROR] 评估准确性失败: {e}")
        return 0.0

# ========== 评估详细性 ==========
def evaluate_detail(description, config):
    """
    根据描述中提及的属性类别数量计算详细性分数
    """
    # 从配置中获取属性类别和关键词
    attributes = config.get('attributes', {
        "gender": ["男", "女", "man", "woman", "男性", "女性", "男生", "女生"],
        "top": ["T恤", "衬衫", "夹克", "上衣", "外套", "毛衣", "卫衣", "背心", "短袖", "长袖"],
        "bottom": ["裤", "裙", "裤子", "裙子", "牛仔裤", "短裤", "长裤"],
        "shoes": ["鞋", "鞋子", "运动鞋", "皮鞋", "靴子", "拖鞋"],
        "bag": ["包", "背包", "手提包", "挎包", "拎包"]
    })
    
    found = set()
    description_lower = description.lower()
    
    for attr, keywords in attributes.items():
        for kw in keywords:
            if kw in description_lower:
                found.add(attr)
                break
    
    return len(found) / len(attributes) if attributes else 0.0

# ========== 计算综合得分 ==========
def compute_score(prompt, description, image, tokenizer, clip_model, clip_processor, device, config):
    """
    计算提示词的综合得分
    S = I / T （信息密度），即单位token带来的有效信息量
    """
    # 从配置中获取权重
    w_a = config.get('weights', {}).get('accuracy', 0.6)
    w_d = config.get('weights', {}).get('detail', 0.4)
    
    # 计算token数量
    T = count_tokens(prompt, tokenizer, config)
    if T == 0:
        return {"tokens": 0, "accuracy": 0.0, "detail": 0.0, "info_score": 0.0, "final_score": 0.0}
    
    # 计算准确性得分
    A = evaluate_accuracy(description, image, clip_model, clip_processor, device)
    
    # 计算详细性得分
    D = evaluate_detail(description, config)
    
    # 计算信息程度分数
    I = w_a * A + w_d * D
    
    # 计算综合得分（信息密度）
    S = I / T
    
    return {"tokens": T, "accuracy": A, "detail": D, "info_score": I, "final_score": S}

# ========== 加载提示词 ==========
def load_prompts(prompts_dir):
    """
    从指定目录加载所有提示词模板
    """
    prompts = []
    try:
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(prompts_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                    prompts.append({"name": filename, "text": prompt_text})
        print(f"[INFO] 已加载 {len(prompts)} 个提示词模板")
    except Exception as e:
        print(f"[ERROR] 加载提示词失败: {e}")
    return prompts

# ========== 加载描述文件 ==========
def load_descriptions(descriptions_file):
    """
    加载已生成的描述文件（CSV格式）
    """
    descriptions = []
    try:
        with open(descriptions_file, "r", encoding="utf-8") as f:
            # 跳过表头
            next(f)
            for line in f:
                # 处理CSV行，注意处理可能包含逗号的caption字段
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    image_id = parts[0]
                    prompt = parts[1]
                    # caption可能包含逗号，需要特殊处理
                    caption = ",".join(parts[2:-1])
                    timestamp = parts[-1]
                    descriptions.append({
                        "image_id": image_id,
                        "prompt": prompt,
                        "caption": caption,
                        "timestamp": timestamp
                    })
        print(f"[INFO] 已加载 {len(descriptions)} 条描述数据")
    except Exception as e:
        print(f"[ERROR] 加载描述文件失败: {e}")
    return descriptions

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="提示词效果评估算法")
    parser.add_argument("--config", type=str, default="./configs/prompt_evaluation.yaml", help="配置文件路径")
    parser.add_argument("--prompts_dir", type=str, help="提示词模板目录")
    parser.add_argument("--descriptions_file", type=str, help="已生成的描述文件（CSV格式）")
    parser.add_argument("--images_dir", type=str, help="图像目录")
    parser.add_argument("--output_file", type=str, help="评估结果输出文件")
    parser.add_argument("--w_a", type=float, help="准确性权重")
    parser.add_argument("--w_d", type=float, help="详细性权重")
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 命令行参数优先级高于配置文件
    if args.prompts_dir:
        config['data_paths']['prompts_dir'] = args.prompts_dir
    if args.descriptions_file:
        config['data_paths']['descriptions_file'] = args.descriptions_file
    if args.images_dir:
        config['data_paths']['images_dir'] = args.images_dir
    if args.output_file:
        config['data_paths']['output_file'] = args.output_file
    if args.w_a is not None:
        config['weights']['accuracy'] = args.w_a
    if args.w_d is not None:
        config['weights']['detail'] = args.w_d
    
    # 确保必要的参数存在
    data_paths = config.get('data_paths', {})
    descriptions_file = data_paths.get('descriptions_file')
    images_dir = data_paths.get('images_dir', './data/market1501/bounding_box_test')
    output_file = data_paths.get('output_file', './outputs/prompt_evaluation.json')
    
    if not descriptions_file:
        print("[ERROR] 必须提供描述文件路径")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 初始化模型
    print("[INFO] 初始化CLIP模型...")
    clip_model, clip_processor, device = init_clip_model(config)
    if clip_model is None:
        print("[ERROR] 无法初始化CLIP模型，程序退出")
        return
    
    # 初始化tokenizer
    print("[INFO] 初始化tokenizer...")
    tokenizer = None
    try:
        tokenizer_model = config.get('tokenizer', {}).get('model_name', 'gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        print(f"[ERROR] 初始化tokenizer失败: {e}")
        print(f"[INFO] 将使用配置的备选分词方法")
    
    # 加载描述文件
    descriptions = load_descriptions(descriptions_file)
    if not descriptions:
        print("[ERROR] 没有加载到描述数据，程序退出")
        return
    
    # 按提示词分组描述
    prompt_to_descriptions = {}
    for desc in descriptions:
        if desc["prompt"] not in prompt_to_descriptions:
            prompt_to_descriptions[desc["prompt"]] = []
        prompt_to_descriptions[desc["prompt"]].append(desc)
    
    # 评估每个提示词
    results = []
    show_progress_bar = config.get('advanced', {}).get('show_progress_bar', True)
    
    if show_progress_bar:
        iterator = tqdm(prompt_to_descriptions.items(), desc="评估提示词")
    else:
        iterator = prompt_to_descriptions.items()
    
    for prompt_text, prompt_descriptions in iterator:
        prompt_results = []
        
        for desc in prompt_descriptions:
            try:
                # 加载对应的图像
                image_id = desc["image_id"]
                # 假设图像文件名与image_id对应
                image_path = os.path.join(images_dir, f"{image_id}.jpg")
                
                # 检查图像是否存在
                if not os.path.exists(image_path):
                    # 尝试其他可能的扩展名
                    for ext in [".png", ".jpeg"]:
                        alt_path = os.path.join(images_dir, f"{image_id}{ext}")
                        if os.path.exists(alt_path):
                            image_path = alt_path
                            break
                    else:
                        print(f"[WARNING] 图像文件不存在: {image_id}")
                        continue
                
                # 打开图像
                image = Image.open(image_path).convert("RGB")
                
                # 计算得分
                score_result = compute_score(
                    prompt_text, desc["caption"], image, tokenizer, 
                    clip_model, clip_processor, device, 
                    config
                )
                
                # 保存结果
                prompt_results.append({
                    "image_id": image_id,
                    "caption": desc["caption"],
                    "scores": score_result
                })
                
            except Exception as e:
                print(f"[ERROR] 处理图像 {desc['image_id']} 失败: {e}")
                continue
        
        # 如果有结果，计算该提示词的平均得分
        if prompt_results:
            avg_tokens = np.mean([r["scores"]["tokens"] for r in prompt_results])
            avg_accuracy = np.mean([r["scores"]["accuracy"] for r in prompt_results])
            avg_detail = np.mean([r["scores"]["detail"] for r in prompt_results])
            avg_info_score = np.mean([r["scores"]["info_score"] for r in prompt_results])
            avg_final_score = np.mean([r["scores"]["final_score"] for r in prompt_results])
            
            # 根据配置决定是否保存单个结果
            save_individual_results = config.get('output', {}).get('save_individual_results', True)
            
            result_entry = {
                "prompt": prompt_text,
                "num_samples": len(prompt_results),
                "average_scores": {
                    "tokens": avg_tokens,
                    "accuracy": avg_accuracy,
                    "detail": avg_detail,
                    "info_score": avg_info_score,
                    "final_score": avg_final_score
                }
            }
            
            if save_individual_results:
                result_entry["individual_results"] = prompt_results
            
            results.append(result_entry)
    
    # 根据配置排序结果
    sort_by = config.get('output', {}).get('sort_by', 'final_score')
    sort_order = config.get('output', {}).get('sort_order', 'descending')
    
    try:
        results.sort(
            key=lambda x: x["average_scores"][sort_by], 
            reverse=(sort_order == 'descending')
        )
    except KeyError:
        print(f"[WARNING] 排序键 {sort_by} 不存在，使用默认排序")
        results.sort(key=lambda x: x["average_scores"]["final_score"], reverse=True)
    
    # 保存结果
    pretty_print = config.get('output', {}).get('pretty_print', True)
    indent = 2 if pretty_print else None
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=indent)
    
    # 输出总结
    print("\n[INFO] 提示词评估结果总结：")
    for i, result in enumerate(results[:5]):  # 只显示前5个
        print(f"\n排名 {i+1}:")
        print(f"提示词: {result['prompt'][:50]}...")
        print(f"样本数: {result['num_samples']}")
        print(f"平均token数: {result['average_scores']['tokens']:.2f}")
        print(f"平均准确性: {result['average_scores']['accuracy']:.4f}")
        print(f"平均详细性: {result['average_scores']['detail']:.4f}")
        print(f"平均信息分数: {result['average_scores']['info_score']:.4f}")
        print(f"平均综合得分: {result['average_scores']['final_score']:.6f}")
    
    print(f"\n[INFO] 完整评估结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
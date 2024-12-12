import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from pathlib import Path
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def calculate_cost(token_stats: dict) -> float:
    """计算API调用成本
    
    Args:
        token_stats: token使用统计字典
        
    Returns:
        float: 花费金额(美元)
    """
    input_tokens = token_stats["input_tokens"] + token_stats["prompt_tokens"]
    output_tokens = token_stats["completion_tokens"]
    return (input_tokens * 1.5 + output_tokens * 5) / 1000000

def log_consumption(video_path: str, response: str, token_stats: dict, cost: float):
    """记录API调用消费
    
    Args:
        video_path: 处理的视频路径
        response: 响应内容
        token_stats: token使用统计
        cost: 花费金额
    """
    log_dir = Path("log/consume")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "video.json"
    
    # 读取现有记录
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
    
    # 添加新记录
    logs.append({
        "video_path": video_path,
        "response": response,
        "token_stats": token_stats,
        "cost_usd": cost
    })
    
    # 保存记录
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def analyze_video(video_path: str, prompt_text: str = "请分析以下视频内容：", model: str = "pro") -> dict:
    """
    使用 Gemini API 分析视频文件
    
    Args:
        video_path: 视频文件路径 (仅支持 .mp4 格式)
        prompt_text: 提示文本，默认为"请分析以下视频内容："
        model: 模型选择，可选 "lite" 或 "pro"，默认为 "pro"
    
    Returns:
        dict: 包含分析结果和token统计的字典
    """
    # 模型名称映射
    model_mapping = {
        "lite": "gemini-1.5-flash",
        "pro": "gemini-1.5-pro"
    }
    
    # 获取实际的模型名称
    model_name = model_mapping[model]
        
    try:
        with open(video_path, "rb") as video_file:
            video_content = video_file.read()
            video_data = {
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": video_content
                }
            }
            
        # genai
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt_text, video_data])
        
        # 获取完整的token使用统计
        input_tokens = model.count_tokens([prompt_text, video_data])
        usage_metadata = response.usage_metadata
        
        result = {
            "content": response.text,
            "token_stats": {
                "input_tokens": input_tokens.total_tokens,
                "prompt_tokens": usage_metadata.prompt_token_count,
                "completion_tokens": usage_metadata.candidates_token_count,
                "total_tokens": usage_metadata.total_token_count
            }
        }
        
        # 计算并记录消费
        cost = calculate_cost(result["token_stats"])
        log_consumption(video_path, result["content"], result["token_stats"], cost)
        result["cost_usd"] = cost
        
        return result
            
    except Exception as e:
        return {"error": f"处理失败: {str(e)}"}

if __name__ == "__main__":
    # 使用示例：
    video_path = "llm/test_data/fanning.mp4"
    prompt = "请分析这个视频中的人在做什么动作，是否和热舒适状态有关，用 json 格式回答{{'content': '视频描述'}}"
    result = analyze_video(video_path, prompt, model="pro")
    print(result["content"])  # 打印分析结果
    print("Token统计:", result["token_stats"])  # 打印token统计信息
    print(f"花费 $ {result['cost_usd']:.6f}")
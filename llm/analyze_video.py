import google.generativeai as genai
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 配置 API 密钥
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_video(video_path: str, prompt_text: str = "请分析以下视频内容：", model: str = "pro") -> str:
    """
    使用 Gemini API 分析视频文件
    
    Args:
        video_path: 视频文件路径 (支持 .mp4 格式)
        prompt_text: 提示文本，默认为"请分析以下视频内容："
        model: 模型选择，可选 "lite" 或 "pro"，默认为 "pro"
    
    Returns:
        str: 分析结果文本，如果失败则返回错误信息
    """
    # 模型名称映射
    model_mapping = {
        "lite": "gemini-1.5-flash",
        "pro": "gemini-1.5-pro"
    }
    
    # 验证模型参数
    if model not in model_mapping:
        raise ValueError(f"不支持的模型类型: {model}，仅支持 {', '.join(model_mapping.keys())}")
    
    # 获取实际的模型名称
    model_name = model_mapping[model]

    # 构建提示格式
    prompt = [{"text": prompt_text}]
    
    # 检查文件格式
    ext = video_path.lower().split('.')[-1]
    if ext != 'mp4':
        raise ValueError(f"不支持的视频格式: {ext}，仅支持 mp4 格式")
        
    try:
        with open(video_path, "rb") as video_file:
            video_content = video_file.read()
            prompt.append({
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": video_content
                }
            })
            
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
            
    except Exception as e:
        return f"处理失败: {str(e)}"

# 使用示例：
result = analyze_video("llm/test_data/fanning.mp4", "请分析这个视频中的人在做什么动作，是否和热舒适状态有关，用 json 格式回答{{'content': '视频描述'}}", model="pro")
print(result)

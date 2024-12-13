import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import cv2
import base64
from datetime import datetime
from pathlib import Path
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_frames(video_path: str, fps: int = 5) -> list:
    """
    从视频中提取指定帧率的图片
    
    Args:
        video_path: 视频文件路径
        fps: 每秒提取的帧数，默认5帧
    
    Returns:
        list: 包含所有提取帧的列表
    """
    frames = []
    video = cv2.VideoCapture(video_path)
    
    # 获取视频的FPS和总帧数
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # 将BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 编码为JPEG格式
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            # 转换为base64字符串
            image_data = base64.b64encode(buffer).decode('utf-8')
            frames.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
            })
        
        frame_count += 1
    
    video.release()
    return frames

def analyze_video(video_path: str, prompt_text: str = "请分析以下视频内容：", fps: int = 5) -> dict:
    """
    使用 Gemini API 分析视频帧
    
    Args:
        video_path: 视频文件路径
        prompt_text: 提示文本
        fps: 每秒提取的帧数
    
    Returns:
        dict: 包含分析结果和token统计的字典
    """
    try:
        # 提取视频帧
        frames = extract_frames(video_path, fps)
        
        # 构建输入内容
        input_content = [prompt_text] + frames
            
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(input_content)
        
        # 获取完整的token使用统计
        input_tokens = model.count_tokens(input_content)
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
        
        return result
            
    except Exception as e:
        return {"error": f"处理失败: {str(e)}"}

if __name__ == "__main__":
    # 使用示例：
    video_path = "llm/test_data/fanning.mp4"
    prompt = "请分析这个视频中的人在做什么动作，是否和热舒适状态有关，用 json 格式回答{{'content': '视频描述'}}"
    result = analyze_video(video_path, prompt, fps=5)
    print(result["content"])  # 打印分析结果
    print("Token统计:", result["token_stats"])  # 打印token统计信息
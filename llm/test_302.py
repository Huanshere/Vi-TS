from dotenv import load_dotenv
import requests
import json
import os
import cv2
import base64

load_dotenv()
MODEL = "gemini-2.0-flash-exp"
URL = "https://api.302.ai/v1/chat/completions"
API_KEY = os.getenv("API_302_KEY")

# TODO 注意分辨率缩放 768x768 以下都算 258 tokens
# 1~5 fps 根据场景动态选择，动作幅度大 5，动作幅度小 3，无动作幅度 1 甚至 0.4
def extract_frames(video_path: str, fps: int = 5) -> list:
    """
    从视频中提取指定帧率的图片
    
    Args:
        video_path: 视频文件路径
        fps: 每秒提取的帧数，默认5帧
    
    Returns:
        list: 包含所有提取帧的图片URL列表
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
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })
        
        frame_count += 1
    
    video.release()
    return frames

def analyze_video(video_path: str, prompt_text: str = "请分析以下视频内容：", fps: int = 5) -> dict:
    """
    使用 302.ai API 分析视频帧
    
    Args:
        video_path: 视频文件路径
        prompt_text: 提示文本
        fps: 每秒提取的帧数
    
    Returns:
        dict: API 响应结果
    """
    
    # 提取视频帧
    frames = extract_frames(video_path, fps)
    
    # 构建消息内容
    content = [{"type": "text", "text": prompt_text}] + frames
    
    payload = json.dumps({
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "response_format": {"type": "json_object"}
    })
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(URL, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"处理失败: {str(e)}"}

if __name__ == "__main__":
    # 使用示例：
    video_path = "llm/test_data/fanning.mp4"
    prompt = "请分析这个视频中的人在做什么动作，是否和热舒适状态有关，用json格式回答{{'content': '视频描述'}}"
    result = analyze_video(video_path, prompt, fps=5)
    print(json.dumps(result, ensure_ascii=False, indent=2))
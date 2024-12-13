from dotenv import load_dotenv
import requests
import json
import os
import cv2
import base64
from pathlib import Path
from datetime import datetime
from rich import print as rprint
import json_repair

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
            # 直接使用原始 BGR 帧，不进行颜色映射
            _, buffer = cv2.imencode('.jpg', frame)
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

def log_consumption(video_path: str, response: dict):
    """记录API调用记录
    
    Args:
        video_path: 处理的视频路径
        response: 响应内容
    """
    log_dir = Path("log/consume")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "video_302.json"
    
    # 读取现有记录
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
    
    # 添加新记录
    logs.append({
        "timestamp": datetime.now().isoformat(),
        "video_path": video_path,
        "response": response,
        "token_stats": response.get("usage", {})
    })
    
    # 保存记录
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def analyze_video(video_path: str, prompt_text: str = "Analyze the following video:", fps: int = 5) -> dict:
    """
    Analyze video frames using 302.ai API
    
    Args:
        video_path: Path to video file
        prompt_text: Prompt text
        fps: Frames per second to extract
    
    Returns:
        dict: API response
    """
    rprint("🎥 [bold blue]Extracting frames from video...[/]")
    frames = extract_frames(video_path, fps)
    rprint(f"✨ [green]Successfully extracted {len(frames)} frames[/]")
    
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
        rprint("🚀 [bold yellow]Sending request to API...[/]")
        response = requests.post(URL, headers=headers, data=payload)
        response.raise_for_status()
        result = response.json()
        
        rprint("📝 [blue]Logging consumption data...[/]")
        log_consumption(video_path, result)
        
        rprint("✅ [bold green]Analysis completed successfully![/]")
        return result
    except Exception as e:
        error_msg = f"❌ Processing failed: {str(e)}"
        rprint(f"[bold red]{error_msg}[/]")
        return {"error": error_msg}

if __name__ == "__main__":
    video_path = "llm/test_data/fanning.mp4"
    prompt = "请分析这个视频中的人在做什么动作，是否和热舒适状态有关，用 JSON 格式回答: {{'content': '视频描述'}}"
    
    rprint("\n🎬 [bold cyan]Starting Video Analysis[/]\n")
    result = analyze_video(video_path, prompt, fps=3)
    
    rprint("\n📊 [bold magenta]Analysis Results:[/]")
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    rprint(f"[green]{json.dumps(json_repair.loads(content), indent=2, ensure_ascii=False)}[/]")
    
    rprint("\n📈 [bold magenta]Token Statistics:[/]")
    rprint(f"[yellow]{json.dumps(result.get('usage', {}), indent=2)}[/]")
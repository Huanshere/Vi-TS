from dotenv import load_dotenv
import requests
import json
import cv2
import base64
from pathlib import Path
from datetime import datetime
from rich import print as rprint
import json_repair

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.prompts import get_analyze_clothing_prompt, get_analyze_video_prompt

load_dotenv()
MODEL = "gemini-2.0-flash-exp"
URL = "https://api.302.ai/v1/chat/completions"
API_KEY = os.getenv("API_302_KEY")

def extract_frames(video_path: str, fps: int = 5) -> list:
    """从视频提取帧"""
    frames, video = [], cv2.VideoCapture(video_path)
    frame_interval = int(video.get(cv2.CAP_PROP_FPS) / fps)
    
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret: break
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"}
            })
        frame_count += 1
    
    video.release()
    return frames

def log_consumption(file_path: str, response: dict, log_title: str = "video_302"):
    """记录API调用记录"""
    log_file = Path(f"log/consume/{log_title}.json")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logs = json.load(log_file.open('r', encoding="utf-8")) if log_file.exists() else []
    logs.append({
        "timestamp": datetime.now().isoformat(),
        "file_path": file_path,
        "response": response
    })
    
    json.dump(logs, log_file.open('w', encoding="utf-8"), ensure_ascii=False, indent=2)

def analyze_frames(frames: list, prompt_text: str = "Analyze the following images:") -> dict:
    """分析图片序列"""
    try:
        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}] + frames}],
            "response_format": {"type": "json_object"}
        }
        
        rprint("🚀 [bold cyan]Sending request to API...[/]")
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        response = requests.post(URL, headers=headers, json=payload)
        response.raise_for_status()
        rprint("✅ [bold green]Analysis completed successfully![/]")
        result = json_repair.loads(response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
        return result
    except Exception as e:
        error_msg = f"❌ Processing failed: {str(e)}"
        rprint(f"[bold red]{error_msg}[/]")
        return {"error": error_msg}

def analyze_video(video_path: str, fps: int = 5, print_result: bool = True) -> dict:
    """分析视频帧"""
    rprint(f"\n🎬 [bold cyan]Starting Video Analysis <{video_path}> ...[/]")
    frames = extract_frames(video_path, fps)
    rprint(f"✨ [bold green]Successfully extracted <{len(frames)}> frames[/]")
    
    prompt = get_analyze_video_prompt()
    result = analyze_frames(frames, prompt)
    log_consumption(video_path, result, log_title="video_302")
    if print_result:
        rprint(f"[green]{json.dumps(result, indent=2, ensure_ascii=False)}[/]")
    return result

def analyze_image(image_path: str, print_result: bool = True) -> dict:
    """分析单张图片"""
    rprint(f"\n🖼️ [bold cyan]Starting Image Analysis <{image_path}> ...[/]")
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    frames = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"}
    }]
    rprint("🖼️ [bold blue]Processing image...[/]")
    prompt = get_analyze_clothing_prompt()
    result = analyze_frames(frames, prompt)
    log_consumption(image_path, result, log_title="image_302")
    if print_result:
        rprint(f"[green]{json.dumps(result, indent=2, ensure_ascii=False)}[/]")
    return result

if __name__ == "__main__":
    video_path = "ai/test_data/fanning.mp4"
    result = analyze_video(video_path, fps=5)

    img_path = "ai/test_data/cloth.png"
    result = analyze_image(img_path)
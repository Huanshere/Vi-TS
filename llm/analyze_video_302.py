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
        "response": response,
        "token_stats": response.get("usage", {})
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
        return response.json()
    except Exception as e:
        error_msg = f"❌ Processing failed: {str(e)}"
        rprint(f"[bold red]{error_msg}[/]")
        return {"error": error_msg}

def analyze_video(video_path: str, prompt_text: str = "Analyze the following video:", fps: int = 5) -> dict:
    """分析视频帧"""
    rprint("🎥 [bold cyan]Extracting frames from video...[/]")
    frames = extract_frames(video_path, fps)
    rprint(f"✨ [bold green]Successfully extracted {len(frames)} frames[/]")
    
    result = analyze_frames(frames, prompt_text)
    
    if "error" not in result:
        rprint("📝 [bold cyan]Logging consumption data...[/]")
        log_consumption(video_path, result, log_title="video_302")
    
    return result

def analyze_image(image_path: str, prompt_text: str = "Analyze this image:") -> dict:
    """分析单张图片"""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    frames = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"}
    }]
    rprint("🖼️ [bold blue]Processing image...[/]")
    result = analyze_frames(frames, prompt_text)
    log_consumption(image_path, result, log_title="image_302")
    return result

if __name__ == "__main__":
    video_path = "llm/test_data/fanning.mp4"
    prompt = "请分析这个视频中的人在做什么动作，是否和热舒适状态有关，用 JSON 格式回答: {{'content': '视频描述', 'thermal_comfort': '是否和热舒适状态有关'}}"
    
    rprint("\n🎬 [bold cyan]Starting Video Analysis[/]\n")
    result = analyze_video(video_path, prompt, fps=3)
    
    rprint("\n📊 [bold magenta]Analysis Results:[/]")
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    rprint(f"[green]{json.dumps(json_repair.loads(content), indent=2, ensure_ascii=False)}[/]")

    img_path = "llm/test_data/cloth.png"
    prompt = "请分析这张图片中的人的衣着，然后计算服装热阻 clo，用 JSON 格式回答: {{'content': '图片描述','clo': '服装热阻'}}"
    rprint("\n🖼️ [bold cyan]Starting Image Analysis[/]\n")
    result = analyze_image(img_path, prompt)
    rprint("\n📊 [bold magenta]Analysis Results:[/]")
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    rprint(f"[green]{json.dumps(json_repair.loads(content), indent=2, ensure_ascii=False)}[/]")

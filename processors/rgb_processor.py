import os
import sys
import time
from multiprocessing import Queue, Process, Pool
from rich import print as rprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rgb_cam_utils import *
from ai.analyze_video_302 import analyze_video, analyze_image

# 图片和视频分析间隔参数
PROCESS_WORKER_COUNT = 6
IMAGE_ANALYZE_GAP = 30  # 每 30 秒分析一次衣服
VIDEO_DURATION = 5  # 每个分析的视频片段时长(秒)
VIDEO_ANALYZE_FPS = 5 # 分析视频的帧率

# 配置参数
ROTATION = 180 if platform.system() == 'Linux' or platform.system() == 'Windows' else 0 #! 我的 linux 和 windows 上 cam 装反了

VIDEO_RATIO = 16/9  # 视频比例 16:9
SAVE_RESOLUTION = (640, int(640/VIDEO_RATIO))  # 约为 (640, 360)

VIDEO_DIR = "log/video/"
IMAGE_DIR = "log/image/"

def ensure_output_dirs():
    """确保输出目录存在"""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

def rgb_stream(video_q, image_q):
    """视频捕捉和生产者进程"""
    cap = init_camera()
    start_time = time.time()
    current_video_frames = []
    
    target_frames = VIDEO_DURATION * VIDEO_ANALYZE_FPS
    frame_interval = 1.0 / VIDEO_ANALYZE_FPS
    next_frame_time = start_time
    
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = rotate_frame(frame, ROTATION)
            
        if current_time >= next_frame_time:
            current_video_frames.append(frame.copy())
            next_frame_time = start_time + len(current_video_frames) * frame_interval
            
            if len(current_video_frames) >= target_frames:
                timestamp = get_timestamp()
                video_data = {
                    "type": "video",
                    "frames": current_video_frames,
                    "timestamp": timestamp
                }
                video_q.put(video_data)
                
                start_time = current_time
                next_frame_time = start_time
                current_video_frames = []
        
        timestamp = get_timestamp()
        image_data = {
            "type": "image",
            "frame": frame,
            "timestamp": timestamp
        }
        image_q.put(image_data)
        
        cv2.imshow('RGB Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_video_worker(frames, timestamp):
    """单个视频处理工作进程"""
    video_path = os.path.join(VIDEO_DIR, f"video_{timestamp}.mp4")
    save_video(frames, video_path, VIDEO_ANALYZE_FPS, SAVE_RESOLUTION)
    
    try:
        result = analyze_video(video_path, fps=VIDEO_ANALYZE_FPS)
        return result
    except Exception as e:
        rprint(f"❌ [bold red]Error analyzing video: {str(e)}[/]")
        return {"error": str(e)}

def process_image_worker(frame, timestamp):
    """单个图片处理工作进程"""
    image_path = os.path.join(IMAGE_DIR, f"image_{timestamp}.png")
    save_image(frame, image_path)
    
    try:
        result = analyze_image(image_path)
        return result
    except Exception as e:
        rprint(f"❌ [bold red]Error analyzing image: {str(e)}[/]")
        return {"error": str(e)}

def process_video(video_q):
    """视频处理消费者进程"""
    pool = Pool(processes=PROCESS_WORKER_COUNT)
    
    while True:
        video_data = video_q.get()
        timestamp = video_data["timestamp"]
        frames = video_data["frames"]
        
        # 异步提交视频处理任务
        pool.apply_async(process_video_worker, args=(frames, timestamp))

def process_image(image_q):
    """图片处理消费者进程"""
    pool = Pool(processes=PROCESS_WORKER_COUNT)
    last_analyze_time = 0
    
    while True:
        image_data = image_q.get()
        timestamp = image_data["timestamp"]
        frame = image_data["frame"]
        
        current_time = time.time()
        # 检查是否达到分析间隔时间
        if current_time - last_analyze_time >= IMAGE_ANALYZE_GAP:
            pool.apply_async(process_image_worker, args=(frame, timestamp))
            last_analyze_time = current_time

if __name__ == "__main__":
    # 创建输出目录
    ensure_output_dirs()
    
    # 创建队列
    video_queue = Queue()
    image_queue = Queue()
    
    # 创建并启动进程
    producer = Process(target=rgb_stream, args=(video_queue, image_queue))
    video_consumer = Process(target=process_video, args=(video_queue,))
    image_consumer = Process(target=process_image, args=(image_queue,))
    
    producer.start()
    video_consumer.start()
    image_consumer.start()
    
    try:
        producer.join()
    finally:
        video_consumer.terminate()
        image_consumer.terminate()


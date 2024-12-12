from cam_utils.rgb_utils import *
from multiprocessing import Queue, Process, Pool
import time
import os
from datetime import datetime

from llm.analyze_video import analyze_video

# 配置参数
ROTATION = 180
VIDEO_DURATION = 10  # 视频片段时长(秒)

VIDEO_DIR = "log/video/"
IMAGE_DIR = "log/image/"

PROCESS_WORKER_COUNT = 6
SAVE_RESOLUTION = (426, 240)
SAVE_FPS = 10

def ensure_output_dirs():
    """确保输出目录存在"""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

def rgb_stream(video_q, image_q):
    """视频捕捉和生产者进程"""
    cap = init_camera()
    frame_count = 0
    start_time = time.time()
    current_video_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = rotate_frame(frame, ROTATION)
        frame_count += 1
        
        # 收集视频帧
        current_video_frames.append(frame.copy())
        
        # 每10秒保存一段视频
        if time.time() - start_time >= VIDEO_DURATION:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_data = {
                "type": "video",
                "frames": current_video_frames,
                "timestamp": timestamp
            }
            video_q.put(video_data)
            
            # 同时保存当前帧作为图片
            image_data = {
                "type": "image",
                "frame": frame.copy(),
                "timestamp": timestamp
            }
            image_q.put(image_data)
            
            # 重置计数器
            start_time = time.time()
            current_video_frames = []
        
        # 显示实时画面
        cv2.imshow('RGB Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_video_worker(frames, timestamp):
    """单个视频处理工作进程"""
    video_path = os.path.join(VIDEO_DIR, f"video_{timestamp}.mp4")
    width, height = SAVE_RESOLUTION
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, SAVE_FPS, (width, height))
    
    for frame in frames:
        resized_frame = cv2.resize(frame, SAVE_RESOLUTION)
        out.write(resized_frame)
    out.release()

    print(f"保存视频片段: {video_path}")

    # 分析视频
    print(f"开始分析视频片段: {video_path}")
    prompt = "请分析这个视频中的人在做什么动作，是否和热舒适状态有关，用 json 格式回答{{'content': '视频描述'}}"
    result = analyze_video(video_path, prompt, model="pro")
    print(f"分析视频片段: {video_path}, 完成, 结果: {result['content']}")

def process_video(video_q):
    """视频处理消费者进程"""
    pool = Pool(processes=PROCESS_WORKER_COUNT)
    
    while True:
        video_data = video_q.get()
        timestamp = video_data["timestamp"]
        frames = video_data["frames"]
        
        # 异步提交视频处理任务
        pool.apply_async(process_video_worker, args=(frames, timestamp))

def process_image_worker(frame, timestamp):
    """单个图片处理工作进程"""
    image_path = os.path.join(IMAGE_DIR, f"image_{timestamp}.png")
    cv2.imwrite(image_path, frame)
    print(f"保存图片: {image_path}")

def process_image(image_q):
    """图片处理消费者进程"""
    pool = Pool(processes=PROCESS_WORKER_COUNT)
    
    while True:
        image_data = image_q.get()
        timestamp = image_data["timestamp"]
        frame = image_data["frame"]
        
        # 异步提交图片处理任务
        pool.apply_async(process_image_worker, args=(frame, timestamp))

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


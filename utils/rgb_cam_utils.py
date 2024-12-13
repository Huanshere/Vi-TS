import platform
import cv2
from rich import print as rprint

def init_camera():
    system = platform.system()
    if system == "Linux":
        # Linux: 使用 check_cameras 获取摄像头 ID
        from utils.check_cam import check_specific_cameras
        _, rgb_cam_id = check_specific_cameras()
        cap = cv2.VideoCapture('/dev/video' + str(rgb_cam_id))
    elif system == "Darwin" or system == "Windows":
        cap = cv2.VideoCapture(0) # 使用默认0
    
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")
    
    # 尝试设置1080p分辨率
    target_width, target_height = 640, 480
    rprint(f"尝试设置分辨率: {target_width}x{target_height}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    
    # 确认设置后的分辨率
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if actual_width != target_width or actual_height != target_height:
        rprint(f"[yellow]警告: 无法设置为目标分辨率！实际分辨率: {actual_width}x{actual_height}[/yellow]")
    
    return cap

def rotate_frame(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def save_video(frames, video_path, fps, resolution):
    """保存视频帧到文件"""
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames:
        resized_frame = cv2.resize(frame, resolution)
        out.write(resized_frame)
    out.release()
    rprint(f"💾 [bold green]Saved video clip:[/] {video_path}")

def save_image(frame, image_path):
    """保存单帧图像到文件"""
    cv2.imwrite(image_path, frame)
    rprint(f"💾 [bold green]Saved image:[/] {image_path}")

def get_timestamp():
    """获取当前时间戳字符串"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    cap = init_camera()
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


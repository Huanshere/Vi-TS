import cv2
import os

def check_camera(cam_id):
    """检查指定ID的摄像头并打印信息"""
    try:
        # 尝试打开摄像头
        cap = cv2.VideoCapture(f'/dev/video{cam_id}')
        
        if not cap.isOpened():
            print(f"Camera {cam_id}: 无法打开")
            return False
        
        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\nCamera {cam_id}:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        
        # 尝试读取一帧
        ret, frame = cap.read()
        if ret:
            print("Status: 可以正常读取图像")
        else:
            print("Status: 无法读取图像")
        
        # 释放摄像头
        cap.release()
        return True
        
    except Exception as e:
        print(f"Camera {cam_id}: 错误 - {str(e)}")
        return False

def main():
    print("正在扫描可用摄像头...\n")
    
    # 检查 /dev/video* 设备
    video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
    
    if not video_devices:
        print("未找到任何摄像头设备")
        return
    
    print(f"找到 {len(video_devices)} 个可能的视频设备")
    
    # 逐个检查摄像头
    for device in sorted(video_devices):
        cam_id = device.replace('video', '')
        check_camera(int(cam_id))

if __name__ == "__main__":
    main()

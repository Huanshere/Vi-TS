def check_camera(cam_id):
    """检查指定ID的摄像头并打印详细信息"""
    try:
        # 尝试打开摄像头
        cap = cv2.VideoCapture(f'/dev/video{cam_id}')
        
        if not cap.isOpened():
            print(f"摄像头 {cam_id}: 无法打开")
            return False
        
        # 获取所有可用的摄像头属性
        properties = {
            'WIDTH': cv2.CAP_PROP_FRAME_WIDTH,
            'HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,
            'FPS': cv2.CAP_PROP_FPS,
            'FORMAT': cv2.CAP_PROP_FORMAT,
            'MODE': cv2.CAP_PROP_MODE,
            'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS
        }
        
        print(f"\n📷 摄像头 {cam_id}:")
        
        # 获取并打印所有属性
        for name, prop in properties.items():
            value = cap.get(prop)
            if value != -1:  # 某些属性可能不被支持
                if name in ['WIDTH', 'HEIGHT']:
                    print(f"- {name}: {int(value)}px")
                elif name == 'FPS':
                    print(f"- {name}: {value:.1f}")
                else:
                    print(f"- {name}: {value}")
        
        # 尝试读取一帧
        ret, frame = cap.read()
        if ret:
            print("- 状态: ✅ 可以正常读取图像")
            print(f"- 实际分辨率: {frame.shape[1]}x{frame.shape[0]}")
            # 特别检查是否为256x384
            if frame.shape[1] == 256 and frame.shape[0] == 384:
                print("⚠️ 注意: 检测到256x384分辨率")
        else:
            print("- 状态: ❌ 无法读取图像")
        
        # 尝试设置不同的分辨率
        test_resolutions = [
            (256, 384),  # 特别关注的分辨率
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (2560, 1440)
        ]
        
        print("\n支持的分辨率:")
        for width, height in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width == width and actual_height == height:
                mark = "✅"
                if width == 256 and height == 384:
                    mark = "⭐"  # 特别标记256x384分辨率
                print(f"{mark} {width}x{height}")
            else:
                print(f"❌ {width}x{height}")
        
        # 释放摄像头
        cap.release()
        return True
        
    except Exception as e:
        print(f"摄像头 {cam_id}: 错误 - {str(e)}")
        return False

def main():
    print("🔍 正在扫描可用摄像头...\n")
    
    # 检查 /dev/video* 设备
    video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
    
    if not video_devices:
        print("❌ 未找到任何摄像头设备")
        return
    
    print(f"📝 找到 {len(video_devices)} 个可能的视频设备")
    
    # 逐个检查摄像头
    for device in sorted(video_devices):
        cam_id = device.replace('video', '')
        check_camera(int(cam_id))

if __name__ == "__main__":
    import cv2
    import os
    main()

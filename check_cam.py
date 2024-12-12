import cv2

def check_specific_cameras():
    """
    检查前10个摄像头，寻找特定配置的两个摄像头
    返回: 元组 (256x384摄像头ID, 640x480摄像头ID) 或在不满足条件时抛出异常
    """
    available_cams = []
    
    # 检查前10个摄像头
    for cam_id in range(10):
        try:
            cap = cv2.VideoCapture(f'/dev/video{cam_id}')
            if not cap.isOpened():
                continue
                
            # 测试256x384
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == 256 and actual_height == 384:
                available_cams.append((cam_id, "256x384"))
                
            # 测试640x480
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == 640 and actual_height == 480:
                available_cams.append((cam_id, "640x480"))
                
            cap.release()
            
        except Exception:
            continue
    
    # 检查结果
    if len(available_cams) != 2:
        raise Exception(f"需要恰好2个摄像头，但找到了 {len(available_cams)} 个")
        
    # 确保一个是256x384，一个是640x480
    cam_256x384 = None
    cam_640x480 = None
    
    for cam_id, resolution in available_cams:
        if resolution == "256x384":
            cam_256x384 = cam_id
        elif resolution == "640x480":
            cam_640x480 = cam_id
            
    if cam_256x384 is None or cam_640x480 is None:
        raise Exception("未找到所需的摄像头配置(需要一个256x384和一个640x480)")
        
    return (cam_256x384, cam_640x480)

def main():
    try:
        cam_small, cam_large = check_specific_cameras()
        print(f"✅ 找到符合要求的摄像头:")
        print(f"- 256x384摄像头: /dev/video{cam_small}")
        print(f"- 640x480摄像头: /dev/video{cam_large}")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")

if __name__ == "__main__":
    main()

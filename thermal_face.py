# 系统相关
import os
import sys
import time
import cv2
import numpy as np
import json
from rich import print as rprint
from check_cam import check_specific_cameras

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from face_detect_setting import *

# Global variables
COUNTER = 0
FPS = 0
START_TIME = 0
DETECTION_RESULT = None

# Model parameters
try:
    CAMERA_ID, _ = check_specific_cameras()
except Exception as e:
    rprint(f"[red]❌ 错误：无法找到合适的摄像头: {str(e)}[/red]")
    sys.exit(1)
GAP = 5  # 每隔5秒保存一次

def save_result(result, unused_output_image, timestamp_ms):
    global FPS, COUNTER, START_TIME, DETECTION_RESULT
    if COUNTER % FPS_AVG_FRAME_COUNT == 0:
        FPS = FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
        START_TIME = time.time()
    DETECTION_RESULT = result
    COUNTER += 1
    return timestamp_ms

def run():
    last_save_time = time.time()
    options.result_callback = save_result
    # 打开摄像头
    cap = cv2.VideoCapture('/dev/video' + str(CAMERA_ID), cv2.CAP_V4L)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from thermal camera.')

        # 分离图像数据
        imdata, thdata = np.array_split(frame, 2)
        # 转换颜色格式
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=0.6)
        # 应用热图
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_BONE)

        # 创建mediapipe图像
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=heatmap)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # 显示FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (LEFT_MARGIN, ROW_SIZE)
        cv2.putText(heatmap, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        if DETECTION_RESULT:
            for face_landmarks in DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in face_landmarks
                ])

                # 绘制面部标志点
                mp_drawing.draw_landmarks(
                    image=heatmap,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # 获取并保存面部温度
                face_keypoints = {10: 'Nose', 234: 'Left Eye', 454: 'Right Eye', 152: 'Mouth'}
                for keypoint_id, keypoint_name in face_keypoints.items():
                    temp_avg, temp_matrix = get_landmark_temp(keypoint_id, face_landmarks, heatmap, thdata)

                    # Get the coordinates for placing the text
                    landmark = face_landmarks[keypoint_id]
                    HEIGHT, WIDTH, _ = heatmap.shape
                    x, y = int(landmark.x * WIDTH), int(landmark.y * HEIGHT)
                    # Display the temperature on the heatmap
                    temp_text = f"{keypoint_name}: {temp_avg} C"
                    cv2.putText(heatmap, temp_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 添加 rprint 输出
                    emoji_map = {
                        'Nose': '👃',
                        'Left Eye': '👁️',
                        'Right Eye': '👁️',
                        'Mouth': '👄'
                    }
                    rprint(f"{emoji_map[keypoint_name]} {keypoint_name}温度 | Temperature: [bold red]{temp_avg}°C[/]")

                    current_time = time.time()
                    if current_time - last_save_time >= GAP:
                        log_data = {
                            "timestamp": current_time,
                            "temperature": temp_avg,
                            "temperature_matrix": temp_matrix
                        }
                        log_dir = "log"
                        if not os.path.exists(log_dir):
                            os.makedirs(log_dir)
                        log_file = os.path.join(log_dir, f"{keypoint_name}.json")

                        with open(log_file, "a+") as f:
                            f.write(json.dumps(log_data) + "\n")
                        last_save_time = current_time
                
        # 显示热图
        cv2.imshow('Thermal Face Landmarker', heatmap)
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()

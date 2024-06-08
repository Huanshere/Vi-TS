import sys
import time
import cv2
import numpy as np
import json
import os
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from config import (
    COUNTER, FPS, START_TIME, DETECTION_RESULT, CAMERA_ID, ROW_SIZE,
    LEFT_MARGIN, TEXT_COLOR, FONT_SIZE, FONT_THICKNESS, FPS_AVG_FRAME_COUNT,
    mp_face_mesh, mp_drawing,mp_drawing_styles, options, detector,
    get_landmark_temp
)

def save_result(result, unused_output_image, timestamp_ms):
    global FPS, COUNTER, START_TIME, DETECTION_RESULT
    if COUNTER % FPS_AVG_FRAME_COUNT == 0:
        FPS = FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
        START_TIME = time.time()
    DETECTION_RESULT = result
    COUNTER += 1

def run():
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

                # 获取面部温度
                temp, temp_matrix = get_landmark_temp(10, face_landmarks, heatmap, thdata)
                face_keypoint = { 10: 'Nose', 234: 'Left Eye', 454: 'Right Eye', 152: 'Mouth' }
                
        # 显示热图
        cv2.imshow('Thermal Face Landmarker', heatmap)
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()

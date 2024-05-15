#!/usr/bin/env python3
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from config import (
    COUNTER, FPS, START_TIME, DETECTION_RESULT, CAMERA_ID, WIDTH, HEIGHT,
    ROW_SIZE, LEFT_MARGIN, TEXT_COLOR, FONT_SIZE, FONT_THICKNESS,
    FPS_AVG_FRAME_COUNT, LABEL_BACKGROUND_COLOR, LABEL_PADDING_WIDTH,
    mp_face_mesh, mp_drawing, mp_drawing_styles, options, detector
)

# 添加颜色映射和对比度的变量
color_map = cv2.COLORMAP_JET
alpha = 1.0
beta = 0

def save_result(result, unused_output_image, timestamp_ms):
    global FPS, COUNTER, START_TIME, DETECTION_RESULT
    if COUNTER % FPS_AVG_FRAME_COUNT == 0:
        FPS = FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
        START_TIME = time.time()
    DETECTION_RESULT = result
    COUNTER += 1

def run():
    options.result_callback = save_result
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        print(f"Frame shape: {frame.shape}, data type: {frame.dtype}")
        imdata, thdata = np.array_split(frame, 2)
        print(f"imdata shape: {imdata.shape}, data type: {imdata.dtype}")
        
        bgr = cv2.convertScaleAbs(imdata, alpha=alpha, beta=beta)  # 直接使用imdata而不转换颜色空间
        # 计算调整后的宽度和高度,保持原始的宽高比
        original_height, original_width = imdata.shape[:2]
        aspect_ratio = original_width / original_height
        new_height = HEIGHT
        new_width = int(new_height * aspect_ratio)
        
        bgr = cv2.resize(bgr, (new_width, new_height))
        heatmap = cv2.applyColorMap(bgr, color_map)

        rgb_frame = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

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

                # 获取额头的位置 注意x y相反！
                forehead_x = int(face_landmarks[10].y * WIDTH)
                forehead_y = int(face_landmarks[10].x * HEIGHT)

                # 计算额头的温度
                hi = thdata[forehead_y][forehead_x][0]
                lo = thdata[forehead_y][forehead_x][1]
                lo = lo * 256
                rawtemp = hi + lo
                forehead_temp = (rawtemp / 64) - 273.15
                forehead_temp = round(forehead_temp, 2)

                # 在图像上显示额头温度
                cv2.putText(heatmap, f"Forehead Temp: {forehead_temp} C", (LEFT_MARGIN, ROW_SIZE * 2),
                            cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

                mp_drawing.draw_landmarks(
                    image=heatmap,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=heatmap,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=heatmap,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        cv2.imshow('face_landmarker', heatmap)
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
#!/usr/bin/env python3
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from config import (COUNTER,FPS,START_TIME,DETECTION_RESULT,CAMERA_ID,WIDTH,HEIGHT,
ROW_SIZE,LEFT_MARGIN,TEXT_COLOR,FONT_SIZE,FONT_THICKNESS,FPS_AVG_FRAME_COUNT,LABEL_BACKGROUND_COLOR,LABEL_PADDING_WIDTH,
mp_face_mesh,mp_drawing,mp_drawing_styles,options,detector)

def save_result(result, unused_output_image, timestamp_ms):
    global FPS, COUNTER, START_TIME, DETECTION_RESULT
    if COUNTER % FPS_AVG_FRAME_COUNT == 0:
        FPS = FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
        START_TIME = time.time()
    DETECTION_RESULT = result
    COUNTER += 1

def get_forehead_temperature(thdata, face_landmarks, frame_height, frame_width):
    # 获取额头节点的坐标
    forehead_landmark = face_landmarks[10]
    x, y = int(forehead_landmark.x * frame_width), int(forehead_landmark.y * frame_height)

    # 获取额头节点的温度
    hi = thdata[y // 2][x // 2][0]
    lo = thdata[y // 2][x // 2][1]
    lo = lo * 256
    rawtemp = hi + lo
    temp = (rawtemp / 64) - 273.15
    temp = round(temp, 2)

    return temp

def run():
    options.result_callback = save_result
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        imdata, thdata = np.array_split(frame, 2)

        rgb_frame = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        rgb_frame = cv2.flip(rgb_frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (LEFT_MARGIN, ROW_SIZE)
        cv2.putText(rgb_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        if DETECTION_RESULT:
            for face_landmarks in DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])
                
                # 获取额头节点的温度并在图像上显示
                forehead_temp = get_forehead_temperature(thdata, face_landmarks, frame_height, frame_width)
                temp_text = 'Forehead Temp: {:.1f}°C'.format(forehead_temp)
                temp_location = (LEFT_MARGIN, ROW_SIZE * 2)
                cv2.putText(rgb_frame, temp_text, temp_location, cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                
                mp_drawing.draw_landmarks(
                    image=rgb_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        cv2.imshow('face_landmarker', rgb_frame)
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from config import (
    COUNTER, FPS, START_TIME, DETECTION_RESULT, 
    CAMERA_ID, WIDTH, HEIGHT,
    ROW_SIZE, LEFT_MARGIN, TEXT_COLOR, FONT_SIZE, FONT_THICKNESS, 
    FPS_AVG_FRAME_COUNT, LABEL_BACKGROUND_COLOR, LABEL_PADDING_WIDTH,
    mp_face_mesh, mp_drawing, mp_drawing_styles,
    options, detector
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

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (LEFT_MARGIN, ROW_SIZE)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        if DETECTION_RESULT:
            for face_landmarks in DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in face_landmarks
                ])
                
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        current_frame = cv2.copyMakeBorder(
            current_frame, 0, 0, 0, LABEL_PADDING_WIDTH,
            cv2.BORDER_CONSTANT, None, LABEL_BACKGROUND_COLOR
        )

        if DETECTION_RESULT:
            legend_x = current_frame.shape[1] - LABEL_PADDING_WIDTH + 20
            legend_y = 30
            bar_max_width = LABEL_PADDING_WIDTH - 40
            bar_height = 8
            gap_between_bars = 5
            text_gap = 5

            face_blendshapes = DETECTION_RESULT.face_blendshapes

            if face_blendshapes:
                for category in face_blendshapes[0]:
                    category_name = category.category_name
                    score = round(category.score, 2)

                    text = f"{category_name} ({score:.2f})"
                    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                    cv2.putText(current_frame, text, (legend_x, legend_y + (bar_height // 2) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                    bar_width = int(bar_max_width * score)
                    cv2.rectangle(current_frame, (legend_x + text_width + text_gap, legend_y),
                                  (legend_x + text_width + text_gap + bar_width, legend_y + bar_height),
                                  (0, 255, 0), -1)

                    legend_y += (bar_height + gap_between_bars)

        cv2.imshow('face_landmarker', current_frame)

        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
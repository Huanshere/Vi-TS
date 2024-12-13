import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 模型参数
# 从 https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task 获得
MODEL = 'configs/face_landmarker.task'
NUM_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.75
MIN_FACE_PRESENCE_CONFIDENCE = 0.75
MIN_TRACKING_CONFIDENCE = 0.75

# 绘制参数
ROW_SIZE = 50
LEFT_MARGIN = 24
TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 1
FONT_THICKNESS = 1
FPS_AVG_FRAME_COUNT = 10

# MediaPipe face mesh parameters
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def placeholder_result_callback(result, unused_output_image, timestamp_ms):
    pass

# FaceLandmarker options and detector
base_options = python.BaseOptions(model_asset_path=MODEL)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=NUM_FACES,
    min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
    min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    # output_face_blendshapes=True,
    result_callback=placeholder_result_callback
)
detector = vision.FaceLandmarker.create_from_options(options)
    
def get_landmark_temp(landmark_id, face_landmarks, heatmap, thdata):
    # 获取landmark区域
    landmark = face_landmarks[landmark_id]
    HEIGHT, WIDTH, _ = heatmap.shape
    x, y = int(landmark.x * WIDTH), int(landmark.y * HEIGHT)

    temp_matrix = []
    for i in range(-1, 2):
        row = []
        for j in range(-1, 2):
            tx, ty = x + j, y + i
            # 如果 tx 或 ty 超出范围，用中心值替代
            if tx < 0 or tx >= WIDTH or ty < 0 or ty >= HEIGHT:
                temp = (thdata[y][x][0] + thdata[y][x][1] * 256) / 64 - 273.15
            else:
                temp = (thdata[ty][tx][0] + thdata[ty][tx][1] * 256) / 64 - 273.15
            temp = round(temp, 2)
            row.append(temp)
        temp_matrix.append(row)
    
    # 移除离群值并计算平均温度
    temps = [temp for row in temp_matrix for temp in row]
    temps.sort()
    if len(temps) > 2:
        temps = temps[1:-1]  # 去掉最大和最小值
    temp_avg = sum(temps) / len(temps)
    temp_avg = round(temp_avg, 2)
    
    # 在热图上绘制结果
    cv2.circle(heatmap, (x, y), 5, (255, 255, 255), -1)
    cv2.putText(heatmap, str(temp_avg) + ' C', (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return temp_avg, temp_matrix

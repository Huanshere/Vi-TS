import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global variables
COUNTER = 0
FPS = 0
START_TIME = 0
DETECTION_RESULT = None

# Model parameters
MODEL = 'face_landmarker.task'
NUM_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
CAMERA_ID = 0
WIDTH = 1280
HEIGHT = 960

# Drawing parameters
ROW_SIZE = 50
LEFT_MARGIN = 24
TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 1
FONT_THICKNESS = 1
FPS_AVG_FRAME_COUNT = 10
LABEL_BACKGROUND_COLOR = (255, 255, 255)
LABEL_PADDING_WIDTH = 1500

# MediaPipe face mesh parameters
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# FaceLandmarker options and detector
base_options = python.BaseOptions(model_asset_path=MODEL)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=NUM_FACES,
    min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
    min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    output_face_blendshapes=True,
    result_callback=None  # Will be set in face_detect.py
)
detector = vision.FaceLandmarker.create_from_options(options)
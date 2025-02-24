
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cam = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

screen_w, screen_h = pyautogui.size()

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]

def get_average_coords(landmarks, indices):
    x = sum([landmarks[idx].x for idx in indices]) / len(indices)
    y = sum([landmarks[idx].y for idx in indices]) / len(indices)
    return x, y

SMOOTHING_FACTOR = 0.9
prev_screen_x, prev_screen_y = screen_w / 2, screen_h / 2

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        left_iris_x, left_iris_y = get_average_coords(landmarks, LEFT_IRIS)
        right_iris_x, right_iris_y = get_average_coords(landmarks, RIGHT_IRIS)

        left_eye_x, left_eye_y = get_average_coords(landmarks, LEFT_EYE_INDEXES)
        right_eye_x, right_eye_y = get_average_coords(landmarks, RIGHT_EYE_INDEXES)

        eye_offset_x = ((left_iris_x - left_eye_x) + (right_iris_x - right_eye_x)) / 2
        eye_offset_y = ((left_iris_y - left_eye_y) + (right_iris_y - right_eye_y)) / 2

        norm_eye_offset_x = eye_offset_x * 250  
        norm_eye_offset_y = eye_offset_y * 250  

        screen_x = screen_w / 2 + norm_eye_offset_x * (screen_w / 2)
        screen_y = screen_h / 2 + norm_eye_offset_y * (screen_h / 2)

        screen_x = prev_screen_x * SMOOTHING_FACTOR + screen_x * (1 - SMOOTHING_FACTOR)
        screen_y = prev_screen_y * SMOOTHING_FACTOR + screen_y * (1 - SMOOTHING_FACTOR)

        prev_screen_x, prev_screen_y = screen_x, screen_y

        screen_x = max(0, min(screen_x, screen_w - 1))
        screen_y = max(0, min(screen_y, screen_h - 1))

        pyautogui.moveTo(screen_x, screen_y)

        for idx in LEFT_IRIS + RIGHT_IRIS:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        for idx in LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        left_eye_blink = [landmarks[145], landmarks[159]]
        if (left_eye_blink[0].y - left_eye_blink[1].y) < 0.004:
            pyautogui.click(button='RIGHT')

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()    
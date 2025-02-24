import av
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import torch
import torchvision.transforms.v2.functional as TF
import pyautogui


@dataclass(frozen=True)
class FM:
    """FaceMeshLandmarks
    import mediapipe.python.solutions.face_mesh_connections as fm
    https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
    """
    LEFT_IRIS_CENTER = [468]
    RIGHT_IRIS_CENTER = [473]
    CENTER = [6]
    LEFT_IRIS = [476, 475, 474, 477]
    RIGHT_IRIS = [471, 470, 469, 472]
    LEFT_EYE = [
        362,
        398,
        384,
        385,
        386,
        387,
        388,
        466,
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
    ]
    RIGHT_EYE = [
        33,
        246,
        161,
        160,
        159,
        158,
        157,
        173,
        133,
        155,
        154,
        153,
        145,
        144,
        163,
        7,
    ]

    @staticmethod
    def plot(draw, landmarks, color="red", width=0):
        points = landmarks.ravel().tolist() + landmarks[0].tolist()
        draw.line(points, fill=color, width=width)

    @staticmethod
    def point(draw, landmark, color="red", width=1):
        points = np.array([landmark - width, landmark + width])
        draw.ellipse(points.ravel().tolist(), fill=color)


st.set_page_config(page_title="Gaze", page_icon="👁️")

@st.cache_resource()
def load_gaze_model(pretrained='GazeTR-H-ETH.pt'):
    model = Model().eval()
    model.load_state_dict(torch.load(pretrained, 'cpu'))
    return model


def get_gaze(img):
    img = TF.resize(TF.center_crop(TF.to_image(img) / 255, min(img.size)), 224)[None]
    with torch.inference_mode():
        gaze = GazeTR({'face': img})[0].numpy()
    return gaze


@st.cache_resource()
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        static_image_mode=False,
    )


def get_landmarks(img, face_mesh=load_face_mesh()):
    if not isinstance(img, np.ndarray):
        img = np.asanyarray(img)
    landmarks = face_mesh.process(img).multi_face_landmarks
    if landmarks:
        landmarks = landmarks[0].landmark
        output = np.zeros((len(landmarks), 2))
        for i, landmark in enumerate(landmarks):
            output[i] = (landmark.x, landmark.y)
        height, width = img.shape[:2]
        output = np.floor(output * (width, height))
        return np.clip(output, 0, (width - 1, height - 1)).astype("uint16")
    return None


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_image()
    img = process(img.transpose(Image.FLIP_LEFT_RIGHT))
    return av.VideoFrame.from_image(img)


def process(img):
    landmarks = get_landmarks(img)
    if landmarks is not None:
        draw = ImageDraw.Draw(img)

        left = landmarks[FM.LEFT_IRIS_CENTER]
        left_center = landmarks[FM.LEFT_EYE].mean(0)

        right = landmarks[FM.RIGHT_IRIS_CENTER].mean(0)
        right_center = landmarks[FM.RIGHT_EYE].mean(0)

        left_center += (0, -3)
        right_center += (0, -3)

        FM.point(draw, left_center, 'red', 3)
        FM.point(draw, left, 'green', 3)
        FM.point(draw, right_center, 'red', 3)
        FM.point(draw, right, 'green', 3)

        center = ((left - left_center) + (right - right_center)) / 2

        FM.point(draw, landmarks[FM.CENTER] + center * (50, 100), 'orange', 5)


        img_w, img_h = img.size
        center_x, center_y = landmarks[FM.CENTER][0]
        center_offset_x, center_offset_y = center[0]
        screen_w, screen_h = pyautogui.size()
        new_mouse_x = int((center_x + center_offset_x * 50) * screen_w / img_w)
        new_mouse_y = int((center_y + center_offset_y * 150) * screen_h / img_h)

        pyautogui.moveTo(new_mouse_x, new_mouse_y)

        left_eye_blink = [landmarks[145], landmarks[159]]
        if (left_eye_blink[0][1] - left_eye_blink[1][1]) < 4:
         pyautogui.click(button='RIGHT')

    return img

def app():
    webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
    )


if __name__ == "__main__":
    app()

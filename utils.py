from collections import deque
import cv2
import numpy as np

## Author: https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/utils/cvfpscalc.py
class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded

def init_video_capture_device(cap_height, cap_width):
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    return webcam

def draw_fps_on_frame(frame, fps):
    cv2.putText(frame, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def preprocess_frame(frame, cvFpsCalc):
    frame = cv2.flip(frame, 1)
    fps = cvFpsCalc.get()
    frame = draw_fps_on_frame(frame, fps)
    return frame

def pressed_key(key):
    return cv2.waitKey(10) & 0xFF == ord(key)

def exist_landmarks(landmarks):
    return landmarks.multi_hand_landmarks is not None

def bounding_box_from_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_bounding_box(frame, bounding_box):
    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),
                    (0, 0, 0), 1)

    return frame

import cv2
import numpy as np
from utils import CvFpsCalc, init_video_capture_device, preprocess_frame, pressed_key

def get_hand_landmarks():
    return None

def capture_webcam(cap_width, cap_height):
    webcam = init_video_capture_device(cap_height=cap_height, cap_width=cap_width)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    while(True):
        _, frame = webcam.read()
        frame = preprocess_frame(frame, cvFpsCalc=cvFpsCalc)
        cv2.imshow('frame', frame)
        if pressed_key('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()



capture_webcam(600, 800)


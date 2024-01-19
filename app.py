import cv2
import numpy as np
from utils import CvFpsCalc, init_video_capture_device, preprocess_frame, pressed_key, bounding_box_from_landmarks, exist_landmarks, draw_bounding_box
import mediapipe as mp 
from configs import *

def init_hand_landmarks_model():
    mp_hands = mp.solutions.hands
    mp_hands_landmark_detector = mp_hands.Hands(
        static_image_mode= True,
        max_num_hands= num_hands,
        min_detection_confidence= min_detection_confidence,
        min_tracking_confidence= min_tracking_confidence,
    )
    return mp_hands_landmark_detector

def process_landmarks(frame, landmarks):
    for hand_landmark, handedness in zip(landmarks.multi_hand_landmarks, landmarks.multi_handedness):
        hand_bounding_box = bounding_box_from_landmarks(frame, hand_landmark)
        frame = draw_bounding_box(frame, hand_bounding_box)
        print(hand_bounding_box)
        cv2.imshow('frame', frame)

def capture_webcam(cap_width, cap_height):
    webcam = init_video_capture_device(cap_height=cap_height, cap_width=cap_width)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    mp_hands_landmark_detector = init_hand_landmarks_model()
    
    while(True):
        _, frame = webcam.read()
        frame = preprocess_frame(frame, cvFpsCalc=cvFpsCalc)
        landmarks = mp_hands_landmark_detector.process(frame)
        if exist_landmarks(landmarks):
            process_landmarks(frame, landmarks=landmarks)
        else:
            cv2.imshow('frame', frame)

        if pressed_key('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()



capture_webcam(600, 800)


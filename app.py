import cv2
import numpy as np
from utils import *
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
        list_of_landmarks = convert_landmarks_to_list(frame, hand_landmark)
        frame = draw_landmarks(frame, list_of_landmarks)
        cv2.imshow('frame', frame)

def save_landmarks_mode(frame, landmarks, key):
    for hand_landmark, handedness in zip(landmarks.multi_hand_landmarks, landmarks.multi_handedness):
        hand_bounding_box = bounding_box_from_landmarks(frame, hand_landmark)
        frame = draw_bounding_box(frame, hand_bounding_box)
        list_of_landmarks = convert_landmarks_to_list(frame, hand_landmark)
        frame = draw_landmarks(frame, list_of_landmarks)
        if is_number_key(key):
            save_land_mark(list_of_landmarks, int(chr(key & 0xFF)), landmark_csv_path)
        cv2.imshow('frame', frame)

def capture_webcam(cap_width, cap_height):
    webcam = init_video_capture_device(cap_height=cap_height, cap_width=cap_width)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    mp_hands_landmark_detector = init_hand_landmarks_model()
    global mode

    while(True):
        _, frame = webcam.read()
        input_key = cv2.waitKey(10)
        frame = preprocess_frame(frame, cvFpsCalc=cvFpsCalc)
        landmarks = mp_hands_landmark_detector.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if exist_landmarks(landmarks) and mode ==  "predict":
            process_landmarks(frame, landmarks=landmarks)
        elif exist_landmarks(landmarks) and mode == "save":
            save_landmarks_mode(frame, landmarks, input_key)
        else:
            cv2.imshow('frame', frame)
        
        if pressed_key(input_key, 'q'):
            break
        
        if pressed_key(input_key, 's'):
            mode = "save"
            print("save_mode")
            
        if pressed_key(input_key, 'p'):
            mode= "predict"

    webcam.release()
    cv2.destroyAllWindows()



capture_webcam(frame_width, frame_height)


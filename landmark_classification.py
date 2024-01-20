import tensorflow as tf
from utils import get_normalized_landmarks
from configs import model_save_path, labels_path
import numpy as np
import csv

class HandLandmarkClassificator():
    """
    Classificator with trained weights.
    """
    def __init__(self):
        self.model = tf.keras.models.load_model(model_save_path)
        labels = []
        with open(labels_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:
                labels.append(line)
        self.labels = labels

    def classify(self, landmarks):
        landmarks = [elemento for sublista in landmarks for elemento in sublista]
        landmarks,_ = get_normalized_landmarks(landmarks, csv_read=False)
        predict_result = self.model.predict(np.array([landmarks]))
        predict_result = np.argmax(np.squeeze(predict_result))
        return self.labels[predict_result][0]
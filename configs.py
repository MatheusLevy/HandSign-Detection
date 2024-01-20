# Model Configs
use_static_image_mode= True
min_detection_confidence= 0.1
min_tracking_confidence= 0.2
num_hands = 1

# Video Configs
frame_width= 960
frame_height= 540

# Save Configs
landmark_csv_path = r"./landmarks_save/landmarks.csv"
model_save_path= r"./landmark_classification_model/handlandmark_classification_model.hdf5"
labels_path = r"./landmark_classification_model\labels.csv"
# App configs
mode = "predict"
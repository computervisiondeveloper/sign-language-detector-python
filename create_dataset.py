import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define data directory
DATA_DIR = './data'

# Initialize lists for data and labels
data = []
labels = []

# Function to process images
def process_image(img_path):
    data_aux = []
    x_ = []
    y_ = []

    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        return data_aux, True
    return None, False

# Process each image in the data directory
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue
    
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        processed_data, success = process_image(img_path)

        if success:
            data.append(processed_data)
            labels.append(dir_)

# Save data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Release resources
hands.close()

print("Data processing completed and saved to 'data.pickle'.")

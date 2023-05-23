import os
import cv2
import numpy as np
import mediapipe as mp


def get_hand_keypoints_2d(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y])
        return np.array(keypoints).flatten()
    else:
        return None


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# saving keypoints extracted from images
data_dir = "./asl_images"
output_dir = "./keypoints2d"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for folder in os.listdir(data_dir):
    print(folder)
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            keypoints = get_hand_keypoints_2d(image_path)
            if keypoints is not None:
                output_file = os.path.join(output_dir, f"{folder}_{image_name.split('.')[0]}.npy")
                np.save(output_file, keypoints)

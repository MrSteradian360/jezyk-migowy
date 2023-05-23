import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# camera set
cap = cv2.VideoCapture(0)


# mediapipe hands 
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    model_path = 'trained_model2d/model.h5'
    model = tf.keras.models.load_model(model_path)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Nie udalo sie zdobyc obrazu.")
            continue

        # colour conversion form BGR into RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # hand and keypoints detection
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # keypoint vector
                hand_points = []
                for point in hand_landmarks.landmark:
                    hand_points.append(point.x)
                    hand_points.append(point.y)

                hand_points = np.array(hand_points)

                # keypoint vector normalization
                hand_points = hand_points / np.max(hand_points)

                # reshape keypoint vector
                hand_points = hand_points.astype(np.float32)
                hand_points = hand_points.reshape((1, 42))

                # letter or digit recognition using trained model
                prediction = model.predict(hand_points)

                # get predicted label
                if np.argmax(prediction) < 10:
                    prediction = chr(np.argmax(prediction) + 48)
                else:
                    prediction = chr(np.argmax(prediction) + 65 - 10)

                # displaying recognized letter or char
                cv2.putText(image, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # colour conversion from BGR into RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # displaying image with added keypoints and recognized letter or digit
        cv2.imshow('MediaPipe Hands', image)

        # quit the loop by pressing 'q' keyboard
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

cap.release()




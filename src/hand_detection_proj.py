import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Ustawienie kamerki
cap = cv2.VideoCapture(0)


# Ustawienia mediapipe
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    # # Tworzenie modelu sieci neuronowej
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(128, input_shape=(63,), activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(26, activation='softmax')
    # ])

    # # Kompilacja modelu
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # # Wypisanie struktury modelu
    # model.summary()

    model_path = 'trained_model2'
    model = tf.keras.models.load_model(model_path)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Nie udalo sie zdobyc obrazu.")
            continue

        # Konwersja koloru z BGR na RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Wykrywanie ręki i keypointów
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Tworzenie wektora z keypointami
                hand_points = []
                for point in hand_landmarks.landmark:
                    hand_points.append(point.x)
                    hand_points.append(point.y)
                    hand_points.append(point.z)
                hand_points = np.array(hand_points)

                # Normalizacja wektora keypointów
                hand_points = hand_points / np.max(hand_points)

                # Reshape wektora keypointów do wymiarów akceptowanych przez sieć
                hand_points = hand_points.reshape(-1, 63)

                # Rozpoznawanie litery z wykorzystaniem sieci neuronowej
                prediction = model.predict(hand_points)
                letter = chr(np.argmax(prediction) + 65)

                # Wyświetlanie wykrytej litery
                cv2.putText(image, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Konwersja koloru z BGR na RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Wyświetlanie obrazu z narysowanymi keypointami i wykrytą literą
        cv2.imshow('MediaPipe Hands', image)

        # Wyjście z pętli po naciśnięciu klawisza 'q'
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()



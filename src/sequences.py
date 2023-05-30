from pathlib import Path
from keras.utils import Sequence
import numpy as np
import os
import os
from typing import Iterable
import cv2
import mediapipe as mp
import re
import seaborn as sns
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils


class KeypointsVectorToLetters(Sequence):
    def __init__(self, paths: Iterable[Path], batch_size: int, labels: list[str] = None) -> None:
        """
        Args:
            paths: paths to the folders containg images of signs. Each image name should start with the letter it represents.
            batch_size: size of the batch
        """
        super().__init__()
        self.hands = hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        self.paths = [f for p in paths for f in p.iterdir()
                      if f.suffix == ".jpg"]
        self.batch_size = batch_size

        labels = labels or list({self._extract_label(f) for f in self.paths})
        labels = sorted(labels)
        self.labels_map = {
            label: i for i, label in enumerate(labels)
        }

    def __getitem__(self, index):
        """Gets batch at position `index`.

        Args:
            index: position of the batch in the Sequence.

        Returns:
            A batch
        """
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        paths = self.paths[start_idx:end_idx]
        return np.array([self._extract_keypoints(f) for f in paths]), \
            np.array([self.labels_map[self._extract_label(f)] for f in paths])

    def _extract_label(self, filename: Path):
        first_digit = re.search('\d', filename.stem).start()
        return filename.stem[:first_digit]

    def _extract_keypoints(self, filename: Path):
        keypoints_filename = filename.with_stem(f'{filename.stem}_keypoints')
        if not keypoints_filename.exists():
            image = self._create_keypoints_image(filename)
            cv2.imwrite(keypoints_filename, image)
        keypoints = cv2.imread(str(keypoints_filename))
        keypoints = cv2.cvtColor(keypoints, cv2.COLOR_BGR2GRAY)
        return keypoints

    
    def _create_keypoints_image(self, filename: Path):
        image = cv2.imread(str(filename))
        assert image is not None, f"Image should not be none: {filename}"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            keypoints = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y])
            return np.array(keypoints).flatten()
        else:
            return np.zeros(2 * len(hands.HandLandmark)) * -1


    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        return len(self.paths) // self.batch_size


class KeypointsImageToLetters(KeypointsVectorToLetters):

    def __init__(self, paths: Iterable[Path], batch_size: int, labels: list[str] = None) -> None:
        super(KeypointsImageToLetters, self).__init__(paths, batch_size, labels)
        pallet = sns.color_palette("hls", len(hands.HandLandmark))
        self._landmark_colors = {
            l: c for l, c in zip(hands.HandLandmark, pallet)
        }

    def _extract_keypoints(self, filename: Path):
        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints_image = np.zeros((image.shape[0], image.shape[1], 3))
        results = self.hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    keypoints_image, hand_landmarks, hands.HAND_CONNECTIONS,
                    {l: drawing.DrawingSpec(
                        circle_radius=5, color=self._landmark_colors[l]) for l in hands.HandLandmark}
                )
        return keypoints_image
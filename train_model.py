# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from pathlib import Path
from src.sequences import KeypointsVectorToLetters

# %%
wandb.init(project="sign_language_recognition")

try:
    train_dir = Path('data/asl_alphabet_train')
    train_seq = KeypointsVectorToLetters(train_dir.iterdir(), batch_size=32, labels=[
                                         d.stem for d in train_dir.iterdir()])

    num_classes = len(train_seq.labels_map)
    input_dim = 21 * 3

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=['accuracy'])

    epochs = 25
    callbacks = []
    if wandb.run:
        callbacks.append(WandbCallback())

    model.fit(train_seq, epochs=epochs, callbacks=callbacks)
finally:
    wandb.finish()

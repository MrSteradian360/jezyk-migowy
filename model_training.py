import os
import numpy as np
import tensorflow as tf

# read keypoints data
keypoints_dir = "./keypoints2d"
keypoints_data = []
keypoints_labels = []

for file_name in os.listdir(keypoints_dir):
    print(file_name)
    file_path = os.path.join(keypoints_dir, file_name)
    keypoints = np.load(file_path)
    label = file_name[0]  # Get the first letter of the file name as the label
    if label in '0123456789':
        label = ord(label) - 48
    else:       # letter
        label = ord(label) - 65 + 10

    keypoints_data.append(keypoints)
    keypoints_labels.append(label)

# data into numpy array
keypoints_data = np.stack(keypoints_data)
keypoints_data = keypoints_data.astype(np.float32)

# labels into numpy array in ascii code
keypoints_labels = np.array(keypoints_labels)

# all labels and number of them
unique_labels = list(set(keypoints_labels))
num_classes = len(unique_labels)

# define neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(21 * 2)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

# train model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

epochs = 25
history = model.fit(keypoints_data, keypoints_labels, epochs = epochs)

# save trained model
model.save('trained_model2d/model.h5')
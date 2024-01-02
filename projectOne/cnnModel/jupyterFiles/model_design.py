"""
Authors:
- bardiaf - bardiaf@student.chalmers.se
- ...

Usage: model prototype version 1 - used in cnnModel/models.py
"""

import keras
import os
import cv2
import imghdr
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_directory = '/Users/bardiaforooraghi/Downloads/data'
image_exist = ['jpeg', 'jpg', 'png']
data = tf.keras.utils.image_dataset_from_directory('/Users/bardiaforooraghi/Downloads/data')

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()
data = data.map(lambda x, y: (x / 255, y))


train_size = int(len(data) * .7)
validation_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1


train_data = data.take(train_size)
validation_data = data.skip(train_size).take(validation_size)
test_data = data.skip(train_size + validation_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D((2, 2)))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer
model.add(Dense(3, activation='softmax'))  # Output layer with 3 neurons (0, 1, 2)

    
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


log_directory = '/Users/bardiaforooraghi/Downloads/logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory)

history = model.fit(train_data, epochs=50, batch_size=100, validation_data=validation_data, callbacks=[tensorboard_callback])

img = cv2.imread('/Users/bardiaforooraghi/Downloads/unseen data/malignant/malignant193Mask.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))


yhat = model.predict(np.expand_dims(resize / 255, 0))


predicted_class = np.argmax(yhat)


if predicted_class == 0:
    print('benign')
elif predicted_class == 1:
    print('malignant')
else:
    print('normal')

image = cv2.imread('/Users/bardiaforooraghi/Downloads/Dataset_BUSI_with_GT/malignant/malignant (1).png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary mask
_, binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Invert the binary mask
binary_mask = cv2.bitwise_not(binary_mask)

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Plot the inverted binary mask
plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Inverted Binary Mask')
plt.axis('off')

# Display the plots
plt.show()
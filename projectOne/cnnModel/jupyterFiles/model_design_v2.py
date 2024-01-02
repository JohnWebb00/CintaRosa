#!/usr/bin/env python
# coding: utf-8

# In[417]:

"""
Authors:
- bardiaf - bardiaf@student.chalmers.se

Usage: model prototype version 2
"""


import keras
import os
import cv2
import csv
import imghdr
import pathlib
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[418]:


data_directory = '/Users/bardiaforooraghi/Downloads/data'


# In[419]:


image_exist = ['jpeg', 'jpg', 'png']


# In[420]:


data = tf.keras.utils.image_dataset_from_directory('/Users/bardiaforooraghi/Downloads/data')


# In[421]:


data_iterator = data.as_numpy_iterator()


# In[422]:


batch = data_iterator.next()


# In[423]:


data = data.map(lambda x, y: (x / 255, y))


# In[424]:


train_size = int(len(data) * .7)
validation_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1


# In[425]:


train_data = data.take(train_size)
validation_data = data.skip(train_size).take(validation_size)
test_data = data.skip(train_size + validation_size).take(test_size)


# In[492]:


model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 1)))
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


# In[427]:


log_directory = '/Users/bardiaforooraghi/Downloads/logs'


# In[428]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory)


# In[429]:


history = model.fit(train_data, epochs=50, batch_size=100, validation_data=validation_data, callbacks=[tensorboard_callback])


# In[430]:


yhat = model.predict(np.expand_dims(resize / 255, 0))


# In[431]:


predicted_class = np.argmax(yhat)

if predicted_class == 0:
    print('benign')
elif predicted_class == 1:
    print('malignant')
else:
    print('normal')


# In[489]:


def to_numpy(image_path):
    
    scale_factor = 1 / 255.0
    
    # It ensures image_path is a string
    if not isinstance(image_path, str):
        raise ValueError("Image path should be a string.")

    # It reads the image and converts it to grayscale
    img = Image.open(image_path).convert('L')
            
    # It checks if the image is successfully read
    if img is None:
         raise FileNotFoundError(f"Image not found at path: {image_path}")

    # It resizes the image using to (256, 256)
    resized_img = img.resize((256, 256))
    
    # It converts the resized image to NumPy array
    img_array = np.array(resized_img, dtype = 'float32') * scale_factor

    return img_array


# In[490]:


def images_to_csv(input_directory, output_csv_path):
    
    write_header = not os.path.exists(output_csv_path)
    # Open the CSV file in append mode
    with open(output_csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        if write_header:
            header_row = [f'P-{i}' for i in range(256 * 256)]
            csv_writer.writerow(header_row)

        # Process each image in the input directory
        for filename in os.listdir(input_directory):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                image_path = join(input_directory, filename)

                img_array = to_numpy(image_path)

                # Reshape the 2D array into a 1D array
                flattened_array = img_array.flatten()
                
                if "malignant" in filename:
                    label = 1
                elif "benign" in filename:
                    label = 0
                elif "normal" in filename:
                    label = 2
                else:
                    label = -1  # Placeholder for other labels

                 # Write it to the CSV file
                #csv_writer.writerow(flattened_array)
                csv_writer.writerow([label] + list(flattened_array))

    print(f"Images in '{input_directory}' have been converted and appended to the CSV file '{output_csv_path}'.")


# In[491]:


malignant_directory = "/Users/bardiaforooraghi/Downloads/data/normal/"
csv_file_path = '/Users/bardiaforooraghi/Downloads/images.csv'

images_to_csv(malignant_directory, csv_file_path)


# In[474]:


fashion_test_df = pd.read_csv('/Users/bardiaforooraghi/Downloads/images.csv', sep = ',')

formatted_df = fashion_test_df.head().to_string(index=True, header=True)

print(formatted_df)


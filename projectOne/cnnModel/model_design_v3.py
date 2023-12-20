#!/usr/bin/env python
# coding: utf-8

# In[45]:


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
from shutil import copyfile
from os.path import join
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[46]:


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


# In[47]:


def images_to_csv(input_directory, output_csv_path):
    write_header = not os.path.exists(output_csv_path)
    
    # Open the CSV file in append mode
    with open(output_csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        if write_header:
            # Include the label in the header
            header_row = ['Label'] + [f'P-{i}' for i in range(256 * 256)]
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

                # Write label and pixel data to the CSV file
                csv_writer.writerow([label] + list(flattened_array))

    print(f"Images in '{input_directory}' have been converted and appended to the CSV file '{output_csv_path}'.")


# In[54]:


def prepare_image_data(train_path, test_path, image_shape=(256, 256, 1), test_size=0.2, random_state=12345):

    # Read datasets
    train_dataset = pd.read_csv(train_path, sep=',')
    test_dataset = pd.read_csv(test_path, sep=',')

    # Create training and testing arrays
    training = np.array(train_dataset, dtype='float32')
    testing = np.array(test_dataset, dtype='float32')
    
    # Prepare the training and testing dataset 
    X_train = training[:,1:]
    y_train = training[:,0]

    X_test = testing[:,1:]
    y_test = testing[:,0]

    # Prepare the training and testing dataset    
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
     
    # Reshape the data in a form that CNN can accept
    X_train = X_train.reshape(X_train.shape[0], *image_shape)
    X_test = X_test.reshape(X_test.shape[0], *image_shape)
    X_validate = X_validate.reshape(X_validate.shape[0], *image_shape)

    
    return X_train, X_validate, X_test, y_train, y_validate, y_test


# In[55]:


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


# In[56]:


def train_model(model, X_train, y_train, X_validate, y_validate):
    model.fit(X_train, y_train, batch_size=2, epochs=10)
    model.fit(X_validate, y_validate, batch_size=2, epochs=10)


# In[59]:


train_path = '/Users/bardiaforooraghi/Downloads/train_dataset.csv'
test_path = '/Users/bardiaforooraghi/Downloads/test_dataset.csv'
X_train, X_validate, X_test, y_train, y_validate, y_test = prepare_image_data(train_path, test_path)


# In[60]:


train_model(model, X_train, y_train, X_validate, y_validate)


# In[53]:


images_to_csv('/Users/bardiaforooraghi/Downloads/unseen data/normal', '/Users/bardiaforooraghi/Downloads/test_dataset.csv')


# In[ ]:





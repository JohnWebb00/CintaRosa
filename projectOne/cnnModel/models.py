from django.db import models
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import os
import keras 
import matplotlib 
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix
from os.path import join
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import shutil
from django.conf import settings

# enable heatmap plotting by running matplot on separate thread instead of main
matplotlib.use('agg')

MODEL_VERSION = 'breastCancerModel_v'
MODEL_VERSION_INT = 1

## CNNMODEL = attrs: file, accuracy, heatmap, deployed (TRUE FALSE)
# CNNMODEL.object.filter(deployed=True) --> the current deployed model --> MODEL_VERSION trimmed file path

# Create your models here.
class BreastCancerModelDetection(models.Model):

    model = None # Class variable to store the model
    history_train = None # Class variable to store the training history
    history_validate = None # Class variable to store the valdating history
    history_test = None  # Class variable to store the test history
    versionInt = MODEL_VERSION_INT
    versionNum = MODEL_VERSION
    
    @staticmethod
    def checkCurrVersion():
        tempItr = 1
        while(True):
            # check if file with current version number exists
            file = f"cnnModel/models/{BreastCancerModelDetection.versionNum}{tempItr}.h5"
            
            # if f"{tempItr}" in filename:
            if os.path.exists(file):
                tempItr = tempItr + 1
            else:
                break
        return tempItr

    @staticmethod
    def to_numpy(image_path):
        # if the image is an image, dont need to pass string
        img1 = image_path
        
        # It ensures image_path is a string
        if isinstance(image_path, str):
            #raise ValueError("Image path should be a string.")
            # It reads the image from path
            img1 = cv2.imread(image_path)
        # if the instance is not a path but an actual image
        
        # Turns img to grayscale
        img2 = tf.image.rgb_to_grayscale(img1)
        resized = tf.image.resize(img2, (256, 256))
        final = np.expand_dims((resized / 255), 0)

        return final


    @staticmethod
    def images_to_csv(input_directory, output_csv_path):
        write_header = not os.path.exists(output_csv_path)

        # Open the CSV file in append mode
        with open(output_csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            if write_header:
                header_row = ['Label'] + [f'P-{i}' for i in range(256 * 256)]
                csv_writer.writerow(header_row)

            # Process each image in the input directory
            for filename in os.listdir(input_directory):
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    image_path = join(input_directory, filename)

                    img_array = BreastCancerModelDetection.to_numpy(image_path)

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
                    csv_writer.writerow([label] + list(flattened_array))

        print(f"Images in '{input_directory}' have been converted and appended to the CSV file '{output_csv_path}'.")


    @staticmethod
    def prepare_image_data(dataPath):

        # Read datasets
        data = tf.keras.utils.image_dataset_from_directory(dataPath, label_mode='int', color_mode="grayscale")
        data = data.map(lambda x, y: (x / 255, y))

        # calculate train, validation, test size via 70/20/10 split
        train_size = int(len(data) * .7)
        validation_size = int(len(data) * .2) + 1
        test_size = int(len(data) * .1) + 1

        # Take portion of data from total dataset
        train_data = data.take(train_size)
        validation_data = data.skip(train_size).take(validation_size)
        test_data = data.skip(train_size + validation_size).take(test_size)
        
        return train_data, validation_data, test_data


    @staticmethod
    def createModel():
        # Initializing the model 
        BreastCancerModelDetection.model = Sequential()

        # Convolutional layers
        ## SHould be this dont change
        BreastCancerModelDetection.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
        BreastCancerModelDetection.model.add(MaxPooling2D((2, 2)))
        BreastCancerModelDetection.model.add(Dropout(0.25))  # Dropout layer

        BreastCancerModelDetection.model.add(Conv2D(64, (3, 3), activation='relu'))
        BreastCancerModelDetection.model.add(MaxPooling2D((2, 2)))
        BreastCancerModelDetection.model.add(Dropout(0.25))  # Dropout layer

        BreastCancerModelDetection.model.add(Conv2D(128, (3, 3), activation='relu'))
        BreastCancerModelDetection.model.add(MaxPooling2D((2, 2)))
        BreastCancerModelDetection.model.add(Dropout(0.25))  # Dropout layer
        
        BreastCancerModelDetection.model.add(Conv2D(256, (3, 3), activation='relu'))
        BreastCancerModelDetection.model.add(MaxPooling2D((2, 2)))

        # Flatten layer
        BreastCancerModelDetection.model.add(Flatten())

        # Dense (fully connected) layers
        BreastCancerModelDetection.model.add(Dense(512, activation='relu'))
        BreastCancerModelDetection.model.add(Dropout(0.5))  # Dropout layer
        BreastCancerModelDetection.model.add(Dense(256, activation='relu'))
        BreastCancerModelDetection.model.add(Dropout(0.5))  # Dropout layer
        BreastCancerModelDetection.model.add(Dense(3, activation='softmax'))

        # Complie the model by defineing the cost function and the optimizer for the model as well the used metrics to measure the qulaity of the model 
        BreastCancerModelDetection.model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        path = os.path.join("models", f"{BreastCancerModelDetection.versionNum}{BreastCancerModelDetection.versionInt}.h5")
        
        if(os.path.isfile(path)):
            finalPath = os.path.join("cnnModel", path)
            
            BreastCancerModelDetection.model.load_weights(finalPath)

    # method for traing the model
    @staticmethod
    def train_breast_cancer_model_detection():
        if BreastCancerModelDetection.model is None:
            # creates model but doesnt load weights as none exist yet
            BreastCancerModelDetection.createModel()
        
        print("Model retraining started in backend")
        
        data_OSPath = os.path.join("cnnModel", "kaggle_image_data")
        dataPath = str(data_OSPath)
     
        # Create train, validation and test sets from the image data stored locally
        train_data, validation_data, test_data = BreastCancerModelDetection.prepare_image_data(dataPath)
        
        # Define batch sizes and epochs
        train_batch_size = 10
        train_epochs = 12
        
        # Create TensorBoard callback for logging
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        
        # Train on the training and validation set
        BreastCancerModelDetection.history_train = BreastCancerModelDetection.model.fit(train_data, batch_size=train_batch_size, epochs=train_epochs, callbacks=[tensorboard_callback])
        BreastCancerModelDetection.history_validate = BreastCancerModelDetection.model.fit(validation_data, batch_size=train_batch_size, epochs=train_epochs, callbacks=[tensorboard_callback])

        # evaluate the model
        BreastCancerModelDetection.history_test = BreastCancerModelDetection.model.evaluate(test_data)
        
        # Checks current version then returns with new version
        new_version = BreastCancerModelDetection.checkCurrVersion()
        
        path = os.path.join("models", f"{BreastCancerModelDetection.versionNum}{new_version}.h5")
        finalPath = os.path.join("cnnModel", path)
        
        # save weights to folder
        BreastCancerModelDetection.model.save(finalPath)
        
        # Find labels of test data for confusion matrix creation
        y_test = np.concatenate([y for x, y in test_data], axis=0)
        
        # create predicted classes which represents a set of predictions and their labels
        predicted_classes = BreastCancerModelDetection.model.predict(test_data, verbose='auto', batch_size=train_batch_size) #length 25, due to only needing to predict 25 images
        predicted_classes = np.argmax(predicted_classes, axis=1)
        
        # Create confusion matrix based on the predicted classes and their actual values
        conf_matrix = confusion_matrix(y_test, predicted_classes)
        
        # Plot heat map based on the confusion matrix
        heatmap = sns.heatmap(conf_matrix, annot=True)
        fig = heatmap.get_figure()
        
        # Create heatmap folder
        os.makedirs(os.path.join("cnnModel", "heatMap"), exist_ok=True)
        
        # Save heatmap to folder
        heatMapName = os.path.join("heatMap", f"{BreastCancerModelDetection.versionNum}{new_version}.png")
        finalPathHeatmap = os.path.join("cnnModel", heatMapName)
        fig.savefig(finalPathHeatmap) 
        shutil.move(finalPathHeatmap, settings.MEDIA_ROOT)
                
        # Create Model Image (only for backend NOT to be returned)
        plot_name = os.path.join("model_plot", f'model_{new_version}.png')
        dot_img_file = os.path.join("cnnModel", plot_name)
        tf.keras.utils.plot_model(BreastCancerModelDetection.model, to_file=dot_img_file, show_shapes=True)
        
        test_acc = BreastCancerModelDetection.history_test.pop()
        train_acc = BreastCancerModelDetection.history_train.history['accuracy'].pop()
        val_acc = BreastCancerModelDetection.history_validate.history['accuracy'].pop()
    
        # Returns final validation accuracy to save as part of model as well as path to weights and heatmap
        return train_acc, test_acc, val_acc, finalPath, f"{BreastCancerModelDetection.versionNum}{new_version}.png"

        
    @staticmethod  
    def predict(pathOfImg):
        #### Extract data before passing to model ####
        if BreastCancerModelDetection.model is None:
            raise ValueError("Model has not been initialized.")
        
        else:
            try:
                #turn img to np.array
                np_predict = BreastCancerModelDetection.to_numpy(pathOfImg)
                # predict via the np array
                
                prediction_arr = BreastCancerModelDetection.model.predict(np_predict)
               
                predicted_class = np.argmax(prediction_arr)
                
                # returns int (0 == benign, 1 == malignant, 2 == normal) and explainable AI img
                return predicted_class
            except FileNotFoundError:
                print(f"File not found at {pathOfImg}")
                


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
from skimage.segmentation import mark_boundaries

# Explainable AI
from lime import lime_image
from PIL import Image 
from matplotlib import cm

# enable heatmap plotting by running matplot on separate thread instead of main
matplotlib.use('agg')

MODEL_VERSION = 'breastCancerModel_v'
MODEL_VERSION_INT = 1

## CNNMODEL = attrs: file, accuracy, heatmap, deployed (TRUE FALSE)
# CNNMODEL.object.filter(deployed=True) --> the current deployed model --> MODEL_VERSION trimmed file path

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
        # img2 = tf.image.rgb_to_grayscale(img1)
        resized = tf.image.resize(img1, (256, 256)) 
        final = np.expand_dims((resized / 255), 0)

        return final


    @staticmethod
    def prepare_image_data(dataPath):

        # Read datasets
        data = tf.keras.utils.image_dataset_from_directory(dataPath, label_mode='int') #, color_mode="grayscale")
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
        BreastCancerModelDetection.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
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
        
        curr_batch_size = 16
        train_epochs = 10
        # Create TensorBoard callback for logging
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        
        # Train on the training and validation set
        BreastCancerModelDetection.history_train = BreastCancerModelDetection.model.fit(train_data, batch_size=curr_batch_size, epochs=train_epochs, callbacks=[tensorboard_callback])
        BreastCancerModelDetection.history_validate = BreastCancerModelDetection.model.fit(validation_data, batch_size=curr_batch_size, epochs=train_epochs, callbacks=[tensorboard_callback])

        # evaluate the model
        BreastCancerModelDetection.history_test = BreastCancerModelDetection.model.evaluate(test_data)
        
        # Checks current version then returns with new version
        new_version = BreastCancerModelDetection.checkCurrVersion()
        
        path = os.path.join("models", f"{BreastCancerModelDetection.versionNum}{new_version}.h5")
        finalPath = os.path.join("cnnModel", path)
        
        # save weights to folder
        BreastCancerModelDetection.model.save(finalPath)
        
        # Find labels of validation data for confusion matrix creation
        y_labels = np.concatenate([y for x, y in test_data], axis=0)
    
        # Find predicted classes to compare with actual labels
        predictions = BreastCancerModelDetection.model.predict(test_data)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Create confusion matrix based on the predicted classes and their actual values
        conf_matrix = confusion_matrix(y_labels, pred_classes)
        
        # Plot heat map based on the confusion matrix
        heatmap = sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=['Benign', 'Malignant', 'Normal'], yticklabels=['Benign', 'Malignant', 'Normal'])
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
        
        os.makedirs(os.path.join("cnnModel", "model_plot"), exist_ok=True)
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
               
                # Get the predicted class (via the index at which the max value is) from the prediction (meaning the final class predicted)
                predicted_class = np.argmax(prediction_arr)
                print(prediction_arr)
                
                # Create lime explainer
                explainer = lime_image.LimeImageExplainer(random_state=42)
                
                # Resize image and turn it into a numpy array
                image = tf.keras.preprocessing.image.load_img(pathOfImg, target_size=(256, 256))
                input_arr = tf.keras.preprocessing.image.img_to_array(image)
                
                # Exlpain instance via predict image
                explanation = explainer.explain_instance(
                                input_arr, 
                                BreastCancerModelDetection.model.predict)

                # Return original image and mask from the predicited class
                image, mask = explanation.get_image_and_mask(predicted_class, 
                                                            hide_rest=False)
                
                # Normalize image, convert to range between 0, 255 then turn into an integer then into a PIL image
                # mark_boundaries just allows us to see where the model has marked its prediction
                finalImage = Image.fromarray((mark_boundaries(image, mask) * 255).astype(np.uint8))
                
                # returns int (0 == benign, 1 == malignant, 2 == normal) and explainable AI img and mask for explanation
                return predicted_class, finalImage, prediction_arr
            except FileNotFoundError:
                print(f"File not found at {pathOfImg}")
                


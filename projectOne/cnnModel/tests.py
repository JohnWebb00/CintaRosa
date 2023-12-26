import unittest
from models import BreastCancerModelDetection 
from groupOneApp.models import ML_model 
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import keras
from projectOne.groupOneApp.models import Prediction

class modelTests(unittest.TestCase):
    def test_createModel(self):
        # Call method to create the model
        BreastCancerModelDetection.createModel()

        # Assert that the model is an instance of a Sequential model
        self.assertIsInstance(BreastCancerModelDetection.model, Sequential)

        # Create list of expected layers
        expectedLayers = [
            Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), 1, activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), 1, activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='sigmoid')
        ]
        #Assert that number of actual layers matches the number of expected layers
        modelLayers = BreastCancerModelDetection.model.layers
        self.assertEqual(len(modelLayers), len(expectedLayers))

        for modelLayer, expectedLayer in zip(modelLayers, expectedLayers):
            self.assertIsInstance(modelLayer, type(expectedLayer))
        
        expectedOptimizer = keras.optimizers.Adam(0.001)
        expectedLoss = 'sparse_categorical_crossentropy'
        expectedMetrics = ['accuracy']

        # Assert that the model is contains the expected optimizer, loss, and metric parameters
        self.assertEqual(BreastCancerModelDetection.model.optimizer.get_config(), expectedOptimizer.get_config())
        self.assertEqual(BreastCancerModelDetection.model.loss, expectedLoss)
        self.assertEqual(BreastCancerModelDetection.model.metrics_names, expectedMetrics)
        
        path = os.path.join("models", f"{BreastCancerModelDetection.versionNum}{BreastCancerModelDetection.versionInt}.h5")

        if os.path.exists(path):
            os.remove(path)
            
    def test_prepare_image_data_not_none(self):
        #Assign data path to use as input for the prepare_image_data function
        data_OSPath = os.path.join("cnnModel", "kaggle_image_data")
        dataPath = str(data_OSPath)
        
        #Call prepare_image_data function and assign returned values to variables
        train_data, validation_data, test_data = BreastCancerModelDetection.prepare_image_data(dataPath)
        
        #Assert that returned data is present
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(validation_data)
        self.assertIsNotNone(test_data)
        
            
    def test_train_breast_cancer_model_returns_all_values(self):
        # Can function and assign returned values to variables 
        train_acc, test_acc, val_acc, finalPath, heatmapPath = BreastCancerModelDetection.train_breast_cancer_model_detection()

        # Assert expected return types and paths
        self.assertTrue(isinstance(train_acc, float))  
        self.assertTrue(isinstance(test_acc, float))  
        self.assertTrue(isinstance(val_acc, float)) 
        self.assertTrue(finalPath.startswith("cnnModel/models/"))  
        self.assertTrue(heatmapPath.startswith("cnnModel/heatMap/"))
        
    def test_train_breast_cancer_model_no_intialized_model(self):
        #Save current model
        currentModel = BreastCancerModelDetection.model
        
        #Set model to None so that no model is selected
        BreastCancerModelDetection.model = None

        # Assert that a ValueError is raised with the corresponding message
        with self.assertRaises(ValueError) as context:
            BreastCancerModelDetection.train_breast_cancer_model_detection()

        self.assertIn("Model has not been initialized.", str(context.exception))
        
        #Set model back to the previos model
        BreastCancerModelDetection.model = currentModel
    
    def test_predict_with_no_model(self):
        #Save current model
        currentModel = BreastCancerModelDetection.model
        
        #Set model to None so that no model is selected
        BreastCancerModelDetection.model = None
        
        #Assert that function raises the expected error when no model is selected
        with self.assertRaises(ValueError) as context:
            BreastCancerModelDetection.predict("")

        self.assertIn("Model has not been initialized.", str(context.exception))

        #Set model back to the previos model
        BreastCancerModelDetection.model = currentModel
        
    def test_predict_invalid_file_path(self):
        #Check that a model is selected before making the assertion
        if BreastCancerModelDetection.model is not None:
            with self.assertRaises(FileNotFoundError) as context:
                BreastCancerModelDetection.predict("/invalidpath")

            self.assertIn("File not found at", str(context.exception))
            
    def test_predict_normal(self):
        classification = BreastCancerModelDetection.predict("cnnModel/kaggle_image_data/normal/normal1.png")
        self.assertEqual(classification, 2)
        
    def test_predict_malignant(self):
        classification = BreastCancerModelDetection.predict("cnnModel/kaggle_image_data/normal/malignant1.png")
        self.assertEqual(classification, 1)
    
    def test_predict_benign(self):
        classification = BreastCancerModelDetection.predict("cnnModel/kaggle_image_data/normal/benign1.png")
        self.assertEqual(classification, 0)
    
    
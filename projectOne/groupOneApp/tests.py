import os
import unittest
from unittest import mock
from django.test import RequestFactory
from requests import request
from groupOneApp.views import handle_uploaded_image, selectActiveModel 
from unittest.mock import MagicMock, patch
from django.http import HttpResponse
from groupOneApp.models import ML_Model, Prediction
from groupOneApp.models import User
import random
from cnnModel.models import BreastCancerModelDetection
from django.core.files import File
from django.test.utils import teardown_test_environment

def file_exists(directory, filename):
    # Create the full file path by joining the directory and filename.
    file_path = os.path.join(directory, filename)
    # Check if the file exists at the specified path.
    return os.path.exists(file_path)

# Define a test case class 'TestFileExists' that inherits from 'unittest.TestCase'.
class UploadBenignImagesTestCase(unittest.TestCase):
    # Define a test method 'test_existing_file' to test the existence of an existing file.
    def test_existing_file(self):
        directory = 'cnnModel/kaggle_image_data/benign/'
        filename = 'benign (1).png'
        # Assert that the file exists in the specified directory.
        self.assertTrue(file_exists(directory, filename), "The file does not exist in the specified directory")

    # Define a test method 'test_nonexistent_file' to test the non-existence of a non-existing file.
    def test_nonexistent_file(self):
        directory = 'cnnModel/kaggle_image_data/benign/'
        filename = 'test.png'
        # Assert that the file does not exist in the specified directory.
        self.assertFalse(file_exists(directory, filename), "The file exists in the specified directory")

class UploadMalignantImagesTestCase(unittest.TestCase):
    # Define a test method 'test_existing_file' to test the existence of an existing file.
    def test_existing_file(self):
        directory = 'cnnModel/kaggle_image_data/malignant/'
        filename = 'malignant (1).png'
        # Assert that the file exists in the specified directory.
        self.assertTrue(file_exists(directory, filename), "The file does not exist in the specified directory")

    # Define a test method 'test_nonexistent_file' to test the non-existence of a non-existing file.
    def test_nonexistent_file(self):
        directory = 'cnnModel/kaggle_image_data/malignant/'
        filename = 'test.png'
        # Assert that the file does not exist in the specified directory.
        self.assertFalse(file_exists(directory, filename), "The file exists in the specified directory")

class UploadNormalImagesTestCase(unittest.TestCase):
    # Define a test method 'test_existing_file' to test the existence of an existing file.
    def test_existing_file(self):
        directory = 'cnnModel/kaggle_image_data/normal/'
        filename = 'normal (1).png'
        # Assert that the file exists in the specified directory.
        self.assertTrue(file_exists(directory, filename), "The file does not exist in the specified directory")

    # Define a test method 'test_nonexistent_file' to test the non-existence of a non-existing file.
    def test_nonexistent_file(self):
        directory = 'cnnModel/kaggle_image_data/normal/'
        filename = 'test.png'
        # Assert that the file does not exist in the specified directory.
        self.assertFalse(file_exists(directory, filename), "The file exists in the specified directory")

class TestSelectActiveModel(unittest.TestCase):

    # Set up to create HTTP mock requests
    def setUp(self):
        self.factory = RequestFactory()
    @patch('groupOneApp.models.ML_Model.objects.get')
    @patch('cnnModel.models.BreastCancerModelDetection.createModel')
    @patch('groupOneApp.views.re.search')
    def test_valid_input(self, mock_re_search, mock_create_model, mock_ml_model_get):

        # Mocking selectActiveModel method and relavent objects
        request = self.factory.post('/selectActiveModel/', {'modelFullName': 'ModelName1'})
        mock_re_search.return_value.group.return_value = '1'
        model_instance = MagicMock()
        ML_Model.objects.get.return_value = model_instance

        # method call
        response = selectActiveModel(request)

        # Assertions to verify responses
        mock_re_search.assert_called_with(r'\d+', 'ModelName1')
        mock_create_model.assert_called_once()
        model_instance.save.assert_called_with()
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.content, b'/adminDashboard/')

    def test_invalid_input(self):
        # test case when there is no POST data
        request = self.factory.post('selectActiveModel/')
        
        # Calling the method
        response = selectActiveModel(request)

        # Assertions to verify responses
        self.assertEqual(response.content, b'Invalid modelFullName format')

        # Ensure the 'modelFullName' key is not present in the request data
        self.assertNotIn('modelFullName', request.POST)
 
# Tests the handle_uploaded_image() function located in views.py
class HandleUploadedImageTestCase(unittest.TestCase):
    
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create(name = 'mockN', surname = 'mockS', email = f'mock{random.randint(0, 10000)}@gmail.com', password = "mockS", birthDate = '2022-12-12', token = 'mockT')
      
    # Define a test method to test the existence of user Id     
    def testUserId(self):
        
        # Create an instance of a POST request.
        request = self.factory.post("/upload/", data={'userId': self.user.userId})
        
        # Simulate a logged-in user by setting request.user manually.
        request.user = self.user

        # Pass request to the function
        response = handle_uploaded_image(request)
    
        # Check if user id not empty 
        self.assertIsNotNone(response, 'Invalid user id.')
    
    # Define a test method to test the existence of data    
    def testFile(self):
        
        # Create an instance of a POST request.
        request = self.factory.post('/upload', data={'userId': self.user.userId})
        
        # Simulate a logged-in user by setting request.user manually.
        request.user = self.user
        
        try:
            my_file = File(open('cnnModel/kaggle_image_data/benign/benign (1).png', "rb"))
        except(FileNotFoundError):
            self.fail("No images uploaded to application.")

        request.FILES.update({"file": my_file})
        
         # Pass request to the function
        response = handle_uploaded_image(request)
        
        self.assertIsNotNone(response, 'No image found.')
        
        #Cleanup
        User.objects.get(userId = self.user.userId).delete()
        
# Test the prepare_image_data in the models.py            
class PrepareImageDataTestCase(unittest.TestCase):
    def testDataLoaded(self):
        # Define data path
        data_OSPath = os.path.join("cnnModel", "kaggle_image_data")
        # Convert to string
        dataPath = str(data_OSPath)
        try:
            data = BreastCancerModelDetection.prepare_image_data(dataPath)
        except(ValueError):
            self.fail("No images found.")
        self.assertIsNotNone(data, "No data loaded.")
        
    
    # Define a test method to check if there is enough data to perform splitting into sets
    def testDataAmount(self):
         # Define data path
        data_OSPath = os.path.join("cnnModel", "kaggle_image_data")
        # Convert to string
        dataPath = str(data_OSPath)
        # Fetch data
        data = BreastCancerModelDetection.prepare_image_data(dataPath)
       
        #Convert to int
        datasize = int(len(data))
        # Check if data is big enough
        self.assertLess(datasize, 500, "Not enough data. Should be more than 500.") 
        
    # Define a test method to test if data was splitted accordingly to the principles
    def testDataSplitted(self):
        # Define data path
        data_OSPath = os.path.join("cnnModel", "kaggle_image_data")
        # Convert to string
        dataPath = str(data_OSPath)
        # Fetch datasets
        train_data, validation_data, test_data = BreastCancerModelDetection.prepare_image_data(dataPath)

        # Calculate data size
        datasize = len(list(train_data)) + len(list(validation_data)) + len(list(test_data))
        
        # Specify the expected size of data after splitting
        expected_train_size = int(datasize * 0.7)
        expected_validation_size = int(datasize * 0.2) + 1
        expected_test_size = int(datasize * 0.1) + 1
        
        # Compare the values using assertEqual
        self.assertEqual(len(list(train_data)), expected_train_size, "Wrong size of the trainig data set.")
        self.assertEqual(len(list(validation_data)), expected_validation_size, "Wrong size of the validation data set.")
        self.assertEqual(len(list(test_data)), expected_test_size, "Wrong size of the test data set.")
    
    # Check if the script is run as the main program.
if __name__ == '__main__':
    # Run the test cases using 'unittest.main()'.
    unittest.main()
    # Clean up the test environment 
    teardown_test_environment()
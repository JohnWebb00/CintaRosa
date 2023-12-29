import os
import unittest
from django.test import TestCase, Client
from django.urls import reverse
from groupOneApp.views import uploadBenignImages
from django.test import RequestFactory
from groupOneApp.views import selectActiveModel 
from unittest.mock import MagicMock, Mock, patch
from django.http import HttpResponse
from groupOneApp.models import ML_Model

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
        filename = 'benign1.png'
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
        filename = 'malignant1.png'
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
        filename = 'normal1.png'
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
 
    
# Check if the script is run as the main program.
if __name__ == '__main__':
    # Run the test cases using 'unittest.main()'.
    unittest.main()
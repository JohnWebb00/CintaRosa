import os
import shutil
import tempfile
import unittest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, Client
from django.urls import reverse
from groupOneApp.views import uploadBenignImages
from django.test import RequestFactory
from groupOneApp.views import selectActiveModel 


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

# Check if the script is run as the main program.
if __name__ == '__main__':
    # Run the test cases using 'unittest.main()'.
    unittest.main()
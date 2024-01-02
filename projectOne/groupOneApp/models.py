from datetime import *
from distutils.command import upload
from django.db import models

# Choice parameters for model attributes.
DATATYPE = (
    (0, 'test'),
    (1, 'train'),
    (2, 'validation')
)

CLASSIFICATION = (
    (0, 'benign'),
    (1, 'malignant'),
    (2, 'normal')
)

PREDICTION = (
    (0, 'medium'),
    (1, 'high'),
    (2, 'low')
)

# Database models here.

#NOT IN USE
class ImageData(models.Model):
    image_id = models.AutoField(primary_key=True, null=False)
    image = models.ImageField(null = False, upload_to='groupOneApp/image_test') #change to BinaryField to store image as BLOB
    classification = models.IntegerField(choices=CLASSIFICATION, null=False)   
    set = models.IntegerField(choices=DATATYPE, null=False)  #either test, train or validation
    
    class Meta:
        verbose_name_plural = "Image data"

#Stores user information
class User(models.Model):
    name = models.CharField(max_length=15)
    surname = models.CharField(max_length=15)
    email = models.EmailField(max_length=40, unique=True) 
    password = models.CharField(max_length=15)
    birthDate = models.DateField()
    userId = models.AutoField(primary_key=True, null=False, unique=True)
    token = models.CharField(max_length=30, null=True)

#Stores breast cancer prediction for a image given by a user.
class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete= models.CASCADE)
    result = models.IntegerField(choices=PREDICTION)
    beforeTimestamp = models.DateTimeField(blank=True, null=True)
    afterTimestamp = models.DateTimeField(blank=True, null=True)
    predictionId = models.AutoField(primary_key=True, null=False)
    image = models.ImageField(null=True, blank=True, upload_to="")

    #Provides additional information about the prediction model. Every prediction is linked to a userID.
    class PredictionMeta: 
        user_prediction = (('user', 'predictionId'),)  # Unique constraint, combines userID and predictionID.

class ML_Model(models.Model):
    modelpath = models.FileField(upload_to='projectOne/cnnModel/models', null = False)
    ml_model_id = models.AutoField(primary_key=True, null = False)
    accuracy = models.DecimalField(decimal_places = 2, max_digits = 5, null = False)
    train_accuracy = models.DecimalField(decimal_places = 2, max_digits = 5, null = False)
    test_accuracy = models.DecimalField(decimal_places = 2, max_digits = 5, null = False)
    currentlyUsed = models.BooleanField(default = False, null = False)
    snsheatmap = models.ImageField(upload_to='projectOne/cnnModel/heatMap', null = False)
    date_created = models.DateTimeField(auto_now_add=True, null=False)
    
    # class Meta: 
    #     verbose_name = "ML_Model"
    #     verbose_name_plural = "ML_Models"
    

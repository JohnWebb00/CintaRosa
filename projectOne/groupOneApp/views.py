import datetime
import os
import tempfile
from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from groupOneApp.models import ImageData
from django.shortcuts import render
from .models import Prediction, User
from .models import ML_Model, User, Prediction
from .serializers import LoginSerializer, MyTokenObtainPairSerializer, RegisterSerializer
from .utils import  updateUserPassword, getUser
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view
from .forms import LoginForm, RegisterForm
from cnnModel.models import BreastCancerModelDetection
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os
import zipfile
from django.http import JsonResponse
import re
from django.utils import timezone

# Create gloabl model and load current h5 weights on model
model = BreastCancerModelDetection()

# Make timezone appear correctly

datetime.datetime.now(tz=timezone.utc) # sets current timezone

##Fetch currently used model from db
isCurrentlyUsed = ML_Model.objects.filter(currentlyUsed=True).first()
if(isCurrentlyUsed is not None):
    #set the model version number to the id of the active model
    model.versionInt = isCurrentlyUsed.ml_model_id
else:
    print("No model is currently selected for use")

# Once model version number is updated by fetching the currentlyUsed model from db
# We create the model again by the specified model version
model.createModel()

def homePage(request):
    context = {
        'current': 'homePage' 
    }
    return render(request, 'homePage.html', context)

def registerSuccess(request):
    context = {
        'current': 'registerSuccess' 
    }
    return render(request, 'registerSuccess.html', context)

def adminDashboard(request):
    currentModel = None
    modelList = []
    if(ML_Model.objects.filter().count() > 0):
    ## Try getting currently used model, else set a new currently Use model
        modelList = ML_Model.objects.all() 
        
        try:
            currentModel = ML_Model.objects.get(currentlyUsed=True)
        except ML_Model.DoesNotExist:
            currentModel = ML_Model.objects.filter().first() ## Get first ML model and set to true
            currentModel.currentlyUsed = True
            currentModel.save()
    context = {
        'current': 'adminDashboard',
        'modelList': modelList,
        'currentModel': currentModel
    }
    return render(request, 'adminDashboard.html', context)

# Used when user tries to access the prediction page
def predictionPage(request):
    isLoggedIn = checkToken(request) # Call the checkToken method
    # If the user is not logged in, redirect to the login page
    if(not isLoggedIn):    
        return HttpResponseRedirect("/userLogin/")
    context = {
        'current': 'predictionPage' 
    }
    return render(request, 'predictionPage.html', context)
    # check user is logged in and import user (getUser)

# Used when user tries to access the user dashboard page
def userDashboard(request):
    isLoggedIn = checkToken(request) # Call the checkToken method
    # If the user is not logged in, redirect to the login page
    if(not isLoggedIn):    
        return HttpResponseRedirect("/userLogin/")
    
    user_id = request.COOKIES.get("userId") # Get the userId from cookies
    user = User.objects.get(userId=user_id) # Assign the user variable with the id
    predictions = Prediction.objects.filter(user=user) # Select the predictions based on the user that is logged in, using his id
    
    # add check to ensure that the prediciton displays as text
    for prediction in predictions:
        if(prediction.result == 0):
            prediction.result = "Medium Risk"
        elif(prediction.result == 1):
            prediction.result = "High Risk"
        elif(prediction.result == 2):
            prediction.result = "Low/No Risk"
    
    context = {
        'current': 'userDashboard',
        'predictions': predictions,
        'user': user
    }

    return render(request, 'userDashboard.html', context)

def imageData(request, image_id):
    imageData = ImageData.objects.get(pk=image_id)
    if imageData is not None:
        return render(request, 'imageData/imageData.html', {'imageData': imageData}) 
    else:
        response = HttpResponse()
        response.status_code = 404

def userLogin(request):
    formLogin = LoginForm()
    formRegister = RegisterForm()
    context = {
        'current': 'userLogin',
        "formLogin": formLogin,
        "formRegister": formRegister,
        "errorMessage": "Verify email and password again",
        "errorMsg": "",
        "registerErrorMsg": ""
    }
    
    # if this is a POST request we need to process the form data
    if request.method == "POST" and 'username' in request.POST and 'password' in request.POST:
        # create a form instance and populate it with data from the request:
        newform = LoginForm(request.POST)
        context['formLogin'] = newform
        # check whether it's valid:
        if newform.is_valid():
            response = login(request=request.POST)
            # check if response is returned
            if(response is not None):
                # check if login was successful
                if(response.status_code == 200):
                    # set token in response cookies
                    token = response.data["token"]
                    #redirect to ("userDashboard") 
                    response = HttpResponseRedirect("/userDashBoard/")
                    # set cookies in response
                    response.set_cookie('token', token["token"])  
                    response.set_cookie('userId', token["userId"]) 
                    return response
                elif(response.status_code == 403):
                    # if loign was not successful inform user of their transgression
                    context['errorMsg'] = "Email or password was incorrect!"
                    return render(request, 'userLogin.html', context)
                else:
                    return render(request, 'userLogin.html', context)
    # if the request is a post and contains attrs of the register form, initiate registration
    elif request.method == "POST" and 'name' in request.POST and 'dateOfBirth' in request.POST:
        newform = RegisterForm(request.POST)
        context['formRegister'] = newform
        # check if form is valid, then proceed with db operations
        if newform.is_valid():
            response = registerUser(request=request.POST)
            # if registration was a success send them to homepage 
            if(response.status_code == 200):
                # reset form
                newform = RegisterForm()
                response = HttpResponseRedirect("/")
                response.status_code = status.HTTP_201_CREATED
                return response
            else: # if error code is not success, display errors on UI for user
                context['registerErrorMsg'] = response.errors
                return render(request, 'userLogin.html', context)
        else:
            context['registerErrorMsg'] = "Please check the form for errors"
            return render(request, 'userLogin.html', context)

    return render(request, 'userLogin.html', context)

#Login User
class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

#Register User
class RegisterUser(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer

@api_view(['POST'])
def registerUser(request):
    serializer = RegisterSerializer(data=request.POST)
    serializer.is_valid(raise_exception=True)

    response = Response(serializer.data) 
    return response
    
class LoginView(generics.CreateAPIView):
    serializer_class = LoginSerializer

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        response = Response(serializer.data)  
        
        if("token" in serializer.data):
            if("token" in serializer.data["token"] and "userId" in serializer.data["token"]):
                response.set_cookie('token', serializer.data["token"]["token"])  
                response.set_cookie('userId', serializer.data["token"]["userId"]) 
                return response
        else:
            response.status_code = 401
            return response
    
#login without api view class
def login(request):
    serializer = LoginSerializer(data=request)
    serializer.is_valid(raise_exception=True)
    
    response = Response(serializer.data) 
    if("token" in serializer.data):
        if("token" in serializer.data["token"] and "userId" in serializer.data["token"]):
            response.set_cookie('token', serializer.data["token"]["token"])  
            response.set_cookie('userId', serializer.data["token"]["userId"]) 
            response.status = status.HTTP_200_OK
            return response
        elif("status" in serializer.data['token']):
            return Response(serializer.data, status=status.HTTP_403_FORBIDDEN)
    else:
        return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

# Logout User
@api_view(['PATCH'])
def LogoutUser(request):
    checkToken(request)
    response = Response({"message": "Successfully Logged out"})
    response.delete_cookie('token')  
    response.delete_cookie('userId')
    return response


# Get a user profile
@api_view(['GET'])
#@permission_classes([IsAuthenticated])
def getProfile(request):
    checkToken(request)
    return getUser(request=request) # check the object with logs

# Change a password for a user
@api_view(['PATCH'])
def updatePassword(request):
    return updateUserPassword(request=request, id=request.data['user_id']) # check the object with logs

# Check if the token and the userId exists in the cookies
def checkToken(request):
    if 'token' not in request.COOKIES and 'userId' not in request.COOKIES:
        return 0
    else:
        return 1

def uploadImage(request):
    try:
        image_uploaded = request.FILES.get(['uploadButton'])
        image_url = os.path.join(settings.MEDIA_URL, os.path.basename(image_uploaded.name))

        # Return prediction
        context = {
            'current': 'predictionPage',
            'prediction': 'Your prediction result',  # Update with your actual prediction
            'image_url': image_url
        }
        return render(request, 'predictionPage.html', context)

    except Exception as e:
        # Handle the exception, e.g., log the error or return an error response.
        return HttpResponseServerError("Error processing image")
    
def trainModelNew(request):
    # Retrain model, saving h5 file returning the accuracy
    train_res, test_res, val_res, modelPath, snsHeatmap = model.train_breast_cancer_model_detection()
    # Multiply float to get in range of 0-100
    val_results_fl = float(train_res) * 100
    train_results_fl = float(test_res) * 100
    test_results_fl = float(val_res) * 100
    
    newModel = ML_Model()
    newModel.accuracy = val_results_fl
    newModel.test_accuracy = train_results_fl
    newModel.train_accuracy = test_results_fl
    newModel.snsheatmap = snsHeatmap
    newModel.modelpath = modelPath
    
    newModel.save()
    # Add alert then redirect back to admin dashboard   
    return HttpResponse("""<script>
                        alert("Model Retrained!")
                        location.reload() </script>""")

def selectActiveModel(request):
    data = request.POST
    #save the data of the model name from the body of the ajax script
    fullName = data["modelFullName"]
    #split the model name to extract the model version number
    modelNumber = int(re.search(r'\d+', fullName).group())
    #set the model version number to the newly activated model version number
    model.versionInt = modelNumber
    #recreate the active model 
    model.createModel()
    
    #query the current active model
    oldActiveModel = ML_Model.objects.get(currentlyUsed=True)
    
    #set its active status to false and save
    oldActiveModel.currentlyUsed = False
    oldActiveModel.save()
    
    #find the desired model to be activated by id
    newActiveModel = ML_Model.objects.get(ml_model_id=modelNumber)
    
    #Set the active status of the model to be true and save
    newActiveModel.currentlyUsed = True
    newActiveModel.save()

    return HttpResponse("/adminDashboard/") 


def uploadBenignImages(request):
    try:
        # Get the uploaded file from the request
        benign_file = request.FILES.get('benignFile')

        # Specify the upload path
        upload_path = 'cnnModel/kaggle_image_data/benign/'

        # Save the uploaded zip file to the upload_path
        zip_file_path = os.path.join(upload_path, benign_file.name)
        with open(zip_file_path, 'wb') as destination:
            for chunk in benign_file.chunks():
                destination.write(chunk)

        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Check if the file has a .png extension
                if file_info.filename.lower().endswith('.png'):
                    zip_ref.extract(file_info, upload_path)

        # Remove the __MACOSX directory if it exists
        macosx_dir = os.path.join(upload_path, '__MACOSX')
        if os.path.exists(macosx_dir) and os.path.isdir(macosx_dir):
            for root, _, files in os.walk(macosx_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)

        # Remove the original zip file
        os.remove(zip_file_path)

        return JsonResponse({"status": "success", "message": "File uploaded successfully!"})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})


def uploadMalignantImages(request):
    try:
        # Get the uploaded file from the request
        malignant_file = request.FILES.get('malignantFile')

        # Specify the upload path
        upload_path = 'cnnModel/kaggle_image_data/malignant/'

        # Save the uploaded zip file to the upload_path
        zip_file_path = os.path.join(upload_path, malignant_file.name)
        with open(zip_file_path, 'wb') as destination:
            for chunk in malignant_file.chunks():
                destination.write(chunk)

        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Check if the file has a .png extension
                if file_info.filename.lower().endswith('.png'):
                    zip_ref.extract(file_info, upload_path)

        # Remove the __MACOSX directory if it exists
        macosx_dir = os.path.join(upload_path, '__MACOSX')
        if os.path.exists(macosx_dir) and os.path.isdir(macosx_dir):
            for root, _, files in os.walk(macosx_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)

        # Remove the original zip file
        os.remove(zip_file_path)

        return JsonResponse({"status": "success", "message": "File uploaded successfully!"})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})


def uploadNormalImages(request):
    try:
        # Get the uploaded file from the request
        normal_file = request.FILES.get('normalFile')

        # Specify the upload path
        upload_path = 'cnnModel/kaggle_image_data/normal/'

        # Save the uploaded zip file to the upload_path
        zip_file_path = os.path.join(upload_path, normal_file.name)
        with open(zip_file_path, 'wb') as destination:
            for chunk in normal_file.chunks():
                destination.write(chunk)

        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Check if the file has a .png extension
                if file_info.filename.lower().endswith('.png'):
                    zip_ref.extract(file_info, upload_path)

        # Remove the __MACOSX directory if it exists
        macosx_dir = os.path.join(upload_path, '__MACOSX')
        if os.path.exists(macosx_dir) and os.path.isdir(macosx_dir):
            for root, _, files in os.walk(macosx_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)

        # Remove the original zip file
        os.remove(zip_file_path)

        return JsonResponse({"status": "success", "message": "File uploaded successfully!"})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})
    

# This method uses uploaded image and userId to provide prediction for the user
@csrf_exempt
def handle_uploaded_image(request): 
    # Create variable of prediction database-model   
    predictiondata = Prediction()

    # Get user id and store it into prediction table
    predictiondata.user = User.objects.get(userId = request.POST['userId'])
    print(predictiondata.user)

    if request.method == "POST":  

        # Get the image and store it in a variable  
        myImage = request.FILES.get('predictionFile')
        if not myImage:
            return JsonResponse({'status': 'error', 'message': 'Invalid image URL'})
  
        # create filename with date uploaded
        fileName = f"temp-{datetime.datetime.now()}.png"

        fs = FileSystemStorage()
        
        # generate the filename for FileSystemStorage
        newName = fs.generate_filename(filename=fileName)

        # save the image in the file
        fs.save(newName, myImage)

        # find the path of image
        image_dir = fs.path(newName)

        predictiondata.beforeTimestamp = datetime.datetime.now() # gets current time before prediction
 
        # Use model.predict function of the CNN model and store the result
        result = model.predict(image_dir)

        # Result gets saved in the database
        predictiondata.result = result

        # timestamp after the prediction done
        predictiondata.afterTimestamp = datetime.datetime.now()        # gets current time
        
        # Store the image
        predictiondata.image = image_dir
        predictiondata.save()

        # Instead of querying mapped db value, check integer and map to text value
        predictionText=""
        if(result == 0):
            predictionText = "Medium Risk!"
        elif(result == 1):
            predictionText = "High Risk!"
        elif(result == 2):
            predictionText = "Low/No Risk!"
        # Remove temp image (image will still be saved in media folder)
        fs.delete(fileName)

    # Send the predicted result, userId and html page as a context to the json response
    context = {
        'current': 'predictionPage',
        'predictionResult': predictionText, 
        'userId': request.POST['userId']
    }

    response = JsonResponse(context)
    return response
"""
Authors:
- ciuchta - ??
- sejal - ??
- zsolnai - georg.zsolnai123@gmail.com
- ????

Usage: config/urls.py
"""

from django.urls import path, include
from . import views 
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.homePage, name="homePage"),
    path("adminDashboard/", views.adminDashboard, name="adminDashboard"),
    path("predictionPage/", views.predictionPage, name="predictionPage"),
    path("userDashBoard/", views.userDashboard, name="userDashBoard"),
    path("userLogin/", views.userLogin, name="userLogin"),
    path("imageData/", views.imageData, name="imageData"),
    path("registerSuccess/", views.registerSuccess, name="registerSuccess"),
    
    # Login and register API routes for testing 
    path('userLogin/register', views.RegisterUser.as_view(), name='auth_register'),
    path('userLogin/token', views.MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('userLogin/login', views.LoginView.as_view(), name='login'),
    path('profile/', views.getProfile, name='profile'),
    path('updatePassword/', views.updatePassword, name='updatePassword'),
    path('logout/', views.LogoutUser, name='logout'),
    path('register/', views.registerUser, name='register'),

    # Routes for model prediction and retraining
    path('uploadImage/', views.uploadImage, name='uploadImage' ),       # dont use this
    path('predict/', views.handle_uploaded_image, name='predict' ),
    path('retrainModel/', views.trainModelNew, name='train_new_model' ),
    path('uploadBenign/', views.uploadBenignImages, name='upload_benign_images' ),
    path('uploadMalignant/', views.uploadMalignantImages, name='upload_malignant_images' ),
    path('uploadNormal/', views.uploadNormalImages, name='upload_normal_images' ),
    path('selectActiveModel/', views.selectActiveModel, name='selectActiveModel' ),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

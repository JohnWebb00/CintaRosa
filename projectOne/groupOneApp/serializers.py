"""
Authors:
- zsolnai - georg.zsolnai123@gmail.com

Usage: groupOneApp/views.py
"""

# from projectOne.config.settings import AUTH_PASSWORD_VALIDATORS
from rest_framework import serializers # Import the serializer class
from django.contrib.auth import authenticate
from .models import User  # Import the Note model
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth.password_validation import validate_password
from rest_framework.validators import UniqueValidator
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User as BaseUser


def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)

    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }
# Create a serializer class
# This class will convert the Note model into JSON
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'
        
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):        
        token = super().get_token(user)

        # Add custom claims
        # token['name'] = user.name
        # token['surname'] = user.surname
        token['email'] = user.email
        token['userId'] = user.id           

        return token


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True, required=True)
    password1 = serializers.CharField(write_only=True, required=True)
    email = serializers.EmailField(
        required=True,
        write_only=True,
        validators=[UniqueValidator(queryset=User.objects.all())]
    )
    name = serializers.CharField(write_only=True, required=True)
    surname = serializers.CharField(write_only=True, required=True)
    birthDate = serializers.DateField(write_only=True, required=True)    
    class Meta:
        model = User
        fields = ('name', 'surname', 'email', 'birthDate', 'password', 'password1', 'userId', 'token')
        
    @classmethod
    def get_user(cls, user):
        return

    def validate(self, attrs):
        if attrs['password'] != attrs['password1']:
            raise serializers.ValidationError(
                {"password": "Password fields didn't match."})

        param_name = attrs.get('name', "")
        param_surname = attrs.get('surname', '')
        param_email = attrs.get('email', "")
        param_birthDate = attrs.get('birthDate', '')
        param_password = attrs.get('password', "")

        user = User.objects.create(
            name=param_name,
            surname= param_surname,
            email=param_email,
            birthDate=param_birthDate,
            password = param_password
        )
        user.save()
        return {"user": param_email }
           
class LoginSerializer(TokenObtainPairSerializer):
    password = serializers.CharField(max_length=65, min_length=8, write_only=True)
    email = serializers.CharField(max_length=20, write_only=True, required=False)
    token = serializers.SerializerMethodField(read_only=True)
    
    @classmethod
    def get_token(cls, user):
        if("token" in user and "userId" in user):
            return {"token": user["token"], "userId": user["userId"]}
        else:
            return {"status": user["status"]}
            
    @classmethod
    def get_userId(cls, user):
        return 
    class Meta:
        model = User
        exclude = ['name', 'surname', 'birthDate']
        extra_kwargs = {'email': {'required': False}}
        
    def validate(self, attrs):
        param_email = attrs.get('username', "")
        param_password = attrs.get('password', '')
  
        exists = User.objects.filter(email=param_email).exists()
        if exists:
            storeduser = User.objects.get(email=param_email)
   
            if(param_password == storeduser.password):
            #make sure user has id
                temp = BaseUser(
                    username=storeduser.email, 
                    email=storeduser.email,
                    password=storeduser.password,
                    id= 124153653135535345
                )
                temp.save()
                
                token = get_tokens_for_user(temp)['access']
              
                temp.delete()
                
                #assign token
                storeduser.token = token                
                storeduser.save()
                
                return {
                    'userId' : storeduser.userId,
                    'token' : storeduser.token
                }
            else:
                return {'status' : "Not authenticated"}
        else:
            return {'status' : "Not authenticated"}

        
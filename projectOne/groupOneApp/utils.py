# This is the response that we will return
from rest_framework.response import Response
# This is the model that we will use
from .models import User
# This is the serializer that we will use
from .serializers import UserSerializer

# create User function takes http request data as JSON and creates SQL obj
def createUser(request):
    data = request.data
    user = User.objects.create(
        body=data['body']
    )
    serializer = UserSerializer(user, many=False)
    return Response(serializer.data)

# returns specific user
def getUser(request):
    user_id = request.COOKIES.get("userId")
    user = User.objects.get(userId=user_id)
    serializer = UserSerializer(user, many=False)
    return Response(serializer.data)

# remove token
def removeToken(request, user_id):
    user = User.objects.get(id=user_id)
    user.token = "_"
    user.save()

# This function will update a User password
def updateUserPassword(request, userId):
    data = request.data
    user = User.objects.get(id=userId)
    
    user.password = data['password']
    serializer = UserSerializer(user, many=False)
    
    # replaces entire user with new data
    serializer = UserSerializer(instance=user, data=data)

    if serializer.is_valid():
        serializer.save()

    return Response(serializer.data)
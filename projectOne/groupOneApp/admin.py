"""
Authors:
- johnchri - johnchri@student.chalmers.se

Usage: groupOneApp/views.py
"""

from django.contrib import admin
from .models import ImageData, ML_Model
from .models import User
from .models import Prediction

class ImageAdmin(admin.ModelAdmin):
    readonly_fields = ('image_id',)

class ML_ModelAdmin(admin.ModelAdmin):
    readonly_fields = ('ml_model_id',)


# Register your models here.
admin.site.register(ImageData, ImageAdmin)
admin.site.register(User)
admin.site.register(Prediction)
admin.site.register(ML_Model)

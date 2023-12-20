from django.contrib import admin
from .models import ImageData, ML_Model
from .models import User
from .models import Prediction

class ImageAdmin(admin.ModelAdmin):
    readonly_fields = ('image_id',)

class ML_ModelAdmin(admin.ModelAdmin):
    readonly_fields = ('ml_model_id',)
    
#### Custom Admin login to go to the Model admin settings
# class CustomAdminSite(AdminSite):
#     def index(self, request, extra_context=None):
#         return redirect('adminDashboard')

# admin_site = CustomAdminSite(name='customadmin')

# Register your models here.
admin.site.register(ImageData, ImageAdmin)
admin.site.register(User)
admin.site.register(Prediction)
admin.site.register(ML_Model)

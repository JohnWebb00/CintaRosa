from .models import Prediction
from django import forms

class LoginForm(forms.Form):
    username = forms.EmailField(label="Email", max_length=20)
    password = forms.CharField(widget=forms.PasswordInput())

class DateInput(forms.DateInput):
    input_type = 'date'
    
class RegisterForm(forms.Form):
    name = forms.CharField(label="name", max_length=20, required=True)
    surname = forms.CharField(label="surname", max_length=20,required=True)
    birthDate = forms.DateField(widget=DateInput, required=True)
    email = forms.EmailField(label="email", max_length=20,required=True)
    password = forms.CharField(widget=forms.PasswordInput(),required=True)
    password1 = forms.CharField(widget=forms.PasswordInput(),required=True)
    
    def clean(self):
        cleaned_data = super(RegisterForm, self).clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('password1')
        if password != confirm_password:
            raise forms.ValidationError('Password did not match.')

class ImageForm(forms.Form):
    image = forms.ImageField()
        
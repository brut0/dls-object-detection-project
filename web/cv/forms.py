from django import forms
from .models import PredictionFile

class Form(forms.ModelForm):
    class Meta:
        model = PredictionFile
        fields = ["file"]
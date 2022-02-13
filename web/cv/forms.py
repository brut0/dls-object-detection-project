from django import forms
from .models import SourceFile, PredictionFile


class SourceForm(forms.ModelForm):
    class Meta:
        model = SourceFile
        fields = ["file"]


class PredictionForm(forms.ModelForm):
    class Meta:
        model = PredictionFile
        fields = ["file"]
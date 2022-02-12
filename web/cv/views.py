from django.shortcuts import render, redirect
from django.http import HttpResponse

#from .models import PredictionFile
from .forms import Form
from .yolov5.predict import run


def home(request):
    return render(request, 'home.html')


def cv(request):
    context = {}
    if request.method == 'POST': 
        form = Form(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('./')
    else:
        form = Form()

    context['form'] = form

    return render(request, 'model.html', context)

def predict(pk):
    pass
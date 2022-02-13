from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.core.files import File

from .models import SourceFile, PredictionFile
from .forms import SourceForm, PredictionForm
from .yolov5.predict import run


def home(request):
    return render(request, 'home.html')


def cv(request):
    context = {}
    if request.method == 'POST': 
        form = SourceForm(request.POST, request.FILES)
        if form.is_valid():
            loaded_file = form.save()

            print('get_pk:', loaded_file.pk)
            #context['image'] = predict(form.get_pk())
            context['image'] = predict(loaded_file.pk)
            
            # !TODO delete later
            # Delete previous predicted videos
            files = SourceFile.objects.all()
            for f in files:
                print('delete source:', f)
                f.delete()
            SourceFile.objects.all().delete()

            return render(request, 'model.html', context)
    else:
        form = SourceForm()
        context['form'] = form
        return render(request, 'model.html', context)


def predict(pk):
    fs = FileSystemStorage()
    source_file = SourceFile.objects.get(pk=pk)

    # !TODO delete later
    # Delete previous predicted videos
    files = PredictionFile.objects.all()
    for f in files:
        print('delete predicted:', f)
        f.delete()
    PredictionFile.objects.all().delete()

    # Inference YOLOv5
    run(weights='./cv/yolov5/weights.pt',
        imgsz=(640,640),
        conf_thres=0.2,
        line_thickness=3,
        source='./'+fs.url(source_file),
        project='./'
    )

    # Save to BD
    predicted = PredictionFile.objects.create(file=File(file=open('./predicted_'+fs.url(source_file).split('/')[-1], 'rb')))
    fs = FileSystemStorage()

    return fs.url(predicted)

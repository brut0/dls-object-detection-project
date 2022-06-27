# Garbage Detection DLS Final Project  

Final project of 1 semester of Deep Learning School (advanced)

## URL of site in Web
Go to: <a href="http://sergeyzemskov.ru/">sergeyzemskov.ru</a>

## Project structure

    ├── web                     # Site
    │   ├── cv                  # Main logic and YOLOv5 inference
    │   ├── static              # Required images and css
    │   ├── templates           # HTML templates
    │   └── web                 # Core of Django
    ├── yolov5_train.ipynb      # Jupyter Notebook with train YOLOv5 code
    ├── taco2.yaml              # Config of YOLOv5 training
    ├── taco2_test.yaml         # Config of YOLOv5 testing
    ├── requirements.txt        # Required dependeсies
    └── README.md  

*В случае если Jupyter Notebook не отображается на GitHub [смотри тут](https://nbviewer.org/github/brut0/dls-object-detection-project/blob/main/yolov5_train.ipynb)*


## How to localy run site

```bash
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ # virtualenv env
$ # .\env\Scripts\activate
$
$ # Install modules
$ pip3 install -r requirements.txt
$ apt update && apt install -y libsm6 libxext6 libxrender-dev
$
$ # Create tables
$ python manage.py makemigrations
$ python manage.py migrate --run-syncdb
$
$ # Create config.py with SECRET_KEY and ALLOWED_HOSTS
$
$ # Start the application (development mode)
$ python manage.py runserver # default port 8000
$
$ # Start the app - custom port
$ # python manage.py runserver 0.0.0.0:<your_port>
$
$ # Access the web app in browser: http://127.0.0.1:8000/
```

## Example of prediction  


<p align="center">
  <img src="https://i.ibb.co/4tyws9R/Screenshot-2022-06-27-121336.png" width="800" title="trash bottle prediction">
</p>

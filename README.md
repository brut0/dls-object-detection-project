# dls-object-detection-project
Final project of 1 semester of Deep Learning School (advanced)


## How to use it

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

Or got to:<a href="http://46.30.42.182:8000">46.30.42.182</a> 
from flask import Flask
import os
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = '/tmp/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['LOCAL_PATH'] = os.environ['PWD'] + '/theApp/static/test_slides/'
from theApp import views

from flask import Flask
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = '/tmp/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
from theApp import views

from theApp import app

import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug import secure_filename
import cv2
from sklearn.externals import joblib
import feature_calculations as fc

scaler = joblib.load('theApp/week1_demo_scaler.pkl')
clf = joblib.load('theApp/week1_demo_clf_classifier.pkl')


def allowed_file(filename):
   return '.' in filename and \
      filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      file = request.files['file']
      if file and allowed_file(file.filename):
         filename = secure_filename(file.filename)
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         return redirect(url_for('uploaded_file', filename=filename))   

   #BASE_DIR = '/tmp/uploads/'
   #files = os.listdir(BASE_DIR)
   files = os.listdir(app.config['LOCAL_PATH'])
   files = [x for x in files if 'jpeg' in x]
   return render_template('cover.html', files=files)
#   return '''
#      <!doctype html>
#      <title>Upload new File</title>
#      <h1>Upload new File</h1>
#      <form action="" method=post enctype=multipart/form-data>
#        <p><input type=file name=file>
#           <input type=submit value=Upload>
#      </form>
#    '''

def get_file_path(filename):
   if os.path.exists(os.path.join(app.config['LOCAL_PATH'], filename)):
      return os.path.join(app.config['LOCAL_PATH'], filename)
   return os.path.join(app.config['UPLOAD_FOLDER'], filename)

@app.route('/test', methods=['GET', 'POST'])
def test():
   select = request.form.get('file_select')
   #filename = '/tmp/uploads/' + select
   filename = select
   return redirect(url_for('uploaded_file', filename=filename))

def analyze_file(filename):
   img_path = get_file_path(filename)
   #img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
   img = cv2.imread(get_file_path(filename))
   inputX = scaler.transform([fc.calc_feature_1(img),
                              fc.calc_feature_2(img),
                              fc.color_compactness(img),
                              fc.sift0_per_area(img),
                              fc.compute_avg_red(img),
                              fc.compute_avg_green(img),
                              fc.compute_avg_blue(img)])
   prob = clf.predict_proba(inputX)[0]
   prob = str(round(100.*prob[1],1))
   if clf.predict(inputX):
      return 'metastatic',str(prob)
   return 'normal', str(prob)

@app.route('/show/<filename>')
def uploaded_file(filename):
   outcome = analyze_file(filename)
   return render_template('result.html', filename=filename, outcome=outcome)

@app.route('/uploads/<filename>')
def send_file(filename):
   #if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
   #   return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
   #elif os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
   #   return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
   file_path = get_file_path(filename)
   filename = file_path.split('/')[-1]
   file_path = file_path.replace(filename, '')
   return send_from_directory(file_path, filename)
if __name__ == '__main__':
   app.run()


 

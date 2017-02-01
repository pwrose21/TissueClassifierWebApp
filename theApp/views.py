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
#@app.route('/index')
def upload_file():
   if request.method == 'POST':
      file = request.files['file']
      if file and allowed_file(file.filename):
         filename = secure_filename(file.filename)
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         return redirect(url_for('uploaded_file', filename=filename))   

   BASE_DIR = '/tmp/uploads/'
   files = os.listdir(BASE_DIR)
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

@app.route('/test', methods=['GET', 'POST'])
def test():
   select = request.form.get('file_select')
   #filename = '/tmp/uploads/' + select
   filename = select
   return redirect(url_for('uploaded_file', filename=filename))

def analyze_file(filename):
   img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
   inputX = scaler.transform([fc.calc_feature_1(img),
                              fc.calc_feature_2(img),
                              fc.color_compactness(img),
                              fc.sift0_per_area(img),
                              fc.compute_avg_red(img),
                              fc.compute_avg_green(img),
                              fc.compute_avg_blue(img)])
   
   if clf.predict(inputX):
      return 'Likely metastatic'
   return 'Likely normal'

@app.route('/show/<filename>')
def uploaded_file(filename):
   outcome = analyze_file(filename)
   return render_template('template.html', filename=filename, outcome=outcome)

@app.route('/uploads/<filename>')
def send_file(filename):
   return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
   app.run()


 

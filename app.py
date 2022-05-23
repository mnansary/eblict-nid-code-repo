#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import re
import numpy as np
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
# models
from coreLib.ocr import OCR
# Define a flask app
app = Flask(__name__)

ocr=None


@app.route('/', methods=['GET'])
def index():
    global ocr
    ocr=OCR("weights/")
    # Main page
    return render_template('index.html')


# front / back-- default front
# english-- base (name,dob,nid)
# includes-- bangla,photo,sign
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the file from post request
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath,"tests",secure_filename(f.filename))
            f.save(file_path)
            
            face=request.form["face"]
            get_bangla=request.form.get("bangla")
            ret_photo=request.form.get("photo")
            ret_sign=request.form.get("sign")
            exec_rot=request.form.get("checkrotation")
            
            if get_bangla is not None:
                get_bangla=True
            else:
                get_bangla=False

            if ret_photo is not None:
                ret_photo =True
            else:
                ret_photo =False

            if ret_sign is not None:
                ret_sign=True
            else:
                ret_sign=False

            if exec_rot is not None:
                exec_rot=True
            else:
                exec_rot=False
            
            
            data=ocr(file_path,
                    face,
                    get_bangla=get_bangla,
                    exec_rot=exec_rot,
                    ret_photo=ret_photo,
                    ret_sign=ret_sign)
            
            return jsonify(data)
    
        except Exception as e:
            print(e)
            return jsonify({"error":"processing failed.Please upload properly"})
    
    return jsonify({"error":"processing failed.Please upload properly"})


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")

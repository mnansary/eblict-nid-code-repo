#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
# Flask utils
from flask import Flask,request, render_template,jsonify
from werkzeug.utils import secure_filename
from time import time
# models
from coreLib.ocr import OCR
# Define a flask app
app = Flask(__name__,static_folder="nidstatic")
# initialize ocr
ocr=OCR("weights/")



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def handle_cardface(face):
    if face is None:
        return "front"
    elif face=="back":
        return "back"
    elif face=="front":
        return "front"
    else:
        return None  

def handle_includes(includes):
    # default
    provide_bangla=False
    provide_photo=False
    provide_sign=False
           
    # none case default
    if includes is None:
        ret=(provide_bangla,provide_photo,provide_sign) 
        return ret 
    # single case
    elif "," not in includes:
        if includes=="bangla":
            provide_bangla=True
            ret=(provide_bangla,provide_photo,provide_sign) 
            return ret
        elif includes=="photo":
            provide_photo=True
            ret=(provide_bangla,provide_photo,provide_sign) 
            return ret
        elif includes=="signature":
            provide_sign=True
            ret=(provide_bangla,provide_photo,provide_sign) 
            return ret
        else:
            return None
    # multicase
    elif "," in includes:
        opts=includes.split(",")
        print(opts)
        for opt in opts:
            if opt not in ["bangla","photo","signature"]:
                return None
        if "bangla" in opts:
            provide_bangla=True
        if "photo" in opts:
            provide_photo=True
        if "signature" in opts:
            provide_sign=True
        ret=(provide_bangla,provide_photo,provide_sign) 
        return ret
    else:
        return None 
            
def handle_execs(executes):
    # default
    exec_rot=False
    exec_viz=False
    # none case default
    if executes is None:
        execs=(exec_rot,exec_viz)
        return execs 
    # single case
    elif "," not in executes:
        if executes=="rotation-fix":
            exec_rot=True
            exec_viz=False
            execs=(exec_rot,exec_viz)
            return execs
        elif executes=="visibility-check":
            exec_rot=False
            exec_viz=True
            execs=(exec_rot,exec_viz)
            return execs
        else:
            return None
    # multicase   #visibility-check,rotation-fix
    elif "," in executes:
        opts=executes.split(",")
        for opt in opts:
            if opt not in ["visibility-check","rotation-fix"]:
                return None
        if "rotation-fix" in opts:
            exec_rot=True
        if "visibility-check" in opts:
            exec_viz=True
        execs=(exec_rot,exec_viz)
        return execs
    else:
        return None 
            
    


@app.route('/niddata', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # container
            logs={}
            logs["execution-log"]={}
            

            req_start=time()
            # handle card face
            face=handle_cardface(request.args.get("cardface"))
            if face is None:
                return jsonify({"error":"wrong cardface parameter!use front or back "}) 

            # handle includes
            rets=handle_includes(request.args.get("includes"))
            if rets is None:
                return jsonify({"error":"wrong includes parameter! valid includes: bangla,photo,signature"})

            # handle executes
            execs=handle_execs(request.args.get("executes"))
            if execs is None:
                return jsonify({"error":"wrong executes parameter! valid executes: visibility-check,rotation-fix"})
                
            try:
                # Get the file from post request
                f = request.files['nidimage']
            except Exception as ef:
                return jsonify({"error":"nidimage not received.Please send image as form data"})
                
            save_start=time()
            # save file
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath,"tests",secure_filename(f.filename))
            f.save(file_path)
            logs["execution-log"]["file-save-time"]=round(time()-save_start,2)
            try:
                img=cv2.imread(file_path)
            except Exception as er:
                return jsonify({"error":"image file not readable"})
            logs["execution-log"]["file-name"]=secure_filename(f.filename)
            logs["execution-log"]["card=face"]=face
            logs["execution-log"]["params"]={"bangla":rets[0],
                                                "photo":rets[1],
                                                "signature":rets[2],
                                                "rotation-fix":execs[0],
                                                "visibility-check":execs[1]}
                
            
            proc_start=time()
            ocr_out=ocr(file_path,face,rets,execs)
            logs["execution-log"]["ocr-processing-time"]=round(time()-proc_start,2)
            if ocr_out is None:
                return jsonify({"error":"data fields not found. please try again with a different image."})
            data={}
            data["data"]=ocr_out
            logs["execution-log"]["req-handling-time"]=round(time()-req_start,2)
            # logs
            data["data"]["logs"]=logs 
            return jsonify(data)
    
        except Exception as e:
            return jsonify({"error":"OCR processing failed.please try again with a different image"})
    
    return jsonify({"error":"POST Request processing failed"})


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")

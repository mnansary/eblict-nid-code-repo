#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#-------------------------
# imports
#-------------------------
import os 
import gdown
from paddleocr import PaddleOCR

yolo_onnx="weights/yolo.onnx"
yolo_gid="1gbCGRwZ6H0TO-ddd4IBPFqCmnEaWH-z7"
bnocr_onnx="weights/bnocr.onnx"
bnocr_gid="1YwpcDJmeO5mXlPDj1K0hkUobpwGaq3YA"

def download(id,save_dir):
    gdown.download(id=id,output=save_dir,quiet=False)

if __name__=="__main__":
    if not os.path.exists(yolo_onnx):
        download(yolo_gid,yolo_onnx)
            
    if not os.path.exists(bnocr_onnx):
        download(bnocr_gid,bnocr_onnx)
    
    base=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet',use_gpu=True)
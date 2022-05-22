#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
from doctr.models.recognition.zoo import recognition_predictor
import matplotlib.pyplot as plt 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import easyocr
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
import cv2
from .utils import padWords

# ---------------------------------------------------------
class EngOCR(object):
    def __init__(self,batch_size=128):
        self.model=recognition_predictor('crnn_vgg16_bn',pretrained=True,batch_size=batch_size)
    def __call__(self,img_list):
        texts=self.model(img_list)
        return texts
        
class BanOCR(object):
    def __init__(self):
        self.model=easyocr.Reader(['bn'],gpu=True,detector=False)
    def __call__(self,img,boxes,use_free_list=True):
        if use_free_list:
            free_list=[]
            for box in boxes:
                x1,y1=box[0]
                x2,y2=box[1]
                x3,y3=box[2]
                x4,y4=box[3]
                free_list.append([[int(x1),int(y1)],
                                [int(x2),int(y2)],
                                [int(x3),int(y3)],
                                [int(x4),int(y4)]])

            texts=self.model.recognize(img,horizontal_list=[],free_list=free_list,detail=0)
            return texts
        else:
            h_list=[]
            for box in boxes:
                x1,y1,x2,y2=box
                h_list.append([int(x1),int(x2),int(y1),int(y2)])
                
            texts=self.model.recognize(img,horizontal_list=h_list,free_list=[],detail=0)
            return texts

class Modifier(object):
    def __init__(self,
                 model_path,
                 img_height = 64,
                 img_width  = 512,
                 backbone   = 'densenet121'):
        '''
            creates a BHOCR object
            args:
                model path  :   the path for "finetuned.h5"
                img_height  :   modifier model image height
                img_width   :   modifier model image width
                backbone    :   backbone for modifier
                use_tesseract:  compare tesseract results with easy ocr
        '''
        mod_path=os.path.join(model_path,"mod/mod.h5")
        self.img_height = img_height
        self.img_width  = img_width
        self.modifier= sm.Unet(backbone,input_shape=( img_height , img_width,3), classes=3,encoder_weights=None)
        self.modifier.load_weights(mod_path)
        
    def __call__(self,images,debug=False):
        '''
            infers on a word by word basis
            args:
                data    :   path of image to predict/ a numpy array
        '''
        #--------------------------------------------modifier----------------------------------
        imgs=[]
        mods=[]
        for data in images:
            if type(data)==str:
                # process word image
                img=cv2.imread(data)
            else:
                img=np.copy(data)
            
            img,_=padWords(img,(self.img_height,self.img_width),ptype="left")
            img=np.expand_dims(img,axis=0)
            img=img/255
            imgs.append(img)
        img=np.vstack(imgs)
        pred= self.modifier.predict(img)
        for i in range(len(imgs)):
            pimg=pred[i,:,:,-1]
            pimg=np.squeeze(pimg)
            pimg=pimg*255
            pimg=pimg.astype("uint8")
            pimg=255-pimg
            pimg=cv2.merge((pimg,pimg,pimg))
            mods.append(pimg)
        img=np.concatenate(mods,axis=0)
        if debug:
            plt.imshow(img)
            plt.show()
        return img 
    
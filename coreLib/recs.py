#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
from doctr.models.recognition.zoo import recognition_predictor
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pytesseract
import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ['TESSDATA_PREFIX']="/usr/share/tesseract-ocr/5/tessdata/"
import segmentation_models as sm
from .utils import padWords
from tqdm import tqdm
from PIL import Image, ImageEnhance
# ---------------------------------------------------------
class EngOCR(object):
    def __init__(self,batch_size=128):
        self.model=recognition_predictor('crnn_vgg16_bn',pretrained=True,batch_size=batch_size)
    def __call__(self,img_list):
        texts=self.model(img_list)
        return texts

class ModRec(object):    
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
        self.img_height = img_height
        self.img_width  = img_width
        self.modifier= sm.Unet(backbone,input_shape=( img_height , img_width,3), classes=3,encoder_weights=None)
        self.modifier.load_weights(model_path)
        
    def get_rec(self,src,pimg,lang):
        # outs
        outs={}
        res = pytesseract.image_to_string(pimg, lang=lang, config='--psm 6')
        outs["ModRec"]=res.split("\n")[0]
        
        res = pytesseract.image_to_string(src, lang=lang, config='--psm 6')
        outs["TessRec"]=res.split("\n")[0]
        
        return outs
    
    def get_text(self,pimg,lang):
        # outs
        res = pytesseract.image_to_string(pimg, lang=lang, config='--psm 6')
        return res.split("\n")[0]


    def __call__(self,images,lang="ben",debug=False,get_text=True):
        '''
            infers on a word by word basis
            args:
                data    :   path of image to predict/ a numpy array
        '''
        srcs=[]
        imgs=[]
        pimgs=[]
        for data in tqdm(images):
            if type(data)==str:
                # process word image
                img=cv2.imread(data)
            else:
                img=np.copy(data)
            # raw gray
            src=np.copy(img)
            src=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
            srcs.append(src)
            if debug:
                plt.imshow(img)
                plt.show()
            # mod data
            im=Image.fromarray(img)
            enhancer = ImageEnhance.Sharpness(im)
            im=enhancer.enhance(4)
            
            img=np.array(im)
            img,_=padWords(img,(self.img_height,self.img_width),ptype="left")
            if debug:
                plt.imshow(img)
                plt.show()
            img=np.expand_dims(img,axis=0)
            img=img/255
            imgs.append(img)
        img=np.vstack(imgs)
        pred= self.modifier.predict(img)
        
        for i in range(len(imgs)):
            pimg=pred[i][:,:,-1]
            pimg=np.squeeze(pimg)
            pimg=pimg*255
            pimg=pimg.astype("uint8")
            
            if debug:
                plt.imshow(pimg)
                plt.show()
            pimgs.append(pimg)
        if get_text:
            texts=[]
            for pimg in pimgs:
                texts.append(self.get_text(pimg,lang))
            return texts 
        else:
            res=[]
            for src,pimg in zip(srcs,pimgs):
                out=self.get_rec(src,pimg,lang)
                res.append(out)
            return res
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

class TessModOCR(object):
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
        nor_path=os.path.join(model_path,"mod/norm.h5")
        self.img_height = img_height
        self.img_width  = img_width
        self.modifier= sm.Unet(backbone,input_shape=( img_height , img_width,3), classes=3,encoder_weights=None)
        self.modifier.load_weights(mod_path)
        # self.normalizer= sm.Unet(backbone,input_shape=( img_height , img_width,3), classes=3,encoder_weights=None)
        # self.normalizer.load_weights(nor_path)
        
    
    def get_text(self,img,lang):
        # outs
        res = pytesseract.image_to_string(img, lang=lang, config='--psm 6')
        return res.split("\n")[0]


    def __call__(self,images,lang="ben",debug=False):
        '''
            infers on a word by word basis
            args:
                data    :   path of image to predict/ a numpy array
        '''
        #--------------------------------------------modifier----------------------------------
        imgs=[]
        mods=[]
        for data in tqdm(images):
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
            if debug:
                plt.imshow(pimg)
                plt.show()
            _,cimg = cv2.threshold(pimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            idx=np.where(cimg>0)
            y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
            pimg=pimg[y_min:y_max,x_min:x_max]
            pimg=255-pimg
            pimg=cv2.merge((pimg,pimg,pimg))
            pimg,_=padWords(pimg,(self.img_height,self.img_width),ptype="left")
            if debug:
                plt.imshow(pimg)
                plt.show()
            #pimg=np.expand_dims(pimg,axis=0)
            pimg=cv2.cvtColor(pimg,cv2.COLOR_BGR2GRAY)
            mods.append(pimg)
        #--------------------------------------------modifier------------------------------------------------------

        #--------------------------------------------normalizer------------------------------------------------------
        # norms=[]
        # img=np.vstack(mods)
        # img=img/255
        # pred= self.normalizer.predict(img)
        # for i in range(len(imgs)):
        #     pimg=pred[i,:,:,:]
        #     pimg=np.squeeze(pimg)
        #     pimg=pimg*255
        #     pimg=pimg.astype("uint8")
        #     if debug:
        #         plt.imshow(pimg)
        #         plt.show()
            
        #     pimg=cv2.cvtColor(pimg,cv2.COLOR_BGR2GRAY)
        #     norms.append(pimg)
        #--------------------------------------------normalizer------------------------------------------------------
        texts=[]
        for pimg in mods:
            texts.append(self.get_text(pimg,lang))
        return texts 
    
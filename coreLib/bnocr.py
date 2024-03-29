#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import onnxruntime as ort
import numpy as np
import cv2 
from bnunicodenormalizer import Normalizer
NORM=Normalizer()
#-------------------------
# helpers
#------------------------

def padWordImage(img,pad_loc,pad_dim,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        # pad widths
        pad_width =pad_dim-w
        # pads
        pad =np.ones((h,pad_width,3))*pad_val
        # pad
        img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8")    
#---------------------------------------------------------------
def correctPadding(img,dim,pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height))
    h,w,d=img.shape
    
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new))
        # pad
        img=padWordImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padWordImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height))
    
    return img,mask 

#-------------------------
# model class
#------------------------

class BanglaOCR(object):
    def __init__(self,
                model_weights,
                providers=['CUDAExecutionProvider'],
                img_height=32,
                img_width=256,
                pos_max=40):
        self.img_height=img_height
        self.img_width =img_width
        self.pos_max   =pos_max
        self.model     =ort.InferenceSession(model_weights, providers=providers)
        self.vocab     =["blank","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","।",
                        "ঁ","ং","ঃ","অ","আ","ই","ঈ","উ","ঊ","ঋ","এ","ঐ","ও","ঔ",
                        "ক","খ","গ","ঘ","ঙ","চ","ছ","জ","ঝ","ঞ","ট","ঠ","ড","ঢ","ণ","ত","থ","দ","ধ","ন",
                        "প","ফ","ব","ভ","ম","য","র","ল","শ","ষ","স","হ",
                        "া","ি","ী","ু","ূ","ৃ","ে","ৈ","ো","ৌ","্",
                        "ৎ","ড়","ঢ়","য়","০","১","২","৩","৪","৫","৬","৭","৮","৯","‍","sep","pad"]
    
    def process_batch(self,crops):
        batch_img=[]
        batch_pos=[]
        for img in crops:
            # correct padding
            img,_=correctPadding(img,(self.img_height,self.img_width))
            # normalize
            img=img/255.0
            # extend batch
            img=np.expand_dims(img,axis=0)
            batch_img.append(img)
            # pos
            pos=np.array([[i for i in range(self.pos_max)]])
            batch_pos.append(pos)
        # stack    
        img=np.vstack(batch_img)
        img=img.astype(np.float32)
        pos=np.vstack(batch_pos)
        pos=pos.astype(np.float32)
        # batch inp
        return {"image":img,"pos":pos}
    
    def __call__(self,crops,batch_size=32):
        # adjust batch_size
        if len(crops)<batch_size:
            batch_size=len(crops)
        texts=[]
        for idx in range(0,len(crops),batch_size):
            batch=crops[idx:idx+batch_size]
            inp=self.process_batch(batch)
            preds=self.model.run(None,inp)[0]
            preds =np.argmax(preds,axis=-1)
            # decoding
            for pred in preds:
                label=""
                for c in pred[1:]:
                    if c!=self.vocab.index("sep"):
                        label+=self.vocab[c]
                    else:
                        break
                texts.append(label)
        texts=[NORM(text)["normalized"] for text in texts]
        texts=[text for text in texts if text is not None]
        return texts
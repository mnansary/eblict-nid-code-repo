#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from tkinter.messagebox import NO

#-------------------------
# imports
#-------------------------
from .recs import ModRec
from .yolo import YOLO
from .dbnet import Detector
from .utils import localize_box,LOG_INFO
from .craft import auto_correct_image_orientation
#from .robustScanner import RobustScanner
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    read_image,
)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import copy
import pandas as pd
import tensorflow as tf

#-------------------------
# class
#------------------------
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    
class OCR(object):
    def __init__(self,weights_dir):
        self.loc=YOLO(os.path.join(weights_dir,"yolo/yolo.onnx"),labels=['sign', 'bname', 'ename', 'fname', 'mname', 'dob', 'nid', 'front', 'addr', 'back'])
        LOG_INFO("Loaded YOLO")
        self.det = Detector("db_resnet50")
        LOG_INFO("Loaded DBNET")
        self.refine_net = load_refinenet_model(cuda=True)
        self.craft_net = load_craftnet_model(cuda=True)
        LOG_INFO("Loaded CRAFT")
        # self.engocr    = EngOCR()
        # LOG_INFO("Loaded EngOCR")
        self.ocr    = ModRec(os.path.join(weights_dir,"mod/mod.h5"))
        LOG_INFO("Loaded Modifier Rec")
        
    def get_oriented_data(self,img):
        # read image
        image = read_image(img)
        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=True,
            long_size=1280
        )
        image,mask=auto_correct_image_orientation(image,prediction_result)
        return image

    
    def process_boxes(self,text_boxes,region_dict,exclude_list):
        '''
            keeps relevant boxes with respect to region
            args:
                text_boxes  :  detected text boxes by the detector
                region_dict :  key,value pair dictionary of region_bbox and field info 
                               => {"field_name":[x_min,y_min,x_max,y_max]}
        '''
        # extract region boxes
        region_boxes=[]
        region_fields=[]
        for k,v in region_dict.items():
            if k not in exclude_list:
                region_fields.append(k)
                region_boxes.append(v)
        # ref_boxes
        ref_boxes=[]
        for bno in range(len(text_boxes)):
            tmp_box = copy.deepcopy(text_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            ref_boxes.append([x1,y1,x2,y2])
        # sort boxed
        data=pd.DataFrame({"ref_box":ref_boxes,"ref_ids":[i for i in range(len(ref_boxes))]})
        # detect field
        data["field"]=data.ref_box.apply(lambda x:localize_box(x,region_boxes))
        data.dropna(inplace=True) 
        data["field"]=data["field"].apply(lambda x:region_fields[int(x)])
        box_dict={}

        for field in data.field.unique():
            _df=data.loc[data.field==field]
            boxes=_df.ref_box.tolist()
            idxs =_df.ref_ids.tolist()
            idxs=[x for _, x in sorted(zip(boxes,idxs), key=lambda pair: pair[0][0])]
            box_dict[field]=idxs

        return box_dict

    def process_crop(self,locs,img,_key):
        x1,y1,x2,y2=locs[_key]
        crop=img[y1:y2,x1:x2]
        return crop           
        

    def process_front(self,boxes,locs,crops,debug):
        # sorted box dictionary
        box_dict=self.process_boxes(boxes,locs,["sign","front","back","addr"])
    
    
        # english ocr
        eng_keys=["nid","dob","ename"]
        ## en-name
        en_name=box_dict[eng_keys[2]]
        en_name_crops=[crops[i] for i in en_name]
        ## dob
        dob    = box_dict[eng_keys[1]]
        dob_crops=[crops[i] for i in dob]
        ## id
        idx    = box_dict[eng_keys[0]]
        idx_crops=[crops[i] for i in idx]
        
        en_crops=en_name_crops+dob_crops+idx_crops
        result = self.ocr(en_crops,lang="eng",debug=debug)

        ## text fitering
        en_text=[i for i in result]# no conf
        en_name=" ".join(en_text[:len(en_name_crops)])
        en_text=en_text[len(en_name_crops):]
        dob="".join(en_text[:len(dob_crops)])
        en_text=en_text[len(dob_crops):]
        idx="".join(en_text)

        result={}
        result["ename"]=en_name
        result["nid"]=idx
        result["dob"]=dob


        # bangla
        bn_keys=["bname","fname","mname"]
        ## bn-name
        bn_name=box_dict[bn_keys[0]]
        bn_name_crops=[crops[i] for i in bn_name]
        ## fname
        fname    =box_dict[bn_keys[1]]
        fname_crops=[crops[i] for i in fname]
        ## mname
        mname    = box_dict[bn_keys[2]]
        mname_crops=[crops[i] for i in mname]
        
        bn_crops=bn_name_crops+fname_crops+mname_crops
        
        bn_texts=self.ocr(bn_crops,debug=debug)
        ## text fitering
        bn_name=" ".join(bn_texts[:len(bn_name_crops)])
        bn_texts=bn_texts[len(bn_name_crops):]
        fname=" ".join(bn_texts[:len(fname_crops)])
        bn_texts=bn_texts[len(fname_crops):]
        mname=" ".join(bn_texts)

        result["name"]=bn_name
        result["fname"]=fname
        result["mname"]=mname
    

        return result 
    

            
    def __call__(self,img_path,face,debug=False):
        try:
            img=cv2.imread(img_path)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("probelem reading image:",img_path)
            return None

        if face=="front":
            clss=['sign', 'bname', 'ename', 'fname', 'mname', 'dob', 'nid', 'front']
        else:
            clss=['addr', 'back']
        if debug:
            plt.imshow(img)
            plt.show()
        img=self.get_oriented_data(img)
        if debug:
            plt.imshow(img)
            plt.show()
        
        # check yolo
        img,locs=self.loc(img,clss)
        if img is not None:
            # text detection
            boxes,crops=self.det.detect(img,img,debug=debug)
            if face=="front":
                result=self.process_front(boxes,locs,crops,debug)
                print(result)
            
        
        else:
            print(img_path)
        
        
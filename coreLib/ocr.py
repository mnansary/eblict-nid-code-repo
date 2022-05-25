#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from tkinter.messagebox import NO
from unittest import result

#-------------------------
# imports
#-------------------------
from .yolo import YOLO
from .utils import localize_box,LOG_INFO
from .rotation import auto_correct_image_orientation
from .paddet import Detector
from .checks import processNID,processDob
from paddleocr import PaddleOCR
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import copy
import pandas as pd
from time import time
#-------------------------
# class
#------------------------

    
class OCR(object):
    def __init__(self,weights_dir):
        self.loc=YOLO(os.path.join(weights_dir,"yolo/yolo.onnx"),labels=['sign', 'bname', 'ename', 'fname', 'mname', 'dob', 'nid', 'front', 'addr', 'back'])
        LOG_INFO("Loaded YOLO")
        self.base=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet',use_gpu=True)
        self.det=Detector()
        LOG_INFO("Loaded Paddle")

        
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
        
    #-------------------------------------------------------------------------------------------------------------------------
    # exectutives
    #-------------------------------------------------------------------------------------------------------------------------
    def execite_rotation_fix(self,image):
        result= self.base.ocr(image,rec=False)
        image,mask,angle=auto_correct_image_orientation(image,result)
        # -- coverage
        h,w,_=image.shape
        idx=np.where(mask>0)
        y1,y2,x1,x2 = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        ht=y2-y1
        wt=x2-x1
        coverage=round(((ht*wt)/(h*w))*100,2)  

        rot_info={"operation":"rotation-fix",
                  "optimized-angle":angle,
                  "text-area-coverage":coverage}

        return image,rot_info


    def execite_visibility_check(self):
        viz_info={"operation":"visibility-check",
                  "status":"not-available-yet"}
        return viz_info
    
    #---- place holders--------------------------------
    def get_photo(self):
        return "not-available-yet"
    
    def get_signature(self):
        return "not-available-yet"
    
    def get_addr(self):
        return {"address":"not-available-yet"}

    def get_bangla_info(self):
        bn_info={}
        bn_info["bn-name"]="not-available-yet"
        bn_info["f-name"]="not-available-yet"
        bn_info["m-name"]="not-available-yet"
        return bn_info
        
    #---- place holders--------------------------------
    
    #-------------------------------------------------------------------------------------------------------------------------
    # extractions
    #-------------------------------------------------------------------------------------------------------------------------
    def get_basic_info(self,boxes,locs,crops,debug):
        basic={}
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
        res_eng = self.base.ocr(en_crops,det=False,cls=False)

        ## text fitering
        en_text=[i for i,_ in res_eng]# no conf
        en_name=" ".join(en_text[:len(en_name_crops)])
        en_text=en_text[len(en_name_crops):]
        dob="".join(en_text[:len(dob_crops)])
        en_text=en_text[len(dob_crops):]
        idx="".join(en_text)

        
        basic["en-name"]=en_name
        basic["nid"]=processNID(idx) 
        basic["dob"]=processDob(dob) 

        if basic["nid"] is None:
            basic["nid"]="nid data not found. try agian with different image"
                
        if basic["dob"] is None:
            basic["dob"]="dob data not found.try again with different image"
        return basic 
    
    

            
    def __call__(self,img_path,face,rets,execs,debug=False):
        # return containers
        data={}
        included={}
        executed=[]
        # params
        exec_rot,exec_viz=execs
        ret_bangla,ret_photo,ret_sign=rets
        
        # -----------------------start-----------------------
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
        if face=="front":
            clss=['bname', 'ename', 'fname', 'mname', 'dob', 'nid']
        else:
            clss=['addr']
        if debug:
            plt.imshow(img)
            plt.show()
        
        # visibility place holder
        if exec_viz:
            viz_info=self.execite_visibility_check()
            executed.append(viz_info)
        
        # orientation
        if exec_rot:
            img,rot_info=self.execite_rotation_fix(img)
            executed.append(rot_info)
        
        
        # check yolo
        img,locs=self.loc(img,clss)
        if img is not None:
            if debug:
                plt.imshow(img)
                plt.show()
            
            # text detection
            boxes,crops=self.det.detect(img,self.base)
            if face=="front":
                data["nid-basic-info"]=self.get_basic_info(boxes,locs,crops,debug)
                if ret_bangla:
                    included["bangla-info"]=self.get_bangla_info()
                if ret_photo:
                    included["photo"]=self.get_photo()
                if ret_sign:
                    included["signature"]=self.get_signature()
            else:
                data["nid-back-info"]=self.get_addr()
            # containers
            data["included"]=included
            data["executed"]=executed
            return data 

        else:
            return None
        
        
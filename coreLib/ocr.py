#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from distutils.log import debug

#-------------------------
# imports
#-------------------------
from .yolo import YOLO
from .utils import localize_box,LOG_INFO,download
from .rotation import auto_correct_image_orientation
from .paddet import Detector
from .bnocr import BanglaOCR
from .front import get_bangla_info,get_basic_info
from .back import get_addr,reformat_back_data,get_regional_box_crops
from paddleocr import PaddleOCR
import os
import cv2
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
#-------------------------
# class
#------------------------

    
class OCR(object):
    def __init__(self,   
                 yolo_onnx="weights/yolo.onnx",
                 yolo_gid="1gbCGRwZ6H0TO-ddd4IBPFqCmnEaWH-z7",
                 bnocr_onnx="weights/bnocr.onnx",
                 bnocr_gid="1YwpcDJmeO5mXlPDj1K0hkUobpwGaq3YA"):
        
        if not os.path.exists(yolo_onnx):
            download(yolo_gid,yolo_onnx)
        self.loc=YOLO(yolo_onnx,
                      labels=['sign', 'bname', 'ename', 'fname', 
                              'mname', 'dob', 'nid', 'front', 'addr', 'back'])
        LOG_INFO("Loaded YOLO")
        
        if not os.path.exists(bnocr_onnx):
            download(bnocr_gid,bnocr_onnx)
        self.bnocr=BanglaOCR(bnocr_onnx)
        LOG_INFO("Loaded Bangla Model")        
        
        self.base=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet',use_gpu=True)
        self.line=PaddleOCR(use_angle_cls=True, lang='hi',use_gpu=False)
        self.det=Detector()
        LOG_INFO("Loaded Paddle")

        
    def process_boxes(self,text_boxes,region_dict,includes):
        '''
            keeps relevant boxes with respect to region
            args:
                text_boxes  :  detected text boxes by the detector
                region_dict :  key,value pair dictionary of region_bbox and field info 
                               => {"field_name":[x_min,y_min,x_max,y_max]}
                includes    :  list of fields to be included 
        '''
        # extract region boxes
        region_boxes=[]
        region_fields=[]
        for k,v in region_dict.items():
            if k in includes:
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
    #-------------------------------------------------------------------------------------------------------------------------
    # exectutives
    #-------------------------------------------------------------------------------------------------------------------------
    def get_coverage(self,image,mask):
        # -- coverage
        h,w,_=image.shape
        idx=np.where(mask>0)
        y1,y2,x1,x2 = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        ht=y2-y1
        wt=x2-x1
        coverage=round(((ht*wt)/(h*w))*100,2)
        return coverage  


    def execute_rotation_fix(self,image,mask):
        image,mask,angle=auto_correct_image_orientation(image,mask)
        rot_info={"operation":"rotation-fix",
                  "optimized-angle":angle,
                  "text-area-coverage":self.get_coverage(image,mask)}

        return image,rot_info

            
    def __call__(self,
                 img_path,
                 face,
                 ret_bangla,
                 exec_rot,
                 coverage_thresh=30,
                 debug=False):
        # return containers
        data={}
        included={}
        executed=[]
        # -----------------------start-----------------------
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        src=np.copy(img)
    
        if face=="front":
            clss=['bname', 'ename', 'fname', 'mname', 'dob', 'nid']
        else:
            clss=['addr','back']
        
        try:
            # orientation
            if exec_rot:
                # mask
                mask,_,_=self.det.detect(img,self.base)
                img,rot_info=self.execute_rotation_fix(img,mask)
                executed.append(rot_info)
        except Exception as erot:
            return "text-region-missing"
            
        # check yolo
        img,locs,founds=self.loc(img,clss,face)

        if img is None:
            if len(founds)==0:
                return "no-fields"
            else:
                if not exec_rot:
                    mask,_,_=self.det.detect(src,self.base)
                coverage=self.get_coverage(src,mask)
                if coverage > coverage_thresh:
                    return "loc-error"
                else:
                    return f"coverage-error#{coverage}"
        else:
            if face=="front":
                # text detection [word-based]
                _,boxes,crops=self.det.detect(img,self.base)
                # sorted box dictionary [clss based box_dict]
                box_dict=self.process_boxes(boxes,locs,clss)        
                data["nid-basic-info"]=get_basic_info(box_dict,crops,self.base)
                if ret_bangla:
                    included["bangla-info"]=get_bangla_info(box_dict,crops,self.bnocr)
            
            else:
                # crop image if both front and back is present
                if locs['dob'] is not None and locs["nid"] is not None:
                    img,locs=reformat_back_data(img,locs)
                # text detection 
                line_mask,line_boxes,_=self.det.detect(img,self.line)
                word_mask,word_boxes,word_crops=self.det.detect(img,self.base)
                # regional text
                line_boxes,word_boxes,crops=get_regional_box_crops(line_mask,line_boxes,word_mask,word_boxes,word_crops)
                texts=self.bnocr(crops)
                # get address
                df=get_addr(word_boxes,line_boxes,texts)
                return df
                #data["nid-back-info"]=get_addr(word_boxes,line_boxes,texts)
                
                
            # containers
            data["included"]=included
            data["executed"]=executed
            return data 


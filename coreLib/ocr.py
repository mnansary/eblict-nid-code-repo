#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#-------------------------
# imports
#-------------------------
from .yolo import YOLO
from .utils import localize_box,LOG_INFO,download
from .rotation import auto_correct_image_orientation
from .paddet import Detector
from .bnocr import BanglaOCR
from .checks import processNID,processDob
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

    def create_line_ref_mask(self,image,boxes):
        h,w,_=image.shape
        mask=np.zeros((h,w))
        for box in boxes:
            box=[int(b) for b in box]
            x1,y1,x2,y2=box
            h=y2-y1
            y2=y1+1+h//8
            cv2.rectangle(mask,(x1,y1),(x2,y2),(255, 255, 255), -1)  
        return mask

    def create_line_refs(self,img,boxes,box_ids):
        # boxes
        word_orgs=[]
        for bno in range(len(boxes)):
            tmp_box = copy.deepcopy(boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            word_orgs.append([x1,y1,x2,y2])
        
        # references
        line_refs=[]
        mask=self.create_line_ref_mask(img,word_orgs)
        # Create rectangular structuring element and dilate
        mask=mask*255
        mask=mask.astype("uint8")
        h,w=mask.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//2,1))
        dilate = cv2.dilate(mask, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            line_refs.append([x,y,x+w,y+h])
        line_refs = sorted(line_refs, key=lambda x: (x[1], x[0]))
        # sort boxed
        data=pd.DataFrame({"ref_box":word_orgs,"ref_ids":box_ids})
        # detect field
        data["lines"]=data.ref_box.apply(lambda x:localize_box(x,line_refs))
        data.dropna(inplace=True) 
        data["lines"]=data.lines.apply(lambda x:int(x))
        
        text_dict=[]
        for line in data.lines.unique():
            ldf=data.loc[data.lines==line]
            _boxes=ldf.ref_box.tolist()
            _bids=ldf.ref_ids.tolist()
            _,bids=zip(*sorted(zip(_boxes,_bids),key=lambda x: x[0][0]))
            for idx,bid in enumerate(bids):
                _dict={"line_no":line,"word_no":idx,"crop_id":bid}
                text_dict.append(_dict)
        df=pd.DataFrame(text_dict)
        return df
        

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

    #-------------------------------------------------------------------------------------------------------------------------
    # extractions
    #-------------------------------------------------------------------------------------------------------------------------
    def get_basic_info(self,box_dict,crops):
        basic={}
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
    
    def get_bangla_info(self,box_dict,crops):
        bn_info={}
        # bangla ocr
        bn_keys=["bname","fname","mname"]
        ## b-name
        b_name =box_dict[bn_keys[0]]
        b_crops=[crops[i] for i in b_name]
        ## f-name
        f_name = box_dict[bn_keys[1]]
        f_crops=[crops[i] for i in f_name]
        ## m-name
        m_name = box_dict[bn_keys[2]]
        m_crops=[crops[i] for i in m_name]
        
        crops=b_crops+f_crops+m_crops
        texts= self.bnocr(crops)

        ## text fitering
        b_name=" ".join(texts[:len(b_crops)])
        texts=texts[len(b_crops):]
        f_name=" ".join(texts[:len(f_crops)])
        texts=texts[len(f_crops):]
        m_name=" ".join(texts)
        bn_info["bn-name"]=b_name
        bn_info["f-name"]=f_name
        bn_info["m-name"]=m_name
        return bn_info
    
    def get_addr(self,crops,tdf):
        cids=tdf.crop_id.tolist()
        bn_crops=[crops[i] for i in cids]
        texts=self.bnocr(bn_crops)
        tdf["text"]=texts

        # get text
        line_max=max(tdf.line_no.tolist())
        addr=""
        for l in range(0,line_max+1):
            ldf=tdf.loc[tdf.line_no==l]
            ldf=ldf.sort_values('word_no')
            ltext=" ".join(ldf.text.tolist())+"\n"
            addr+=ltext

        return {"address":addr}


            
    def __call__(self,
                 img_path,
                 face,
                 ret_bangla,
                 exec_rot,
                 coverage_thresh=30):
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
            clss=['addr']
        
        try:
            # orientation
            if exec_rot:
                # mask
                mask=self.det.detect(img,self.base,ret_mask=True)
                img,rot_info=self.execute_rotation_fix(img,mask)
                executed.append(rot_info)
        except Exception as erot:
            return "text-region-missing"
            
        # check yolo
        img,locs,founds=self.loc(img,clss)
        if img is None:
            if len(founds)==0:
                return "no-fields"
            else:
                if not exec_rot:
                    mask=self.det.detect(src,self.base,ret_mask=True)
                coverage=self.get_coverage(src,mask)
                if coverage > coverage_thresh:
                    return "loc-error"
                else:
                    return f"coverage-error#{coverage}"
        else:
            # text detection
            boxes,crops=self.det.detect(img,self.base)
            # sorted box dictionary
            box_dict=self.process_boxes(boxes,locs,clss)    
            if face=="front":
                data["nid-basic-info"]=self.get_basic_info(box_dict,crops)
            
                if ret_bangla:
                    included["bangla-info"]=self.get_bangla_info(box_dict,crops)
            
            else:
                if 'addr' not in box_dict.keys():
                    return "addr-not-located"
                if len(box_dict["addr"])==0:
                    return "addr-not-located"
                # get address boxes
                box_ids=box_dict["addr"]
                addr_boxes=[boxes[int(i)] for i in box_ids]
                # create box dataframe
                tdf=self.create_line_refs(img,addr_boxes,box_ids)            
                data["nid-back-info"]=self.get_addr(crops,tdf)
                
                
            # containers
            data["included"]=included
            data["executed"]=executed
            return data 


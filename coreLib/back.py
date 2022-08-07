#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import pandas as pd
import numpy as np
import copy
from .utils import localize_box


def reformat_back_data(img,locs):
    x1a,y1a,x2a,y2a=locs["addr"]
    x1b,y1b,x2b,y2b=locs["back"]
    dx1,dy1,dx2,dy2=0,max(0,y1a-y1b),x2a-x1b,y2a-y1b
    locs["addr"]=[dx1,dy1,dx2,dy2]
    img=img[y1b:y2b,x1b:x2b]
    return img,locs


def get_regional_box_crops(line_mask,line_boxes,word_mask,word_boxes,word_crops):
    # create region_mask
    idx=np.where(line_mask>0)
    y1,y2,x1,x2 = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    line_reg_mask=line_mask[y1:y2,x1:x2]
    word_reg_mask=word_mask[y1:y2,x1:x2]
    w_reg=x2-x1
    # detect card type
    idx=np.where(line_reg_mask==np.unique(line_reg_mask)[1])
    y1,y2,x1,x2 = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    if x2-x1<w_reg*0.8:
        # smart card
        reg_row = np.where(np. all((word_reg_mask == 0), axis=1)==True)[0].min()
        line_reg_mask=line_reg_mask[:reg_row,:]
        word_reg_mask=word_reg_mask[:reg_row,:]
        
        
    else:
        # old nid card
        h,w=word_reg_mask.shape
        line_reg_mask=line_reg_mask[:h//2,:]
        word_reg_mask=word_reg_mask[:h//2,:]
    # create line ids
    line_ids=np.unique(line_reg_mask)[1:]
    # create word ids
    word_ids=np.unique(word_reg_mask)[1:]
    line_boxes=[line_boxes[int(i)-1] for i in line_ids]
    word_boxes=[word_boxes[int(i)-1] for i in word_ids]
    word_crops=[word_crops[int(i)-1] for i in word_ids]

    return line_boxes,word_boxes,word_crops

def create_sorted_data(word_boxes,line_boxes,texts):
    # word refs
    word_refs=[]
    for bno in range(len(word_boxes)):
        tmp_box = copy.deepcopy(word_boxes[bno])
        x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
        y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
        word_refs.append([x1,y1,x2,y2])
    
    # line refs
    line_refs=[]
    for bno in range(len(line_boxes)):
        tmp_box = copy.deepcopy(line_boxes[bno])
        x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
        y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
        line_refs.append([x1,y1,x2,y2])
    
    # sorted line refs[top to bottom]
    line_refs = sorted(line_refs, key=lambda x: (x[1], x[0]))
    
    # sort boxed
    data=pd.DataFrame({"boxes":word_refs,"texts":texts})
    # detect lines
    data["lines"]=data.boxes.apply(lambda x:localize_box(x,line_refs))
    data.dropna(inplace=True) 
    data["lines"]=data.lines.apply(lambda x:int(x))
    # sorted dataframe
    text_dict=[]
    for line in data.lines.unique():
        ldf=data.loc[data.lines==line]
        wrefs=ldf.boxes.tolist()
        _texts=ldf.texts.tolist()
        _,_texts=zip(*sorted(zip(wrefs,_texts),key=lambda x: x[0][0]))
        for idx,_text in enumerate(_texts):
            _dict={"line_no":line,"word_no":idx,"text":_text}
            text_dict.append(_dict)
    df=pd.DataFrame(text_dict)
    return df

def get_addr(word_boxes,line_boxes,texts):
    df=create_sorted_data(word_boxes,line_boxes,texts)
    # get text
    line_max=max(df.line_no.tolist())
    words=[]
    for l in range(0,line_max+1):
        ldf=df.loc[df.line_no==l]
        ldf=ldf.sort_values('word_no')
        words+=ldf.text.tolist()
    return " ".join(words)

    #return {"address":addr}








    
#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
from .checks import processNID,processDob

def get_basic_info(box_dict,crops,model):
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
    res_eng = model.ocr(en_crops,det=False,cls=False)

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

def get_bangla_info(box_dict,crops,bnocr):
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
    texts= bnocr(crops)

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
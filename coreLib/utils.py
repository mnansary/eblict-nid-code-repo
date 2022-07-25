#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import gdown
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
def download(id,save_dir):
    gdown.download(id=id,output=save_dir,quiet=False)
#------------------------------------
# region-utils 
#-------------------------------------
def intersection(boxA, boxB):
    # boxA=ref
    # boxB=sig
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    x_min,y_min,x_max,y_max=boxB
    selfArea  = abs((y_max-y_min)*(x_max-x_min))
    return interArea/selfArea
#---------------------------------------------------------------
def localize_box(box,region_boxes):
    '''
        lambda localization
    '''
    max_ival=0
    box_id=None
    for idx,region_box in enumerate(region_boxes):
        ival=intersection(region_box,box)
        if ival==1:
            return idx
        if ival>max_ival:
            max_ival=ival
            box_id=idx
    if max_ival==0:
        return None
    return box_id
#---------------------------------------------------------------
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
# ---------------------------------------------------------
class EngOCR(object):
    def __init__(self,batch_size=128):
        self.model=recognition_predictor('crnn_vgg16_bn',pretrained=True,batch_size=batch_size)
    def __call__(self,img_list):
        texts=self.model(img_list)
        return texts
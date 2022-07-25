#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
from coreLib.ocr import OCR
from pprint import pprint
ocr=OCR()
img_path="tests/test.jpg"
data=ocr(img_path,"back",True,True)
pprint(data)
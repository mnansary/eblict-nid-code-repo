#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#------------------------------------------------------------------------------------
from difflib import get_close_matches
from datetime import datetime
import re

months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
numse=[str(i) for i in range(10)]
#------------------------------------------------------------------------------------

def checkYear(year):
        if int(year)>=1900 and int(year)<datetime.today().year:
            return True
        else:
            return False

def checkDay(day):
    if int(day)>0 and int(day)<32:
        return True
    else:
        return False

def checkMonth(text,year,day):
    pattern=f"{day}(.*){year}"
    r=re.search(pattern,text)
    if r is None:
        return None
    else:
        month=r.group(1)
        if len(month)>0:
            if month in months:
                return month
            else:
                mlist=get_close_matches(month,months,n=1)
                if len(mlist)==0:
                    return None
                else:
                    month=mlist[0]
                    return month
        else:
            
            return None

def processDob(text):
    text=text.replace(" ",'')
    nums=[i for i in text if i in numse]
    # check if any number is present
    if len(nums)>0:
        num="".join(nums)
        year=num[-4:]
        day=num[:2]
        if checkDay(day) and checkYear(year):
            month=checkMonth(text,year,day)
            if month is not None:
                dob=day+month+year
                #dob=datetime.strptime(dob, "%d%b%Y").strftime("%Y-%m-%d")         
                return dob
            else:
                return None 
        else:
            return None
    else:
        return None

def processNID(text):
    text=text.replace(" ",'')
    num="".join([i for i in text if i in numse])
    
    if len(num) in [10,13,17]:
        _text=text.split(num[:3])[-1]
        if num[:3]+_text==num:
            return num
        else:
            return None
    else:
        return None
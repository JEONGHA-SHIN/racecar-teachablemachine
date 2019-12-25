#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:17:02 2019

@author: racecar
"""


import cv2

cap = cv2.VideoCapture(0)

cap.set(3,320)
cap.set(4,240)

while True:
    ret,frame = cap.read()
    
    if ret:
        cv2.imshow('video',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
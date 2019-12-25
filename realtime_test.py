#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:21:38 2019

@author: racecar
"""
import tensorflow.keras
from PIL import Image
import numpy as np
import cv2

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)



while True:
    ret, frame = cap.read()
    if ret:
        #cv2.imshow('video', frame)
        new_image = frame.copy()
        
        if new_image.ndim == 2:
            pass
        elif new_image.shape[2] == 3:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        #elif new_image.shape[2] == 4:
        #    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
            
        im_pil = Image.fromarray(new_image)
        image = im_pil
    
        # Make sure to resize all images to 224, 224 otherwise they won't fit in the array
        image = image.resize((224, 224))
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) -1
        na=normalized_image_array[:,:,0:3]
        # Load the image into the array
        data[0] = na
        
        # run the inference
        prediction = model.predict(data)
        print(prediction)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()


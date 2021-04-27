import base64
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
#from matplotlib import pyplot as plt
import imutils
import sys
import os
import numpy 

def base64toCvImage(base64Input):
    """Method for converting from base64image to
    a picture that cv2 can show and use.
    Mostly code from the old function christina wrote
    in the AzureFunctionPython repo"""
    try:
        base64Data = base64Input.partition(",")[2]
        byteData = base64.b64decode(base64Data+"===")#The equals are extra padding(?)
        imageData = BytesIO(byteData)
        pilImg = Image.open(imageData)
        npImage = np.array(pilImg)
        cvImage = cv2.cvtColor(npImage, cv2.COLOR_RGBA2BGRA)
        return cvImage
    except Exception as ex:
        print(ex)
        return None

def model(cvImage):
    levelPercent = 0
    try:
        container_gray = cv2.split(cvImage)[0]
        container_gray = cv2.GaussianBlur(container_gray, (7, 7), 0) # blur image
        (T, container_threshold) = cv2.threshold(container_gray, 50, 255, cv2.THRESH_BINARY_INV)# manual threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        container_open = cv2.morphologyEx(container_threshold, cv2.MORPH_OPEN, kernel) # apply opening operation
        contours = cv2.findContours(container_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# find all contours
        contours = imutils.grab_contours(contours)
        
        areas = [cv2.contourArea(contour) for contour in contours]   # sort contours by area
        (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
        #container_clone = cvImage.copy()
        # draw bounding box, calculate aspect and display decision
        cv2.drawContours(container_threshold,contours,-1,(255,0,0),2)
        (x, y, w, h) = cv2.boundingRect(contours[-1])
        cv2.rectangle(container_threshold, (x, y), (x + w, y + h), (0, 0, 255), 2)


        containerHeightInPixels = int(w/0.420) #Change this value to width/Hight of container to find correct ratio
        levelPercent = round((h/containerHeightInPixels)*100,1)

        backtorgb = cv2.cvtColor(container_threshold,cv2.COLOR_GRAY2RGB)
        cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 0, 255), 4)
        #cv2.putText(backtorgb, f"{levelPercent}%", (x + 10, y + 60), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

        noegreier, buffer = cv2.imencode('.png',backtorgb)
        base64string = toBase64(buffer)
        # cv2.putText(container_clone, f"{levelPercent}%", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        #cv2.namedWindow("output4", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions                    
        #cv2.imshow("output4", container_clone)                            # Show image
        #cv2.waitKey(0)
        #print(levelPercent)
        return {"name":"opencv","level":levelPercent,"image":base64string}
    except:
        return {"name":"opencv","level":"Error","image":"data:image/png;base64,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s7X2YJx8bYK0wEBpgkpZbReS72iQcXq06PFYT02QxcCdyRNsEFys0Gdu7oe+XN00SKS+XPraGDVBvgG+PeaCIC4F+G6ZmV4ulRpktm7JqkdYE+QOMDcbmK1BcDcTTRddNtMbZYENgqDvbVltEN2Ga/S7IjUIjv2A5hH3zHyhPkgTBPev2LpaAWnBSoNsAk5MvumC/CjBGiEzuNHWvHPmd0SDdKy6ZaFxrj7sNtBQHrggh/SWFcpPDyf8PE/j2yC/VG21nJmuPkKDJOoLa4M8Sr6BW/JoePvciOPm+NUgR/SgMYpOarm/C1Ku76kNMlVPuYWywtPqnlw3x68GWdHZnYo1jpqfGQHGaxCsPOyCIO9XoE93kTBz5tkgOKVfqS5W1f+mQXB2hgvSacnv0N8QkQPmttkg+ENfHHXKlOsPrgmCoD9dkNNm3z5IE9mpZ5kLgt1/mXLcNfX3XRDcv+aC+JlzcKz+VpaW6KwN0nTA37a+Vr6hjRtUdfLtoe7z0PytnnIV9eonVXfcmpe3bf69vrZj/qCZO7jqTpma/PwG1XP6t86ir/unqH49EREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREdH/AfgHOa1yMaZPehkAAAAASUVORK5CYII="}

def predict(base64Image):
    cvImage = base64toCvImage(base64Image)
    level = model(cvImage)
    return level
def toBase64(image):
    encoded = f'data:image/png;base64,{base64.b64encode(image).decode()}'
    return encoded
    #base64string=base64.b64encode(buffer)
    #base64string="data:image/png;base64,"+base64string
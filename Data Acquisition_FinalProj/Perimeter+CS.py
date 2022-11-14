# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:10:51 2020

@author: ar54482
"""

import os
import cv2
from PIL import Image
from skimage.measure import find_contours
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#N-Stress
#CentroidSize

#define distance
def distance(x1,y1,x2,y2):
     d = (x1 - x2)**2 + (y1-y2)**2 
     return d

CSN1=[] #list to store centroid size 
Filename=[] #store filenames in the list

path=(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\C\ControlP_shapes\segmented\All shapes")

for filename in os.listdir(path):
            
    img = Image.open(os.path.join(path,filename)).convert('L')
    #red, green, blue = img.split()
    coords = find_contours(img,200)
    
    if len(coords)!=0:
        coords = coords[0]
        
    
        x=[i[0] for i in coords]  
       
        y=[i[1] for i in coords] 
        
          #centroid
        xc=sum(x)/len(x)
        
        yc=sum(y)/len(y)
        
        #xnew=x-xc
        #ynew=y-yc #make centroid as (0,0)
        
       
        coords_n=[]
        
        for i in range(0,len(coords)):
            
            coords_n.append([x[i],y[i]])  #store the new coordinates in a list
            
    
#    N=[] #list to store count of each root hair
    D=[] #list to store the distances
    
    
    #centroid length(cl)
    for (x,y) in coords_n:
    
        k=distance(x,y,xc,yc)
        D.append(k)
    

        
    cs=(sum(D))**.5
    CSN1.append(cs)

    Filename.append(filename)

print(len(Filename))
print(len(CSN1))

data={'ID': Filename, 'Area':CSN1}
NS=pd.DataFrame(data)

#Perimeter

path=(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\C\ControlP_shapes\segmented\All shapes")

LN=[] #list to store the perimeter

for filename in os.listdir(path):

  
    imgcolor = cv2.imread(os.path.join(path,filename))
    
    height, width, channels = imgcolor.shape
    

    gray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    _, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    red = 0
    ## Draw the contour 
    for c in contours:
        
        red = red + 10 
    #fill the connected contours
        contours_img = cv2.drawContours(imgcolor, [c], -1, (255, 255, red),4)
 
    #os.chdir(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\contours")
    #cv2.imwrite(filename,contours_img)
    #cv2.destroyAllWindows()
   
    l=cv2.arcLength(contours[0],closed=True)
    LN.append(l)  ######LN!!!

NS['Perimeter']=LN

NS.to_csv(r"C:\Users\ar54482\Desktop\DataCellShapeC.csv", index = False, header = True)


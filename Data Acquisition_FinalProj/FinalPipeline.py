#Nitrogen-stress

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:10:51 2020

@author: ar54482
"""
#Perimeter+Area
import os
import cv2
from PIL import Image
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import find_contours
from skimage import io
from skimage.color import convert_colorspace
from scipy.interpolate import splev, splprep
import numpy as np
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression
import math
import seaborn as sns
import statistics
from scipy.spatial import distance
from skimage import img_as_bool, io, color, morphology
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import json

#N-Stress
#CentroidSize

#define distance
def distance(x1,y1,x2,y2):
     d = (x1 - x2)**2 + (y1-y2)**2 
     return d

CSN1=[] #list to store centroid size 
Filename=[] #store filenames in the list

path=(r"path to folder with segmented & extracted individual roothairs")

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

path=(r"path to folder with segmented & extracted individual roothairs")

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

NS.to_csv(r"path to excel sheet where you want to store your calculated values", index = False, header = True)

#Pre-alignment for ICP

#specify the path
path=(r"path to folder with segmented & extracted individual roothairs")


for filename in os.listdir(path):
    
# Read the image you want the connected components extracted from
    img = cv2.imread(os.path.join(path,filename),0)
    #img = cv2.imread(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\C\ControlP_shapes\segmented\All shapes\28.jpg")
    ##img = cv2.imread(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Trial2\2.jpg',0)
    ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    cv2.imwrite(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\gray.jpg",thresh_img)
    
    imgdata = Image.open(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\gray.jpg")
   
    #imgdata = Image.open(os.path.join(path,filename))
    width,height=imgdata.size
    
    coords=[] #store the white coordinates here
    #count=0
    pix=[]
    boundary=[]
    
    for i in range(width):
        for j in range(height):
            coordinate=i,j
            p=imgdata.getpixel((i,j))
            pix.append(p)
    
            if p==255:      
                coords.append((i,j)) 
                p1=imgdata.getpixel((i,j-1))
                p2=imgdata.getpixel((i-1,j))
                p3=imgdata.getpixel((i,j+1))
                p4=imgdata.getpixel((i+1,j))
    
                if p1==0 or p2==0 or p3==0 or p4==0:
                    boundary.append((i,j))    
    
    x=[i[0] for i in coords]  
    y=[i[1] for i in coords] 
    
    xb=[i[0] for i in boundary]  
    yb=[i[1] for i in boundary] 
    
    
    
    #determine the centroid
    xcd=sum(x)/len(x)
    ycd=sum(y)/len(y)
    
    ##calculate the centroid size
     
    coords_n=[]
    D=[]   #store the distances of boundary points from the centroid
     
    #define distance
    def distance(x1,y1,x2,y2):
     d = (x1 - x2)**2 + (y1-y2)**2 
     return d
     
    for i in range(0,len(coords)):
        coords_n.append([x[i],y[i]])
     
    for (a,b) in coords_n:
        k=distance(a,b,xcd,ycd)
        D.append(k)
     
    cs=(sum(D))**.5/len(D)
    
    
    coords_d=[]
    coords_b=[]
            
    for i in range(0,len(coords)):
        
        coords_d.append(((x[i]-xcd)/cs,(y[i]-ycd)/cs))
    
    for i in range(0,len(boundary)):
        
        coords_b.append(((xb[i]-xcd)/cs,(yb[i]-ycd)/cs))
    
    x1=[i[0] for i in coords_d]   
    y1=[i[1] for i in coords_d] 
    
    x2=[i[0] for i in coords_b]   
    y2=[i[1] for i in coords_b]
    
    
    
    fig, ax = plt.subplots()
    plt.scatter(x1, y1, marker='x',color="white")
    ax.set_facecolor('black')
    ax.set_ylim(-350,350)
    ax.set_xlim(-350,350)
    #plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes\Trial\8a.jpg")
    
    fig, ax = plt.subplots()
    
    
    
    plt.scatter(x1, y1, marker='x',color="white")
    plt.scatter(x2, y2, marker='o',color="red")
    
    
    ax.set_ylim(-350,350)
    ax.set_xlim(-350,350)
    ax.set_facecolor('black')
    #ax.set_axis_off()
    
       # plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes\Trial\temp.jpg")
    
    
    
    """
    Linear regression model
    
    """
    from sklearn.linear_model import LinearRegression
    
    #convert coordinates into array
    X=np.asarray(x1).reshape((-1,1))
    Y=np.asarray(y1)
    
    #Build a model for linear regression and fit your data to calculate the weights
    model = LinearRegression().fit(X,Y)
    
    r_sq=model.score(X,Y)
    
    #print('coefficient of determination:', r_sq)
    
    #print('intercept:', model.intercept_)
    
    #print('slope:', model.coef_)
    
    a = np.linspace(-350,350,50)
    b=model.coef_*(a) + model.intercept_
    
    I=math.degrees(math.atan(model.coef_))
    Ir=math.atan(model.coef_)
    print('Ir',Ir)
    print('model.coef_',model.coef_)
    
    #print(model.coef_)
    if model.coef_>0:
        
    #Decide on which side of the line the point is
    
        L=[]
        R=[]
        
        a1=-300
        a2=300
        
        b1=model.coef_*(a1) + model.intercept_
        b2=model.coef_*(a2) + model.intercept_
        
        
        plt.plot(a,b, '-r')
        
        for (a0,b0) in coords_b:
            v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
            
            if v>0:
                R.append((a0,b0))
            if v<0:
                L.append((a0,b0))
        
        #check (0,0) is on which side
        (a0,b0)=(-100,100)
        v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
      #  print(v)
        
      #  print(b1,b2)
        
    #    if v>0:
    #        print("Right")
    #    
    #    if v<0:
    #        print("Left")
        
                
        #find distance of the point from line
        DL=[]   
        XL=[]
        xl=0     
        for (a0,b0) in L:
            dl=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
            
            DL.append(dl[0])
            
            xl=xl+1
            XL.append(xl)
        
        
        
        DR=[]
        XR=[]
        xr=0 
        for (a0,b0) in R:
            dr=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
            DR.append(dr[0])
            xr=xr+1
            XR.append(xr)
       
        Lm=statistics.mean(DL)
        
        Rm=statistics.mean(DR)
        
        if Lm>Rm:
            H="Left"
        if Rm>Lm:
            H="Right"
            

    
    
    
    if model.coef_<0:
        
        L=[]
        R=[]
        
        a1=-200
        a2=200
        
        b1=model.coef_*(a1) + model.intercept_
        b2=model.coef_*(a2) + model.intercept_
        
        plt.plot(a,b, '-r')
        
        for (a0,b0) in coords_b:
            v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
            
            if v>0:
                L.append((a0,b0))
            if v<0:
                R.append((a0,b0))
        
        #check (0,0) is on which side
        (a0,b0)=(150,150)
        v=((a0 - a1)*(b2 - b1)) - ((a2 - a1)*(b0 - b1))
        #print(v)
        #print(b1,b2)
        
    #    if v>0:
    #        print("Left")
    #    
    #    if v<0:
    #        print("Right")
        
                
        #find distance of the point from line
        DL=[]   
        XL=[]
        xl=0     
        for (a0,b0) in L:
            dl=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
           
            DL.append(dl[0])
            
            xl=xl+1
            XL.append(xl)
        
        
        
        DR=[]
        XR=[]
        xr=0 
        for (a0,b0) in R:
            dr=(abs((model.coef_*a0)-b0+model.intercept_))/math.sqrt(model.coef_**2+1)
            DR.append(dr[0])
            xr=xr+1
            XR.append(xr)
        
        Lm=statistics.mean(DL)
        
        Rm=statistics.mean(DR)
        
        if Lm>Rm:
            H="Left"
           # print("Left")
        if Rm>Lm:
            H="Right"
            #print("Right")
    print(H)
            

    
    if H =="Left":
        
             
        flip=[]    
        b=len(coords_d)    
        for (x3,y3) in coords_d:
            xp=width-x3-1
            flip.append((xp,y3))
    
        
        x3=[i[0] for i in flip]   
        y3=[i[1] for i in flip] 
    
        x3cd=sum(x3)/len(x3)
        y3cd=sum(y3)/len(y3) 
        
        for i in range(0,len(x3)):
        
            flip.append((x3[i]-x3cd,y3[i]-y3cd))
            
        x4=[i[0] for i in flip]   
        y4=[i[1] for i in flip]
        
        fig, ax = plt.subplots()
    
        plt.scatter(x4, y4, marker='x',color="white")    
        
        ax.set_ylim(-350,350)
        ax.set_xlim(-350,350)
        ax.set_facecolor('black')
       # ax.set_axis_off()
       #get the flipped slope value
        X4=np.asarray(x4).reshape((-1,1))
        Y4=np.asarray(y4)
    
        #Build a model for linear regression and fit your data to calculate the weights
        model4 = LinearRegression().fit(X4,Y4)
    
        r_sq=model.score(X4,Y4)
    
        #print('coefficient of determination:', r_sq)
    
        #print('intercept:', model.intercept_)
    
        #print('slope:', model.coef_)
    
        a4 = np.linspace(-350,350,50)
        b4=model4.coef_*(a4) + model4.intercept_
    
        I4=math.degrees(math.atan(model4.coef_))
        Ir4=math.atan(model4.coef_)
        print('Ir4',Ir4)
    
       
       
    #Align its orientation to the mean shape (sd5_20x12L.png44.jpg in the Trial folder)
        
    #model slope in degrees
    S=math.degrees(math.atan(0.87791686))
    Sr=math.atan(0.87791686)
    print('S',S)
    print('Sr',Sr)
    #model slope in radians
    
    #data slope
    #D=abs(I)
    #print('D',D)    
    
    #For right hooked:
    #multiply by rotation matrix with angle as the difference between S and D
    
    coords_d1=[]
    
    
    if H=="Right":
        
        
        if abs(Ir)>Sr:
            
            a=(Ir-Sr)
            r=[[math.cos(a), math.sin(a)],[(-math.sin(a)), math.cos(a)]]
            mr=np.matrix(r)
        
            
            for x in coords_d:
           
                xx=np.matrix(x)  #(D)
                
                mxt=xx.transpose()          #M=(R*D)
                
                rx=mr*(mxt)
                #convert trx into [x,y]
                rx=rx.tolist()
                #print (txnew)
                coords_d1.append(rx)
                
        
        #print(coords_d1[0])
              
        #print(coords_d1[0][0][0])    
        #print(coords_d1[0][1][0])     
            
            coords_d2=[]  #remove the lists in list
            
            
            for i in range(0,len(coords_d1)):
                
                coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
            
            
            x5=[i[0] for i in coords_d2]   
            y5=[i[1] for i in coords_d2]
            
               
            fig, ax = plt.subplots()
            
            plt.scatter(x5, y5, marker='x',color="white")   
            ax.set_facecolor('black')
            
            ax.set_ylim(-350,350)
            ax.set_xlim(-350,350)
            os.chdir(r'path to the folder where you want to save the pre-aligned figures')
            plt.savefig(filename)
            os.chdir(r"path to working directory")
    
            
        if abs(Ir)<Sr:
            a=(Ir-Sr)
            r=[[math.cos(a),(-math.sin(a))],[math.sin(a), math.cos(a)]]
            mr=np.matrix(r)
            
            
                
            for x in coords_d:
           
                xx=np.matrix(x)  #(D)
                
                mxt=xx.transpose()          #M=(R*D)
                
                rx=mr*(mxt)
                #convert trx into [x,y]
                rx=rx.tolist()
                #print (txnew)
                coords_d1.append(rx)
                
                
                
            coords_d2=[]  #remove the lists in list
           
            for i in range(0,len(coords_d1)):
                coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
                
        
                x5=[i[0] for i in coords_d2]   
                y5=[i[1] for i in coords_d2]
                
            
                
           
            fig, ax = plt.subplots()
        
            plt.scatter(x5, y5, marker='x',color="white")   
            ax.set_facecolor('black')
        
            ax.set_ylim(-350,350)
            ax.set_xlim(-350,350)
            os.chdir(r'path to the folder where you want to save the pre-aligned figures")
            plt.savefig(filename)
            os.chdir(r"path to working directory")
    
        
    if H=="Left":
        
        if abs(Ir4)>Sr:
            
            a=(Ir4-Sr)
            r=[[math.cos(a), (-math.sin(a))],[math.sin(a), math.cos(a)]]
            mr=np.matrix(r)
            
            for x in flip:
           
                xx=np.matrix(x)  #(D)
                
                mxt=xx.transpose()          #M=(R*D)
                
                rx=mr*(mxt)
                #convert trx into [x,y]
                rx=rx.tolist()
                #print (txnew)
                coords_d1.append(rx)
                
        
        #print(coords_d1[0])
              
        #print(coords_d1[0][0][0])    
        #print(coords_d1[0][1][0])     
            
            coords_d2=[]  #remove the lists in list
            
            
            for i in range(0,len(coords_d1)):
                
                coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
            
            
            x5=[i[0] for i in coords_d2]   
            y5=[i[1] for i in coords_d2]
            
               
            fig, ax = plt.subplots()
            
            plt.scatter(x5, y5, marker='x',color="white")   
            ax.set_facecolor('black')
            
            ax.set_ylim(-350,350)
            ax.set_xlim(-350,350)
            os.chdir(r'path to the folder where you want to save the pre-aligned figures')
            #plt.savefig(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned\1.jpg')
            plt.savefig(filename)
            os.chdir(r"path to working directory")
            
            
       
            
    
    
        if abs(Ir4)<Sr:
            
            a=2*(Sr-Ir4)
            
            r=[[math.cos(a),(math.sin(a))],[(-math.sin(a)), math.cos(a)]]
            mr=np.matrix(r)
            
              
                    
            for x in flip:
                xx=np.matrix(x)  #(D)
                    
                mxt=xx.transpose()          #M=(R*D)
                    
                rx=mr*(mxt)
                    #convert trx into [x,y]
                rx=rx.tolist()
                    #print (txnew)
                coords_d1.append(rx)
                    
                    
                    
            coords_d2=[]  #remove the lists in list
               
            for i in range(0,len(coords_d1)):
                coords_d2.append((coords_d1[i][0][0],coords_d1[i][1][0]))
                    
            
            x5=[i[0] for i in coords_d2]   
            y5=[i[1] for i in coords_d2]
                    
                
                    
               
            fig, ax = plt.subplots()
        
            plt.scatter(x5, y5, marker='x',color="white")   
            ax.set_facecolor('black')
        
            ax.set_ylim(-350,350)
            ax.set_xlim(-350,350)
            os.chdir(r'path to the folder where you want to save the pre-aligned figures')
            #plt.savefig(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned\1.jpg')
            plt.savefig(filename)
            os.chdir(r"path to working directory")
            
#crop the pre-aligned figures
            
path=(r"path to the folder where you want to save the pre-aligned figures")

for filename in os.listdir(path):
    image = cv2.imread(os.path.join(path,filename))
    
    y=40
    x=54
    h=210
    w=335
    crop = image[y:y+h, x:x+w]
    os.chdir(r'path to the folder where you want to save the pre-aligned cropped figures')
    cv2.imshow('Image', crop)
    cv2.waitKey(0)
    cv2.imwrite(filename,crop)
    
#ICP followed by calculation of the Hausdorf distance
    
sys.path.append(r"C:\Users\ar54482\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\ICP" )

import ICP
import os

path=(r"path to the folder where you want to save the pre-aligned cropped figures")
Hausdorf=[]
Filename=[]
for filename in os.listdir(path):
    icp = ICP.ICP( 
                   binary_or_color = "binary",
                   corners_or_edges = "edges",
                   
                   
                   pixel_correspondence_dist_threshold = 4000,
                   auto_select_model_and_data = 1,
                   max_num_of_pixels_used_for_icp = 200,
                   iterations = 20,
                   model_image = r"path to model image of the dataset", 
                   data_image = os.path.join(path,filename),
                   font_file = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
                )
    
    icp.extract_pixels_from_color_image("model")
    icp.extract_pixels_from_color_image("data")
    

    
     
    with open(r'path to where ICP stores the data coords') as f:
         data_coords= json.load(f)
    
    with open(r'path to where ICP stores the model coords') as f:
         model_coords= json.load(f)
         
    with open(r'path to where ICP stores the superimposed ICP coords') as f:
         ICP_coords= json.load(f)
        
   
    x2=[]
    y2=[]
    
  
        
    
    x2=[i[0] for i in ICP_coords]   
    y2=[i[1] for i in ICP_coords]  
    
  
    x1=[i[0] for i in model_coords]   
    y1=[i[1] for i in model_coords]  
    #
    x=[i[0] for i in data_coords]   
    y=[i[1] for i in data_coords] 
    

    fig, ax = plt.subplots(figsize=(10,10))
    
    
    ax.scatter(x, y, c='r') #red is imgdata
    ax.set_facecolor('black')
    
    
    plt.savefig(r"Path where to store the img data shape")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x1, y1, c='g') #green is model 
    ax.set_facecolor('black')
    
    
    plt.savefig(r"Path where to store the img model shape")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x2, y2, c='b') #blue is superimposed data
    ax.set_facecolor('black')
    plt.savefig(r"Path where to store the superimposed img model shape")
    plt.show()
    
    #GET THE SPLINE FIT
    #Not working!!!
    imgdata = Image.open(r'Path where to store the img data shape')
    graymodel = imgdata.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155)
        
    if len(coords1)!=0:
        coords1 = coords1[1]
        
        x3=[i[0] for i in coords1]  
       
        y3=[i[1] for i in coords1] 
        
 
    
    
    #GET THE SPLINE FIT
    #Not working!!!
    imgmodel = Image.open(r"Path where to store the img model shape")
    graymodel = imgmodel.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155)
        
    if len(coords1)!=0:
        coords1 = coords1[1]
        
        x4=[i[0] for i in coords1]  
       
        y4=[i[1] for i in coords1] 
        

    
    imgsimp = Image.open(r'Path where to store the superimposed img model shape')
    graymodel = imgsimp.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155)
        
    if len(coords1)!=0:
        coords1 = coords1[1]
        
        x5=[i[0] for i in coords1]  
       
        y5=[i[1] for i in coords1] 
        
    
    
        
    #Fit a b-spline to model shape
    tck, u = splprep([x4,y4], s=10)
    new_points = splev(u,tck) #model
    
    #tck, u = splprep([xnew,ynew], s=50)
    #new_points = splev(u, tck)
    
    #Fit a b-spline to the superimposed data shape
    tck, u = splprep([x5,y5],s=0)
    new_points1 = splev(u, tck)
    
    fig, ax = plt.subplots()
    #ax.plot(x3, y3)
    ax.plot(new_points[0], new_points[1], c='g')  #model
    
    fig, ax = plt.subplots()
    ax.plot(new_points1[0], new_points1[1], c='blue')  #superimposed
    #ax.scatter(x, y, c='g')
    #plt.show()    
    
    #METHOD1 - HAUSDORFF DISTANCE
    #Hausdorff Distance    
    data_spline=[]
    
    for i in range(0,len(new_points1[0])):
        x=new_points1[0][i]
        y=new_points1[1][i]
        data_spline.append((x,y))
    
    model_spline=[]
    
    for i in range(0,len(new_points[0])):
        x=new_points[0][i]
        y=new_points[1][i]
        model_spline.append((x,y)) 
    
    u1=np.array(model_spline)
    v1=np.array(data_spline)
    
   
    h=directed_hausdorff(u1,v1)
    Hausdorf.append(h[0])
    print("Hausdorf")
    print(Hausdorf)
    print ("Len of Hausdorf")
    print(len(Hausdorf))
    Filename.append(filename)
    print(Filename)
    data={'ID':Filename, 'Hausdorf':Hausdorf}
    NS=pd.DataFrame(data)
    
    #Write out the dataframe to an excel file
    NS.to_csv(r"Path to the excel sheet to store the calculated values", index = False, header = True)
    

#ICP followed by the calculation of Frechet distance 
    

    #GET THE SPLINE FIT
    #Not working!!!
    
    #get the ordered contours list
    
    imgdata = Image.open(r'Path where to store the img data shape')
    graymodel = imgdata.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155,fully_connected='low',positive_orientation='low')
        
    if len(coords1)!=0:
        coords1 = coords1[0]
        
        x3=[i[0] for i in coords1]  
        
        y3=[i[1] for i in coords1] 
    
    fig, ax = plt.subplots(figsize=(5,5))    
    ax.scatter(x3, y3, c='r') 
    ax.set_facecolor('black')
    
    imgmodel = Image.open(r'Path where to store the img model shape')
    graymodel = imgmodel.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155,fully_connected='low',positive_orientation='high')
        
    if len(coords1)!=0:
        coords1 = coords1[0]
        
        x4=[i[0] for i in coords1]  
       
        y4=[i[1] for i in coords1] 
    
    fig, ax = plt.subplots(figsize=(5,5))    
    ax.scatter(x4, y4, c='g') #green is modeldata
    ax.set_facecolor('black')
    
    imgsimp = Image.open(r'Path where to store the superimposed superimposed img model shape')
    graymodel = imgsimp.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155)
        
    if len(coords1)!=0:
        coords1 = coords1[0]
        
        x5=[i[0] for i in coords1]  
       
        y5=[i[1] for i in coords1] 
    
    fig, ax = plt.subplots(figsize=(5,5))      
    ax.scatter(x5, y5, c='blue') 
    ax.set_facecolor('black')
       
    ##Fit a b-spline to model shape
    tck, u = splprep([x4,y4], s=0)
    new_points = splev(u,tck) #model
    
    ##Fit a b-spline to the superimposed data shape
    tck, u = splprep([x5,y5],s=0)
    new_points1 = splev(u, tck)
    #
    fig, ax = plt.subplots()
    ##ax.plot(x3, y3)
    ax.plot(new_points[0], new_points[1], c='g')   #green is model data
    
    fig, ax = plt.subplots()
    ax.plot(new_points1[0], new_points1[1], c='blue')  #blue is superimposed data
    ##ax.scatter(x, y, c='g')
    ##plt.show()    
    data_spline=[]
    
    for i in range(0,len(new_points1[0])):
        x=new_points1[0][i]
        y=new_points1[1][i]
        data_spline.append((x,y))
    
    model_spline=[]
    
    for i in range(0,len(new_points[0])):
        x=new_points[0][i]
        y=new_points[1][i]
        model_spline.append((x,y)) 
    
    u1=np.array(model_spline)
    v1=np.array(data_spline)
    
    s=u1.shape
    v1.resize(s)
    
    ##METHOD1 - FRECHET'S DISTANCE
    f=frdist(u1,v1)
    print("Frechet's distance")
    print(f)
    Frechet.append(f)
    print("Frechet")
    print(Frechet)
    print ("Len of Frechet")
    print(len(Frechet))
    Filename.append(filename)
    print(Filename)
    data={'ID':Filename, 'Frechet':Frechet}
    NS=pd.DataFrame(data)
        
    #Write out the dataframe to an excel file
    NS.to_csv(r"Path to the excel sheet where to store the calculated values", index = False, header = True)
    
  
    

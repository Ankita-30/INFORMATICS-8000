# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:38:15 2020

@author: ar54482
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:48:00 2019

@author: ar54482
"""
from skimage.measure import find_contours
from skimage import io
from skimage.color import convert_colorspace
from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression
import cv2
import math
import seaborn as sns
import statistics
import os
#specify the path
#path=(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Trial2")


#for filename in os.listdir(path):
    
# Read the image you want the connected components extracted from
#img = cv2.imread(os.path.join(path,filename),0)

img = cv2.imread(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes123\New\195.jpg',0)
ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

cv2.imwrite(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\gray.jpg",thresh_img)

imgdata = Image.open(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\gray.jpg")
   

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

#print(x)

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

#print(x1)

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
        
            
    ## Histogram visualization
#    fig, ax1 = plt.subplots()
#    
#    Left=sns.distplot(DL,  kde=False, hist=True,
#                 bins=int(180/5), color = 'green', 
#                 hist_kws={'edgecolor':'green','normed':False},
#                 kde_kws={'linewidth': 2})
#    
#    
#    #plt.bar(XL,DL)
#    
#    fig, ax2 = plt.subplots()
#    
#    Right=Left=sns.distplot(DR,  kde=False, hist=True,
#                 bins=int(180/5), color = 'blue', 
#                 hist_kws={'edgecolor':'blue','normed':False},
#                 kde_kws={'linewidth': 2})
    

#plt.bar(XR,DR)




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
        
#    fig, ax1 = plt.subplots()
#    
#    Left=sns.distplot(DL,  kde=False, hist=True,
#                 bins=int(180/5), color = 'green', 
#                 hist_kws={'edgecolor':'green','normed':False},
#                 kde_kws={'linewidth': 2})
#    
#    #plt.bar(XL,DL)
#    
#    fig, ax2 = plt.subplots()
#    
#    Right=Left=sns.distplot(DR,  kde=False, hist=True,
#                 bins=int(180/5), color = 'blue', 
#                 hist_kws={'edgecolor':'blue','normed':False},
#                 kde_kws={'linewidth': 2})
#
#

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

   
   
#Align its orientation to the mean shape (8.jpg in the Trial folder)
    
#model slope in degrees
S=math.degrees(math.atan(0.81993034))
Sr=math.atan(0.81993034)
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
        
        a=(Sr-Ir)
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
        ax.set_xlim(-450,450)
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS')
        plt.savefig('195.jpg')
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes123\New')
        
        
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
        ax.set_xlim(-450,450)
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS')
        plt.savefig('195.jpg')
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes123\New')
    
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
        ax.set_xlim(-450,450)
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS')
        plt.savefig(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\195.jpg')
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes123\New')
        
        
   
        


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
        ax.set_xlim(-450,450)
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS')
        plt.savefig(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\195.jpg')
        os.chdir(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes123\New')
        
       
            
            







#new co-ordinates after superimposition
#multiply each boundary point with rotation matrix from ICP

#specify rotation matrix from icp  (R)
#r=[[0.97660566, 0.21503809],[-0.21503809, 0.97660566]]
#r=[[0.983, 0.179],[-0.179, 0.983]] 

#mr=np.matrix(r)

#specify translation matrix from icp
#t=[[-3.398],[-52.757]]

#specify model mean from icp (N)
#n=[[71.803],[19.019]]
#nx=np.matrix(n)   #matrix
#nxt=nx.transpose()

#tx=np.matrix(t)   #(T)
#txt=tx.transpose()    #transpose
#print(tx)
#print(mr)
#print(t)

#empty list to store the new rotated & translated co-ordinates
#coords_nrt=[]

#for x in coords_n:
#   
#    xx=np.matrix(x)  #(D)
#    
#    mxt=xx.transpose()          #M=(R*(D-N)+T+N)
#    
#    rx=mr*(mxt-nxt)
#    
#  
#    
#    trx=rx+txt        #translate by adding the translation matrix from icp
#    tnrx=trx+nxt       #add back the model mean
#
#
#    #convert trx into [x,y]
#    txnew=rx.tolist()
#    #print (txnew)
#    coords_nrt.append(txnew)
#    
#    
##centre the model image at the origin
#print()
#print(coords_nrt[1])    
#    
#    
#    
#    
#    
#
##centre the rotated data image at the origin
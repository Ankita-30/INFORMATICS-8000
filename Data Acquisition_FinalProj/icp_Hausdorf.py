# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:01:23 2020

@author: ar54482
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:45:40 2019

@author: ar54482
"""
import sys
from skimage.measure import find_contours
from scipy.spatial import distance
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize
from skimage import img_as_bool, io, color, morphology
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import numpy as np
import seaborn as sns
import json
import cv2
import pandas as pd
###
sys.path.append(r"C:\Users\ar54482\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\ICP" )
#
##icp = ICP.ICP(
##       binary_or_color = "binary",
##       corners_or_edges = "edges",
##       
##       
##       pixel_correspondence_dist_threshold = 40,
##       auto_select_model_and_data = 1,
##       calculation_image_size = 200,
##       iterations = 100,
##       model_image = r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes\sd2_20x1L.png0.jpg",
##       data_image = r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Allfilledshapes\sd7_20x8aL.png266.jpg",
##       font_file = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
##    )
#icp.extract_pixels_from_binary_image("model")
#icp.extract_pixels_from_binary_image("data")
#icp.icp()
##icp.display_images_used_for_edge_based_icp()
##icp.display_results_as_movie()
#icp.cleanup_directory()
#

#Trial runs
#
import ICP
import os

path=(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\PS\Stress_shapes\segmented\Cropped")
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
                   model_image = r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\PS\Stress_shapes\segmented\Cropped\sd2_20x8L.png80.jpg", 
                   data_image = os.path.join(path,filename),
                   font_file = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
                )
    
    icp.extract_pixels_from_color_image("model")
    icp.extract_pixels_from_color_image("data")
    
    icp.icp()
    
    #icp.display_images_used_for_corner_based_icp()
    #
    #icp.display_results_as_movie()
    #icp.cleanup_directory()
    
    
    #get your rotation and translation matrices automatically
    
    #imgdata = Image.open(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\138.jpg")
    #graydata = imgdata.convert('L')   # 'L' stands for 'luminosity'
    #graydata = np.asarray(graydata)
    #
    #
    #           
    #coords = find_contours(graydata,100)
    #    
    #if len(coords)!=0:
    #    coords = coords[0]
    #    
    #
    #    x=[i[0] for i in coords]  
    #   
    #    y=[i[1] for i in coords] 
    #
    #coords_n=[]
    #  
    #     
    #for i in range(0,len(coords)):
    #    
    #    coords_n.append([x[i],y[i]])
    #    
    #    
    #imgmodel = Image.open(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\41.jpg")
    #graymodel = imgmodel.convert('L')   # 'L' stands for 'luminosity'
    #
    #graymodel = np.asarray(graymodel)
    #
    #
    #
    #coords1 = find_contours(graymodel,100)
    #    
    #if len(coords1)!=0:
    #    coords1 = coords1[0]
    #    
    #    x=[i[0] for i in coords1]  
    #   
    #    y=[i[1] for i in coords1] 
    #
    #coords_n1=[]
    #        
    #for i in range(0,len(coords1)):
    #    
    #    coords_n1.append([x[i],y[i]])
    
    #new co-ordinates after superimposition
    #multiply each boundary point with rotation matrix from ICP
    
    #specify rotation matrix from icp  (R)
    #r=[[0.97660566, 0.21503809],[-0.21503809, 0.97660566]]
    R=icp.R
    
    #mr=np.matrix(r)
    
    #specify translation matrix from icp
    T=icp.T
    
     
    with open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\datacoordinates.txt') as f:
         data_coords= json.load(f)
    
    with open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\modelcoordinates.txt') as f:
         model_coords= json.load(f)
         
    with open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\superimposedcoordinates.txt') as f:
         ICP_coords= json.load(f)
        
    #for i in data_transformed_new1:
    #    print(i)
    x2=[]
    y2=[]
    
    #for i in ICP_coords:
    #    x2.append(i[0])
    #    y2.append(i[1])
        
    
    x2=[i[0] for i in ICP_coords]   
    y2=[i[1] for i in ICP_coords]  
    
    #x2=[i[0][0] for i in coords_nrt]   
    #y2=[i[0][1] for i in coords_nrt]  
    
    x1=[i[0] for i in model_coords]   
    y1=[i[1] for i in model_coords]  
    #
    x=[i[0] for i in data_coords]   
    y=[i[1] for i in data_coords] 
    
    #read coordinates from file as given by the Package
    
    #print(coords_n1)
    #
    #x2=[i[0] for i in coords_n1]   
    #y2=[i[1]for i in coords_n1]  
    ##
    ##
    #tck, u = splprep([x,y], s=10)
    #new_points = splev(u, tck)
    ##
    #tck1, u1 = splprep([x1,y1], s=10)
    #new_points1 = splev(u1, tck1) 
    #
    #tck2, u2 = splprep([x2,y2], s=10)
    #new_points2 = splev(u2, tck2) 
    
    # Data shape plotting and contour detection
    #
    fig, ax = plt.subplots(figsize=(10,10))
    
    
    ax.scatter(x, y, c='r') #red is imgdata
    ax.set_facecolor('black')
    
    
    plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Image\red.jpg")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x1, y1, c='g') #green is model 
    ax.set_facecolor('black')
    
    
    plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Image\green.jpg")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x2, y2, c='b') #blue is superimposed data
    ax.set_facecolor('black')
    plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Image\blue.jpg")
    plt.show()
    
    #GET THE SPLINE FIT
    #Not working!!!
    imgdata = Image.open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Image\red.jpg')
    graymodel = imgdata.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155)
        
    if len(coords1)!=0:
        coords1 = coords1[1]
        
        x3=[i[0] for i in coords1]  
       
        y3=[i[1] for i in coords1] 
        
    #ax.scatter(x3, y3, c='r') #green is modeldata
    #ax.set_facecolor('black')
    #ax.axis('off')
    
    #plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\Image\data.jpg")
    
    
    #GET THE SPLINE FIT
    #Not working!!!
    imgmodel = Image.open(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Image\green.jpg")
    graymodel = imgmodel.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155)
        
    if len(coords1)!=0:
        coords1 = coords1[1]
        
        x4=[i[0] for i in coords1]  
       
        y4=[i[1] for i in coords1] 
        
    #ax.scatter(x4, y4, c='g') #green is modeldata
    #ax.set_facecolor('black')
    #ax.axis('off')
    
    #plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\model.jpg")
    
    
    imgsimp = Image.open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Image\blue.jpg')
    graymodel = imgsimp.convert('L')   # 'L' stands for 'luminosity'
    
    graymodel = np.asarray(graymodel)
    
    
    coords1 = find_contours(graymodel,155)
        
    if len(coords1)!=0:
        coords1 = coords1[1]
        
        x5=[i[0] for i in coords1]  
       
        y5=[i[1] for i in coords1] 
        
    #ax.scatter(x4, y4, c='b') #green is modeldata
    #ax.set_facecolor('black')
    #ax.axis('off')
    
    #plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\simp.jpg")
    
        
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
    data={'ID':Filename, 'Hooking':Hausdorf}
    NS=pd.DataFrame(data)
    
    #Write out the dataframe to an excel file
    NS.to_csv(r"C:\Users\ar54482\Desktop\DataCellShapePS2.csv", index = False, header = True)
    
    #Also write out the filename data in a file
    fF = open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\Filename.txt', 'w')
    print(Filename, file = fF)
    
    #Also write out the Hausdorf data in a file
    fH = open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\Hausdorf.txt', 'w')
    print(Hausdorf, file = fH)
    
   #Filename.to_csv(r"C:\Users\ar54482\Desktop\DataCellShape.csv", index = False, header = True)
 


#METHOD2 - MEDIAL AXIS DISTANCE
   
#    #extract medial axis
#    image = img_as_bool(color.rgb2gray(io.imread(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\3_1.jpg')))
#    #out = morphology.medial_axis(image)
#    out = skeletonize(image)
#    
#    
#    
#    
#    img = cv2.imread(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\3_1.jpg',0)
#    ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#    
#    cv2.imwrite(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\gray.jpg",thresh_img)
#    
#    imgdata = Image.open(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\gray.jpg")
#       
#    
#    width,height=imgdata.size
#    
#    coords=[] #store the white coordinates here
#    #count=0
#    pix=[]
#    boundary=[]
#    
#    for i in range(width):
#        for j in range(height):
#            coordinate=i,j
#            p=imgdata.getpixel((i,j))
#            pix.append(p)
#    
#            if p==255:      
#                coords.append((i,j)) 
#                p1=imgdata.getpixel((i,j-1))
#                p2=imgdata.getpixel((i-1,j))
#                p3=imgdata.getpixel((i,j+1))
#                p4=imgdata.getpixel((i+1,j))
#    
#                if p1==0 or p2==0 or p3==0 or p4==0:
#                    boundary.append((i,j))    
#    
#    
#    xb=[i[0] for i in boundary]  
#    yb=[i[1] for i in boundary] 
#    
#    f=plt.figure(figsize=(15,15))
#    ax0 = f.add_subplot(421)
#    ax1 = f.add_subplot(422)
#    ax0.imshow(image, cmap='gray', interpolation='nearest')
#    ax1.imshow(out, cmap='gray', interpolation='nearest')
#    ax1.scatter(xb,yb, c='white')
#    
#    plt.show()
#    
#    
#    
#    #METHOD3 
#    
#    #determine the centroid
#    xbc=sum(xb)/len(xb)
#    ybc=sum(yb)/len(yb)
#    
#    f=plt.figure(figsize=(15,15))
#    ax2 = f.add_subplot(423)
#    
#    ax2.scatter(xb,yb, c='black')
#    plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\3_1b.jpg")
#    ax2.scatter(xbc,ybc,c='black')
#    ax2.set_ylim(200,0)
#    ax2.set_xlim(0,350)
#    
#    #print("Length of boundary")
#    #print(len(boundary))
#    #print(boundary)
#    
#    
#    #histogram of distance of the boundary points from the centroid
#    D=[] #empty list to store the distances 
#    for p in boundary:
#        pc=(xbc,ybc)
#        d=distance.euclidean(p,pc)
#        D.append(d)
#        
#    
#    ax2 = f.add_subplot(424)    
#    sns.distplot(D,kde=False)
    
    
    #Attempt to fill the boundary holes - Not Working!!
    #from skimage import io, morphology, img_as_bool, segmentation
    #from scipy import ndimage as ndi
    #import matplotlib.pyplot as plt
    #
    #image = img_as_bool(color.rgb2gray(io.imread(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\red.jpg')))
    #out = ndi.distance_transform_edt(~image)
    #out = out < 0.05 * out.max()
    #out = morphology.skeletonize(out)
    #out = morphology.binary_dilation(out, morphology.selem.disk(1))
    #out = segmentation.clear_border(out)
    #out = out | image
    #
    #ax3 = f.add_subplot(424)
    #ax3.imshow(out, cmap='gray')
    ##ax3.set_ylim(300,600)
    ##ax3.set_xlim(0,800)
    ##plt.imsave('/tmp/gaps_filled.png', out, cmap='gray')
    #plt.show()
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

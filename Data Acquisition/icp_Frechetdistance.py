# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:52:43 2020

@author: ar54482
"""

#frechet's distance


import sys
from skimage.measure import find_contours
from scipy.spatial import distance
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize
from skimage.morphology import medial_axis
from skimage import img_as_bool, io, color, morphology
from scipy.spatial.distance import directed_hausdorff
from frechetdist import frdist
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
Frechet=[]
Filename=[]
for filename in os.listdir(path):
    icp = ICP.ICP( 
                   binary_or_color = "color",
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
    
    
     
    with open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\datacoordinates.txt') as f:
         data_coords= json.load(f)
    
    with open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\modelcoordinates.txt') as f:
         model_coords= json.load(f)
         
    with open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Pre-aligned_NS\superimposedcoordinates.txt') as f:
         ICP_coords= json.load(f)
        
    
    x2=[i[0] for i in ICP_coords]   
    y2=[i[1] for i in ICP_coords]  
    
    
    
    x1=[i[0] for i in model_coords]   
    y1=[i[1] for i in model_coords]  
    #
    x=[i[0] for i in data_coords]   
    y=[i[1] for i in data_coords] 
    
    
    
    #plot all shapes in the same figure
    plt.scatter(x,y, c='r')
    plt.scatter(x1,y1, c='g')
    plt.scatter(x2,y2, c='blue')
    
    # Data shape plotting and contour detection
    #
    fig,ax = plt.subplots(figsize=(5,5))
    ax.scatter(x, y, c='r') #red is imgdata
    ax.set_facecolor('black')
    ax.axis('off')
    
    plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\red.jpg")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x1, y1, c='g') #green is model 
    ax.set_facecolor('black')
    ax.axis('off')
    
    plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\green.jpg")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x2, y2, c='blue') #blue is superimposed data
    ax.set_facecolor('black')
    ax.axis('off')
    plt.savefig(r"C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\blue.jpg")
    plt.show()
    
    #GET THE SPLINE FIT
    #Not working!!!
    
    #get the ordered contours list
    
    imgdata = Image.open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\red.jpg')
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
    
    imgmodel = Image.open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\green.jpg')
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
    
    imgsimp = Image.open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\NS\Cropped\blue.jpg')
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
    NS.to_csv(r"C:\Users\ar54482\Desktop\DataCellShapeC1.csv", index = False, header = True)
    
    #Also write out the filename data in a file
    fF = open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\Filename1.txt', 'w')
    print(Filename, file = fF)
    
    #Also write out the Hausdorf data in a file
    fH = open(r'C:\Users\ar54482\Desktop\Data\Main_data\L88-57\Frechet.txt', 'w')
    print(Frechet, file = fH)
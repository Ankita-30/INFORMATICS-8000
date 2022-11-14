# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:13:16 2020

@author: ar54482
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:17:24 2019

@author: ar54482
"""

#Read excel file into python

import pandas as pd
import matplotlib.pyplot as plt
#from scipy.stats import ttest_ind
#import numpy as np
import statistics
import random
import math
import seaborn as sns

dfP = pd.read_csv(r"C:\Users\ar54482\Desktop\DataCellShapeFinalPS.csv")
P=dfP['Perimeter']
Pl = dfP["Perimeter"].values

dfN = pd.read_csv(r"C:\Users\ar54482\Desktop\DataCellShapeFinalNS.csv")
N=dfN['Perimeter']
Nl = dfN["Perimeter"].values

dfC = pd.read_csv(r"C:\Users\ar54482\Desktop\DataCellShapeFinalC.csv")
C=dfC['Perimeter']
Cl = dfC["Perimeter"].values

my_colors=['blue','green','red']


#P=sns.distplot(P,  kde=True, hist=False,
#             bins=int(180/5), color = 'red', 
#             hist_kws=dict(cumulative=True),
#             kde_kws=dict(cumulative=True))
#
#
#N=sns.distplot(N,  kde=True, hist=False,
#             bins=int(180/5), color = 'green', 
#             hist_kws=dict(cumulative=True),
#             kde_kws=dict(cumulative=True))
#
#
#C=sns.distplot(C,  kde=True, hist=False,
#             bins=int(180/5), color = 'blue', 
#             hist_kws=dict(cumulative=True),
#             kde_kws=dict(cumulative=True))

df1=pd.DataFrame({'P Stress':P,'Control':C, 'N Stress':N})
ax=df1.plot(kind="kde",bw_method=0.5,color=my_colors)
ax.set_xlim(-500,2000)
ax.set_ylim(0,0.002)
#
#ax.grid(True, which='both')
#ax.axes.get_xaxis().set_visible(False)
#ax.axes.get_yaxis().set_visible(False)
#plt.title('Length',fontsize=12)
plt.xlabel('Perimeter (in pixels)',fontsize=14)
##plt.ylabel('Probability Density Function',fontsize=14)
plt.ylabel(" ")
plt.show()


#
#Cl=Cl.tolist()
#
#Con=[]
#for x in Cl:
#    if math.isnan(x)==False:
#        Con.append(x)
#        
#
#Pl=Pl.tolist()
#
#Ps=[]
#for x in Pl:
#    if math.isnan(x)==False:
#        Ps.append(x)
#        
#
#Nl=Nl.tolist()
#
#Ns=[]
#for x in Nl:
#    if math.isnan(x)==False:
#        Ns.append(x)
#        
##
###2 sample t-Test
##PC=ttest_ind(a=np.array(Ps),b=np.array(Con),equal_var=False)
##
##NC=ttest_ind(a=np.array(Ns),b=np.array(Con),equal_var=False)
##
##print(PC)
##
##print(NC)
#"""
#Bootstrapping hypothesis testing
#
#Ho=P and C are the same distribution
#Test statistic = difference in their means
#No. of bootstrap samples = 100
#
#"""
#
#"""
#PC p-value
#"""
#####Calculate the test-statistic
#pMean=statistics.mean(Ps)
#
#ConMean=statistics.mean(Con)
#
#Diff_avg=pMean-ConMean
##print(Diff_avg)
##Combine the samples
#Combined=Ps+Con
#
##list for the all the test-statistics
#D=[]
#BS=Ps+Con
#
#for i in range(1,101):
#    
#    ###Get a bootstrap sample with replacement with 700 Control values and 1000 P-Stress values
#    
#    BS1=random.choices(BS,k=len(Ps)+len(Con))
#    
#    
#    #BS=PS+Control #Bootstrap sample 1
#    
#    #get the test-statistic from bootstrap sample 1
#    d=statistics.mean(BS1[0:1226])-statistics.mean(BS1[1226:1975])
#    D.append(abs(d))
#
##calculate p-value
#i=0
#for x in D:
#    if x>=Diff_avg:
#        i=i+1
#        
#p=i/100
#
##plt.hist(D, normed=False, alpha=0.5,bins=15,color='green',cumulative=False)
#print(p)
#
#"""
#Bootstrapping hypothesis testing
#
#Ho=P and C are the same distribution
#Test statistic = difference in their means
#No. of bootstrap samples = 100
#
#"""
#
#"""
#NC p-value
#"""
####Calculate the test-statistic
#nMean=statistics.mean(Ns)
#
#ConMean=statistics.mean(Con)
#
#
#Diff_avg=nMean-ConMean
##print(Diff_avg)
##Combine the samples
#Combined=Ns+Con
#
##list for the all the test-statistics
#D=[]
#BS=Ns+Con
#for i in range(1,101):
#    
#    ###Get a bootstrap sample with replacement with 700 Control values and 1000 P-Stress values
#    
#    BS1=random.choices(BS,k=len(Ns)+len(Con))
#    
#    
#    #BS=PS+Control #Bootstrap sample 1
#    
#    #get the test-statistic from bootstrap sample 1
#    d=statistics.mean(BS1[0:705])-statistics.mean(BS1[705:1454])
#    D.append(abs(d))
#
##calculate p-value
#i=0
#for x in D:
#    if x>=Diff_avg:
#        i=i+1
#        
#p=i/100
#
##plt.hist(D, normed=False, alpha=0.5,bins=15,color='green',cumulative=False)
#print(p)
#
#"""
#Bootstrapping hypothesis testing
#
#Ho=P and C are the same distribution
#Test statistic = difference in their means
#No. of bootstrap samples = 100
#
#"""
#
#"""
#NP p-value
#"""
####Calculate the test-statistic
#pMean=statistics.mean(Ps)
#
#nMean=statistics.mean(Ns)
#
#
#Diff_avg=pMean-nMean
##print(Diff_avg)
##Combine the samples
#Combined=Ns+Con
#
##list for the all the test-statistics
#D=[]
#BS=Ns+Ps
#for i in range(1,101):
#    
#    ###Get a bootstrap sample with replacement with 700 Control values and 1000 P-Stress values
#    
#    BS1=random.choices(BS,k=len(Ns)+len(Ps))
#    
#    
#    #BS=PS+Control #Bootstrap sample 1
#    
#    #get the test-statistic from bootstrap sample 1
#    d=statistics.mean(BS1[0:1226])-statistics.mean(BS1[1226:1931])
#
#    D.append(abs(d))
#
##calculate p-value
#i=0
#for x in D:
#    if x>=Diff_avg:
#        i=i+1
#        
#p=i/100
#
##plt.hist(D, normed=False, alpha=0.5,bins=15,color='green',cumulative=False)
#print(p)
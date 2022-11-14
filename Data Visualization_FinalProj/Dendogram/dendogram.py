# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 08:50:14 2020

@author: ar54482
The code shows a heat-map of the difference between every two shapes across all treatments combined for each of the shape traits like area, perimeter, hausdorf and frechet distance.
"""

# Libraries
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
 

#Frechet Distance
df = pd.read_csv(r'C:\Users\ar54482\Desktop\Frechet.csv')
df = df.set_index('ID')
del df.index.name
x=df['Frechet'].tolist()

#Generate random features and distance matrix.
#x = scipy.rand(40)
D = scipy.zeros([361,361])
for i in range(361):
    for j in range(361):
        D[i,j] = abs(x[i] - x[j])


condensedD = squareform(D)

#Compute and plot first dendrogram.
fig = pylab.figure(figsize=(8,8))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
Y = sch.linkage(condensedD, method='complete')
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])

#Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y = sch.linkage(condensedD, method='complete')
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

#Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

#Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
pylab.colorbar(im, cax=axcolor)
fig.show()
fig.savefig(r'C:\Users\ar54482\Desktop\Plots\Dendogram\Frechet.png')


#Hausdorf Distance
df = pd.read_csv(r'C:\Users\ar54482\Desktop\Hausdorf.csv')
df = df.set_index('ID')
del df.index.name
x=df['Hausdorf'].tolist()

#Generate random features and distance matrix.
#x = scipy.rand(40)
D = scipy.zeros([369,369])
for i in range(361):
    for j in range(361):
        D[i,j] = abs(x[i] - x[j])


condensedD = squareform(D)

#Compute and plot first dendrogram.
fig = pylab.figure(figsize=(8,8))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
Y = sch.linkage(condensedD, method='complete')
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])

#Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y = sch.linkage(condensedD, method='complete')
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

#Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

#Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
pylab.colorbar(im, cax=axcolor)
fig.show()
fig.savefig(r'C:\Users\ar54482\Desktop\Plots\Dendogram\Hausdorf.png')

#Perimeter
df = pd.read_csv(r'C:\Users\ar54482\Desktop\Perimeter.csv')
df = df.set_index('ID')
del df.index.name
x=df['Perimeter'].tolist()

#Generate random features and distance matrix.
#x = scipy.rand(40)
D = scipy.zeros([369,369])
for i in range(361):
    for j in range(361):
        D[i,j] = abs(x[i] - x[j])


condensedD = squareform(D)

#Compute and plot first dendrogram.
fig = pylab.figure(figsize=(8,8))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
Y = sch.linkage(condensedD, method='complete')
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])

#Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y = sch.linkage(condensedD, method='complete')
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

#Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

#Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
pylab.colorbar(im, cax=axcolor)
fig.show()
fig.savefig(r'C:\Users\ar54482\Desktop\Plots\Dendogram\Perimeter.png')

#Area
df = pd.read_csv(r'C:\Users\ar54482\Desktop\Area.csv')
df = df.set_index('ID')
del df.index.name
x=df['Area'].tolist()

#Generate random features and distance matrix.
#x = scipy.rand(40)
D = scipy.zeros([369,369])
for i in range(361):
    for j in range(361):
        D[i,j] = abs(x[i] - x[j])


condensedD = squareform(D)

#Compute and plot first dendrogram.
fig = pylab.figure(figsize=(8,8))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
Y = sch.linkage(condensedD, method='complete')
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])

#Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y = sch.linkage(condensedD, method='complete')
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

#Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

#Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
pylab.colorbar(im, cax=axcolor)
fig.show()
fig.savefig(r'C:\Users\ar54482\Desktop\Plots\Dendogram\Area.png')


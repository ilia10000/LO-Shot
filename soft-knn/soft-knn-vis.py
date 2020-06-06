#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:39:25 2020

@author: ilia10000
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import matplotlib
from soft_knn import SoftKNN

mode = "2points_n_classes" #selection, label, location, label_and_location
num_points = 2
num_classes = 4
granularity=800
n_neighbors = num_points*granularity
ims=[]
fig=plt.figure()
ax1 = fig.add_subplot(1,2,1)
axs = [fig.add_subplot(num_points,2,2*(point+1)) for point in range(num_points)]

def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def animate(frame):
    #frame+=20
    print(frame)
    if mode == "poly_and_center":
        num_classes = 2*(num_points-1)+num_points
        polygon = matplotlib.patches.RegularPolygon((0,0),num_points-1, radius=5)
        dd=polygon.get_verts()[:-1]
        dd=np.append(dd,np.array([[0,0]]),axis=0)
        dy=np.zeros((num_points,num_classes))
        for i in range(num_points-1): 
            #Set the surrounding polygon classes
            dy[i][2*i]=0.4-frame*0.0012
            dy[i][(2*i+1)%(2*(num_points-1))]=0.2+frame*0.0004
            dy[i][(2*i-1)%(2*(num_points-1))]=0.2+frame*0.0004
            dy[i][(2*(num_points-1)+i)]=0.2+frame*0.0004
            #Set the central classes
            dy[-1][(2*(num_points-1)+i)]=0.2+frame*0.0004
        dy[-1][-1]=0.4-frame*0.0004*(num_points-1)
    if mode =="every_other":
        num_classes = 2*num_points
        polygon = matplotlib.patches.RegularPolygon((0,0),num_classes, radius=5)
        dd=polygon.get_verts()[:-1:2]
        dy=np.zeros((num_points,num_classes))
        for i in range(num_points):
            dy[i][2*i]=0.49-frame*0.001
            dy[i][(2*i+1)%num_classes]=0.255+frame*0.0005
            dy[i][(2*i-1)%num_classes]=0.255+frame*0.0005
    if mode == "poly_every_subset":
        num_classes = 2**num_points
        polygon = matplotlib.patches.RegularPolygon((0,0),num_points, radius=5)
        dd=polygon.get_verts()[:-1]
        dy=np.full((num_points,num_classes),num_points/num_classes)
        #Incomplete
    if mode == "2points_n_classes":
        num_classes=4
        dd=np.array([[0,0],[num_classes,0]])
        dy=np.zeros((num_points,num_classes))
        if num_classes ==3:
            dy[0]=np.array([3/5,2/5,0])
        if num_classes==4:
            #dy[0]=np.array([5/12,4/12,3/12,0])
            dy[0]=np.array([6/14,5/14,3/14,0])
        if num_classes==5:
            dy[0]=np.array([60/200,49/200,47/200,44/200,0])
        if num_classes==6:
            dy[0]=np.array([56/210,52/210,42/210,40/210,20/210,0])

     
        dy[1]=dy[0][::-1]    
    #print(dy)
    if not mode != "true":
        distX=[]
        distY=[]
        for i in range(num_points):
            class_list = []
            for j in range(num_classes):
                class_list.append(int(dy[i][j]*granularity)+1*(mode!="2points_n_classes"))
                distY.append(np.repeat(j, class_list[-1]))
            #print(class_list)
            #class_list.append(granularity-sum(class_list))
            #print(class_list)
            #distY.append(np.repeat(i, class_list[-1]))
            distX.append(np.repeat([dd[i]], sum(class_list),axis=0))
            #tempy=
            
            
        distX=np.concatenate(distX)
        distY=np.concatenate(distY)
    distX=dd
    distY=dy
        #print(distX)
        #print(distY)
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFA8E8', '#A8FFE8', '#E8A8FF', '#E8FFA8' ])
    cmap_bold = ListedColormap(['#FF8888', '#88FF88', '#8888FF', '#994282', '#429982', '#824299' ,'#829942' ])
    cmap_bolder = ListedColormap(['#000000', '#000000', '#000000'])
    colors=['#FFAAAA', '#AAFFAA', '#AAAAFF']
    cmap_single = get_cmap(num_classes, "Set3")
    colors=cmap_single(range(num_classes))
    for weights in ['distance']:
        # create KNN classifier
        clf = SoftKNN()
        #clf = neighbors.KNeighborsClassifier(min(n_neighbors, len(distX)), weights=weights)
        clf.fit(distX, distY)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if mode=="2points_n_classes":
            x_min, x_max = -4, num_classes +4
            y_min, y_max = -4, 4
        else:
            x_min, x_max = -20,20
            y_min, y_max = -20,20
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax1.clear()
        ax1.pcolormesh(xx, yy, Z, cmap=cmap_single)
        if not mode =="true": 
            ax1.scatter(distX[:, 0], distX[:, 1], c="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        #plt.title("3-Class classification (k = %i)")
        for i in range(num_points):
            axs[i].pie(dy[i], colors=colors)
            
        if mode== "2points_n_classes":
            axs[0].title.set_text("Left")
            axs[1].title.set_text("Right")
        
import matplotlib.animation as animation

if mode=="2points":
    frames = 190
    interval = 20
elif mode == "true":
    frames = 1
    interval = 1000
else:
    frames = 1
    interval = 200
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False) 
#plt.show()
if frames==1:
    anim.save('{0}_{1}_{2}_alt_zoom_out.png'.format(mode,num_points,num_classes),writer='imagemagick') 
else:
    anim.save('{0}_{1}_{2}_new.gif'.format(mode,num_points,num_classes),writer='imagemagick') 

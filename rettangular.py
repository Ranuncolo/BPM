# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:15:33 2019

@author: 2353588g
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d


width = 120
height = 50
length = 220

dimen = [width,height,length]



# open figure, and make it a little bigger with figsize
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d') #oper a subplot 3d with projection

"""
Function which makes the 6 surfaces
- actually just 4 for now
"""
def face(dim, transp = 0.5, facec = 'c', c_edge = 'k'):
    """
    dim = [width (x), height (y), length(z)]
    transp = transparency, default is .5
    facec = face colour, defaul cyan
    c_edge = edge colour, default black
    """
    p_bot = Rectangle((0,0),dim[0],dim[2],alpha = transp, facecolor = facec, edgecolor= c_edge)
    p_up = Rectangle((0,0),dim[0],dim[2],alpha = transp, facecolor = facec, edgecolor= c_edge)
    p_side1 = Rectangle((0,0),dim[2],dim[1],alpha = transp, facecolor = facec, edgecolor= c_edge)
    p_side2 = Rectangle((0,0),dim[2],dim[1],alpha = transp, facecolor = facec, edgecolor= c_edge)
#    p_front =
#    p_back = 
    return p_bot, p_up, p_side1, p_side2    

"""
Add these surfaces to the plot
"""

#p_bot, p_up, p_side1, p_side2 = face(dimen)

surf = face(dimen, transp=0.3)
ax.add_patch(surf[0])
art3d.pathpatch_2d_to_3d(surf[0], z=0, zdir="z")   
ax.add_patch(surf[1])
art3d.pathpatch_2d_to_3d(surf[1], z=height, zdir="z")   

ax.add_patch(surf[2])
art3d.pathpatch_2d_to_3d(surf[2], z=0, zdir="x")   
ax.add_patch(surf[3])
art3d.pathpatch_2d_to_3d(surf[3], z=width, zdir="x")   

#ax.axis('equal')
ax.set_aspect('auto')
ax.set_xlim(0, 160)
ax.set_ylim(0, 200)
ax.set_zlim(0, 100)

"""
******************************************************************************
Add mesh points
*****************************************************************************
"""

dx = 10.2  #chosen, it will be given as an input

try:
    i = int(dx)
    print("Ok, I can go on, the value is ",i)
except ValueError as err:
    print(err)
    


xx = np.linspace(0,width,round(width/dx+1))
yy = np.linspace(0,height,round(height/dx)+1)
zz = np.linspace(0,length,round(length/dx+1))

#meshgrid make the grid at the bottom surface
X, Z = np.meshgrid(xx,zz)
X = np.resize(X,(-1,1))
Z = np.resize(Z,(-1,1))

# however, it is not enough and I have to increase the number of points
# repeating the vectors with tile
a = np.tile(X,yy.shape[0])
b = np.tile(Z,yy.shape[0])
c = np.tile(yy,X.shape[0])

ax.scatter(a, b, c, marker = 'x', c = 'k')
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:19:05 2019

@author: 2353588g
"""

"""
This script will plot the waveguide and it's mesh
"""

from grid_xyz import grid_xyz
import matplotlib.pyplot as plt
import numpy as np

# insert parameters
width = 50
height = 120  
length = 200 
max_step = 10

# call function to get mesh step in each direction
dx, dy, dz = grid_xyz(width,height,length,max_step)

plt.plot([0,width],[0,0],[0,0],[0,height])

#frame = np.array[(0,width),(0,height)]






#
#from visual import *
#scene.title = "VPython: Draw a rotating cube"
# 
#scene.range = 2
#scene.autocenter = True
# 
#print("Drag with right mousebutton to rotate view.")
#print("Drag up+down with middle mousebutton to zoom.")
# 
#deg45 = math.radians(45.0)  # 0.785398163397
# 
#cube = box()    # using defaults, see http://www.vpython.org/contents/docs/defaults.html 
#cube.rotate( angle=deg45, axis=(1,0,0) )
#cube.rotate( angle=deg45, axis=(0,0,1) )
# 
#while True:                 # Animation-loop
#    rate(50)
#    cube.rotate( angle=0.005, axis=(0,1,0) )
#
#




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

#draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r,r,r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s,e), color="b")
plt.show()
























"""
From
https://stackoverflow.com/questions/44881885/python-draw-parallelepiped/49766400#49766400
"""

#import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
#import matplotlib.pyplot as plt
#
#points = np.array([[-1, -1, -1],
#                  [1, -1, -1 ],
#                  [1, 1, -1],
#                  [-1, 1, -1],
#                  [-1, -1, 1],
#                  [1, -1, 1 ],
#                  [1, 1, 1],
#                  [-1, 1, 1]])
#
#P = [[2.06498904e-01 , -6.30755443e-07 ,  1.07477548e-03],
# [1.61535574e-06 ,  1.18897198e-01 ,  7.85307721e-06],
# [7.08353661e-02 ,  4.48415767e-06 ,  2.05395893e-01]]
#
#Z = np.zeros((8,3))
#for i in range(8): Z[i,:] = np.dot(points[i,:],P)
#Z = 10.0*Z
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#r = [-1,1]
#
#X, Y = np.meshgrid(r, r)
## plot vertices
#ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
#
## list of sides' polygons of figure
#verts = [[Z[0],Z[1],Z[2],Z[3]],
# [Z[4],Z[5],Z[6],Z[7]], 
# [Z[0],Z[1],Z[5],Z[4]], 
# [Z[2],Z[3],Z[7],Z[6]], 
# [Z[1],Z[2],Z[6],Z[5]],
# [Z[4],Z[7],Z[3],Z[0]]]
#
## plot sides
#ax.add_collection3d(Poly3DCollection(verts, 
# facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
#
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#
#plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# draw cube
def rect_prism(x_range, y_range, z_range):
    # TODO: refactor this to use an iterator
    xx, yy = np.meshgrid(x_range, y_range)
    ax.plot_wireframe(xx, yy, z_range[0], color="r")
    ax.plot_surface(xx, yy, z_range[0], color="r", alpha=0.2)
    ax.plot_wireframe(xx, yy, z_range[1], color="r")
    ax.plot_surface(xx, yy, z_range[1], color="r", alpha=0.2)


    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_wireframe(x_range[0], yy, zz, color="r")
    ax.plot_surface(x_range[0], yy, zz, color="r", alpha=0.2)
    ax.plot_wireframe(x_range[1], yy, zz, color="r")
    ax.plot_surface(x_range[1], yy, zz, color="r", alpha=0.2)

    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_wireframe(xx, y_range[0], zz, color="r")
    ax.plot_surface(xx, y_range[0], zz, color="r", alpha=0.2)
    ax.plot_wireframe(xx, y_range[1], zz, color="r")
    ax.plot_surface(xx, y_range[1], zz, color="r", alpha=0.2)


rect_prism(np.array([-1, 1]), np.array([-1, 1]), np.array([-0.5, 0.5]))
plt.show()
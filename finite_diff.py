# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:45:25 2019

@author: 2353588g
"""
import numpy as np
def finite_diff(tensor, grid, dx, dy):
    
    """ 
    This function return the finite differences of a grid matrix in a dict 
    form. It is general, both the Electromagnetic field and the permittivity
    tensor. It is called at the beginnig of each dz-step to get the finite
    differences of the permittivity tensor.
    
    
    INPUT
    - tensor [dict]:    field/permitt tensor (x,y) of this slice
    - grid [tuple]:     grid or reduced grid (x,y) in order to compute finite 
                        diff at these nodes. If the grid is NxM, the r_grid is
                        (N-2)x(M-2)
    - dx [float]:       step size along the x-axis
    - dy [float]:       step size along the y-axis
    
    OUTPUT
    - dic2x [dict]:     second order finite diff along x
    - dic2y [dict]:     second order finite diff along y
    - dic2xy [dict]:    mixed second order finite differences 
    - dicx [dict]:      finite differences along x
    - dicy [dict]:      finite differences along y
    
    """
    
    #preallocation of dictionaries
    dic2x = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}
    dic2y = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}
    dic2xy = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}
    dicx = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}
    dicy = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}
    
#    print(dic2x.keys(), end = ' ')
#    print(dic2x.values(), end = ' ')
#    print('-----------------------')
#    print(tensor.keys(), end = ' ')
#    print(tensor.values(), end = ' ')
    # actual computation of finite differences:
    
    #permittivity tensor of the air, for now I assume it's one since there is 
    #no other values. Just to see if it works.]
    #In this way, when I am going to access something outside the grid, I am
    #that's everything is 1 there. We will see in the future how to change it
    
    #I sue dict.get method ---> dict.get(key,wanted_value)
    ones = np.array([1,1,1,1,1,1,1,1,1])
    
    for key in dic2x:
        dic2x[key] = (tensor.get((key[0]+1,key[1]),ones) - 2 * tensor[key] \
             + tensor.get((key[0]-1,key[1]),ones))/dx**2
    for key in dic2y:
        dic2y[key] = (tensor.get((key[0],key[1]+1),ones) - 2 * tensor[key] \
             + tensor.get((key[0],key[1]-1),ones))/dy**2
    #classical mixed derivative
    for key in dic2xy:
        dic2xy[key] = (tensor.get((key[0]+1,key[1]+1),ones) \
              - tensor.get((key[0]-1,key[1]+1),ones)  \
              - tensor.get((key[0]+1,key[1]-1),ones)  \
              + tensor.get((key[0]-1,key[1]-1),ones))/(4*dx*dy)
    for key in dicx:
        dicx[key] = (tensor.get((key[0]+1,key[1]),ones) - tensor[key])/dx
    for key in dicy:
        dicy[key] = (tensor.get((key[0],key[1]+1),ones) - tensor[key])/dy
    return dic2x, dic2y, dic2xy, dicx, dicy







"""
TEST
import numpy as np


s = 10

s = 10
w = np.linspace(-s,s,2*s+1)
xs,ys = np.meshgrid(w,w)
grid = (xs,np.flipud(ys)) 

e = np.array([1,2,3,4,5,6,7,8,9])
tensor = {(i,j):e for i,j in zip(grid[0].flat,grid[1].flat)}

dx = 1 
dy = 1

l_r = np.linspace(-(s-1),s-1,s*2-1)
x_r,y_r = np.meshgrid(l_r,l_r)
grid_red = (x_r,np.flipud(y_r))
per_dx2, per_dy2, per_dxy, per_dx, per_dy = finite_diff(tensor,grid_red,dx,dy)
"""
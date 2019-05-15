# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:45:25 2019

@author: 2353588g
"""

def finite_diff(tensor, grid, dx, dy):
    
    """ 
    This function return the finite differences of a grid matrix in a dict 
    form. It is in general called for both the Electromagnetic field and the
    permittivity tensor at the beginnig of each dz-step. 
    
    
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
    
    # actual computation of finite differences:
    
    for key in dic2x:
        dic2x[key] = (tensor[key[0]+1,key[1]] - 2 * tensor[key] + tensor[key[0]-1,key[1]])/dx**2
    for key in dic2y:
        dic2y[key] = (tensor[key[0],key[1]+1] - 2 * tensor[key] + tensor[key[0],key[1]-1])/dy**2
    #classical mixed derivative
    for key in dic2xy:
        dic2xy[key] = (tensor[key[0]+1,key[1]+1] - tensor[key[0]-1,key[1]+1]  \
              - tensor[key[0]+1,key[1]-1] + tensor[key[0]-1,key[1]-1])/(4*dx*dy)
    for key in dicx:
        dicx[key] = (tensor[key[0]+1,key[1]] - tensor[key])/dx
    for key in dicy:
        dicy[key] = (tensor[key[0],key[1]+1] - tensor[key])/dy
    return dic2x, dic2y, dic2xy, dicx, dicy


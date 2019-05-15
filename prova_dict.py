# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:09:47 2019

@author: 2353588g
"""

import numpy as np

# make a dictionary from -5 to 5, x y

#size of dict s
s = 5
l = np.linspace(-s,s,s*2+1)

xl, yl = np.meshgrid(l,l)

e = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) 

#make first grid
d = {(i,j):e for i,j in zip(xl.flat, yl.flat)}



#now add funtion which makes finite differences

def D2x(dic,grid,dx):
    # dic1 = 
    # grid is the reducted one 
    
    # initialise fake dictionary2 
    dic2x = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)} 
    
    for key in dic2x:
        dic2x[key] = (dic[key[0]+1,key[1]] - 2 * dic[key] + dic[key[0]-1,key[1]])/dx**2
    return dic2x


l_r = np.linspace(-(s-1),s-1,s*2-1)

x_r,y_r = np.meshgrid(l_r,l_r)
grid_red = (x_r,y_r)

p = D2x(d,grid_red,1) 












# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:31:05 2019

@author: 2353588g
"""
import numpy as np
"""
This function will make a (nodes x 9) matrix which is then used to make a
sparsed banded matrix. The structure is as follows:
    
    i-1,j+1      i,j+1    1+1,j+1
      (0)        (1)        (2)
    i-1,j        i,j      1+1,j  
      (3)        (4)        (5) 
    i-1,j-1      i,j-1    1+1,j-1
      (6)        (7)        (8)  
      
where the number between () is the vector position of each component.

"""


"""
to test if it works, uncomment the following code

s = 5
w = np.linspace(-s,s,2*s+1)
xs,ys = np.meshgrid(w,w)
grid = (xs,np.flipud(ys)) ##why do we need this flipup? It is just
#because I need to start from the top-left corner or the grid to
#solve it or I am gonna have a weird matrix. 

e = np.array([1,2,3,4,5,6,7,8,9])


tensor = {(i,j):e for i,j in zip(grid[0].flat,grid[1].flat)}

dx = 1
dy = 1

exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz = (0,1,2,3,4,5,6,7,8)


alpha, beta, gamma, delta, epsi, psi, phi, chi, zeta = 1,2,3,4,5,6,7,8,9


# tensor is my dictionary with, at each key (x,y), the respective permit_tensor
node_n = tuple(tensor.keys())
Pxx = np.zeros((len(node_n),9))



for l,c in enumerate(node_n):
    Pxx[l,0] = alpha if (c[0]-1,c[1]+1) in tensor else Pxx[l,0]
    Pxx[l,1] = beta if (c[0],c[1]+1) in tensor else Pxx[l,1]
    Pxx[l,2] = gamma if (c[0]+1,c[1]+1) in tensor else Pxx[l,2]
    Pxx[l,3] = delta if (c[0]-1,c[1]) in tensor else Pxx[l,3]
    Pxx[l,4] = epsi
    Pxx[l,5] = psi if (c[0]+1,c[1]) in tensor else Pxx[l,5]
    Pxx[l,6] = phi if (c[0]-1,c[1]-1) in tensor else Pxx[l,6]
    Pxx[l,7] = chi if (c[0],c[1]-1) in tensor else Pxx[l,7]
    Pxx[l,8] = zeta if (c[0]+1,c[1]-1) in tensor else Pxx[l,8]
    print(l,c,Pxx[l])


"""
    



""" Required values for now """
# to make things more clear, I'll name the indices

s = 5
w = np.linspace(-s,s,2*s+1)
xs,ys = np.meshgrid(w,w)
grid = (xs,np.flipud(ys)) ##why do we need this flipup? It is just
#because I need to start from the top-left corner or the grid to
#solve it or I am gonna have a weird matrix. 

e = np.array([1,2,3,4,5,6,7,8,9])


tensor = {(i,j):e for i,j in zip(grid[0].flat,grid[1].flat)}

dx = 1
dy = 1

exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz = (0,1,2,3,4,5,6,7,8)




""" Get the 9 coefficients """
#coefficient for Ex 
"""
E_o = cost*a +1/e*d2a/dx2+1/e*d2c/dxy-ß**2

alpha = - (c/e)/(4*dx*dy) 
beta = 1/dy**2 + 1/e*dc/dx*1/dy
gamma = (c/e)/(4*dx*dy)
delta = (a/e)/dx**2
epsi = -2*(a/e)/dx**2 - 1/e*(2*da/dx+dc/dy)*1/dx - 1/e*dc/dx*1/dy -2/dy**2 + E_o
psi = (a/e)/dx**2 + 1/e*(2*da/dx+dc/dy)*1/dx 
phi = (c/e)/(4*dx*dy)
chi = 1/dy**2
zeta = -(c/e)/(4*dx*dy) 
"""

alpha, beta, gamma, delta, epsi, psi, phi, chi, zeta = 1,2,3,4,5,6,7,8,9


# tensor is my dictionary with, at each key (x,y), the respective permit_tensor
node_n = tuple(tensor.keys())
Pxx = np.zeros((len(node_n),9))


for l,c in enumerate(node_n):
    Pxx[l,0] = alpha if (c[0]-1,c[1]+1) in tensor else Pxx[l,0]
    Pxx[l,1] = beta if (c[0],c[1]+1) in tensor else Pxx[l,1]
    Pxx[l,2] = gamma if (c[0]+1,c[1]+1) in tensor else Pxx[l,2]
    Pxx[l,3] = delta if (c[0]-1,c[1]) in tensor else Pxx[l,3]
    Pxx[l,4] = epsi
    Pxx[l,5] = psi if (c[0]+1,c[1]) in tensor else Pxx[l,5]
    Pxx[l,6] = phi if (c[0]-1,c[1]-1) in tensor else Pxx[l,6]
    Pxx[l,7] = chi if (c[0],c[1]-1) in tensor else Pxx[l,7]
    Pxx[l,8] = zeta if (c[0]+1,c[1]-1) in tensor else Pxx[l,8]
    print(l,c,Pxx[l])


    
    
    
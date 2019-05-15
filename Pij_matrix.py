# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:33:37 2019

@author: 2353588g
"""
import numpy as np
"""
Try to write the matrices ><

4 functions, each retrieving the matrix. My sistem is 
    
                                         ┌          ┐
         d    / u \     d²     / u \     | Pxx  Pxy |  / u \
    2iß ───  |     | ─ ────   |     |  = |          | |     |
         dz   \ v /     dz²    \ v /     | Pyx  Pyy |  \ v /
                                         └          ┘

And I am going to build them all ><

    Notes: since there are more and more indices, I was thinking of having a
    1D array instead of a 2D one for the tensor:
        tensor(x,y) = [e_xx, e_xy, e_xz, e_yx, e_yy, ... ] 
"""
#data needed to do something lol
omega = 2
mu_o = 1
epsilon_o = 2

cost = omega ** 2 * mu_o * epsilon_o
# to make things more clear, I'll name the indices
exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz = (0,1,2,3,4,5,6,7,8)


s = 5
w = np.linspace(-s,s,2*s+1)
xs,ys = np.meshgrid(w,w)
grid = (xs,ys)

e = np.array([1,2,3,4,5,6,7,8,9])


tensor = {(i,j):e for i,j in zip(grid[0].flat,grid[1].flat)}

dx = 1
dy = 1



#per_dx2, per_dy2, per_dxy, per_dx, per_dy = finite_diff(tensor,grid,dx,dy)

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
Pxx = np.zeros((len(node_n),len(node_n)))


""" this actually builds the whole matrix changin values at each step """

#From Pxx = zeros(), at each lines puts what it's due ><
for l,c in enumerate(node_n):
    Pxx[l,l] = epsi
    Pxx[l,l-1] = delta if (c[0]-1,c[1]) in tensor else Pxx[l,l-1]
   # Pxx[l,l+1] = psi if (c[0]+1,c[1]) in tensor else Pxx[l,l+1]
    
       
    #Pxx[l,l-1] = delta [(c[0]-1,c[1]) in tensor]
    
    
        
    
    
    
    








""" 

WRONG

This would be awesome if I only new my E-field. But it's actually what I 
am looking for lol. This piece of software is, therefore, useless. 

Ex_dx2, Ex_dy2, Ex_dxy, Ex_dx, Ex_dy = finite_diff(Ex,grid,dx,dy)
Ey_dx2, Ey_dy2, Ey_dxy, Ey_dx, Ey_dy = finite_diff(Ey,grid,dx,dy)



#I am also combing the tuple (x,y) in the variable g
g = (i,j)

Pxx[g] = tensor[g][exx]/tensor[g][ezz] * Ex_dx2[g] + Ex_dy2[g] \
        + tensor[g][eyx]/tensor[g][ezz] * Ex_dxy[g] \
        + 1/tensor[g][ezz] * (2 * per_dx[g][exx] + per_dy[g][eyx]) * Ex_dx[g] \
        + 1/tensor[g][ezz] * per_dx[g][eyx] * Ex_dy[g] \
        + Ex[g] * (cost*tensor[g][exx]+ 1/tensor[g][ezz]*per_dx2[g][exx] + 1/tensor[g][ezz]*per_dxy[g][eyx] - beta**2)

Pxy[g] = tensor[g][exy]/tensor[g][ezz] * Ey_dx2[g] \
        - (1-tensor[g][eyy]/tensor[g][ezz]) * Ey_dxy[g] \
        + 1/tensor[g][ezz] * per_dx[eyy] * Ey_dy[g] \
        + 1/tensor[g][ezz] * (2 * per_dx[g][exy] + per_dy[g][eyy]) * Ey_dx[g] \
        + Ey[g] * (cost*tensor[g][exy]+ 1/tensor[g][ezz]*per_dx2[g][exy] + 1/tensor[g][ezz]*per_dxy[g][eyy])

Pyx[g] = tensor[g][eyx]/tensor[g][ezz] * Ex_dy2[g] \
        - (1-tensor[g][exx]/tensor[g][ezz]) * Ex_dxy[g] \
        + 1/tensor[g][ezz] * per_dy[g][exx] * Ex_dx[g] \
        + 1/tensor[g][ezz] * (per_dx[g][exx] + 2*per_dy[g][eyx]) * Ex_dy[g] \
        + Ex[g] * (cost*tensor[g][eyx]+ 1/tensor[g][ezz]*per_dxy[g][exx] + 1/tensor[g][ezz]*per_dy2[g][eyx]) 
        
Pyy[g] = tensor[g][eyy]/tensor[g][ezz] * Ey_dy2[g] + Ey_dx2[g] \
        + tensor[g][exy]/tensor[g][ezz] * Ey_dxy[g] \
        + 1/tensor[g][ezz] * (per_dx[g][exy] + 2*per_dy[g][eyy]) * Ey_dy[g] \
        + 1/tensor[g][ezz] * perm_dy[g][exy] * Ey_dx[g] \
        + ey[g] * (cost*tensor[g][eyy]+ 1/tensor[g][ezz]*per_dxy[g][exy] + 1/tensor[g][ezz]*per_dy2[g][eyy] - beta**2)
        
"""

""" old """
"""
Pxx(g) = tensor[g][exx]/tensor[g][ezz] * D2x(u(g)) + D2y(u(g)) \
        + tensor[g][eyx]/tensor[g][ezz] * D2xy(u(g)) \
        + 1/tensor[g][ezz] * (2 * Dx(tensor[g])[exx] + Dy(tensor[g])[eyx]) * Dx(u(g)) \
        + 1/tensor[g][ezz] * Dx(tensor[g])[eyx] * Dy(u(g)) \
        + u(g) * (cost*tensor[g][exx]+ 1/tensor[g][ezz]*D2x(tensor[g])[exx] + 1/tensor[g][ezz]*D2xy(tensor[g])[eyx] - beta**2)
        
Pxy(g) = tensor[g][exy]/tensor[g][ezz] * D2x(v(g)) \
        - (1-tensor[g][eyy]/tensor[g][ezz]) * D2xy(v(g)) \
        + 1/tensor[g][ezz] * Dx(tensor[g])[eyy] * Dy(v(g)) \
        + 1/tensor[g][ezz] * (2 * Dx(tensor[g])[exy] + Dy(tensor[g])[eyy]) * Dx(v(g)) \
        + v(g) * (cost*tensor[g][exy]+ 1/tensor[g][ezz]*D2x(tensor[g])[exy] + 1/tensor[g][ezz]*D2xy(tensor[g])[eyy])
        
Pyx(g) = tensor[g][eyx]/tensor[g][ezz] * D2y(u(g)) \
        - (1-tensor[g][exx]/tensor[g][ezz]) * D2xy(u(g)) \
        + 1/tensor[g][ezz] * Dy(tensor[g])[exx] * Dx(u(g)) \
        + 1/tensor[g][ezz] * (Dx(tensor[g])[exx] + 2*Dy(tensor[g])[eyx]) * Dy(u(g)) \
        + u(g) * (cost*tensor[g][eyx]+ 1/tensor[g][ezz]*D2xy(tensor[g])[exx] + 1/tensor[g][ezz]*D2y(tensor[g])[eyx])    
        
Pyy(g) = tensor[g][eyy]/tensor[g][ezz] * D2y(v(g)) + D2x(v(g)) \
        + tensor[g][exy]/tensor[g][ezz] * D2xy(v(g)) \
        + 1/tensor[g][ezz] * (Dx(tensor[g])[exy] + 2*Dy(tensor[g])[eyy]) * Dy(v(g)) \
        + 1/tensor[g][ezz] * Dy(tensor[g])[exy] * Dx(v(g)) \
        + v(g) * (cost*tensor[g][eyy]+ 1/tensor[g][ezz]*D2xy(tensor[g])[exy] + 1/tensor[g][ezz]*D2y(tensor[g])[eyy] - beta**2)
"""


        
# problem is da/dx and so on. Is it better to have indipendent exx eyy exy
# matrices or evaluete the difference of the whole tensor and only takes one?
# here I am evaluation this difference every time, which is quite inconvienent.
# I could actually evaluate the differences just one, store them all in matrices
# and pick the value I need every time 
        


    
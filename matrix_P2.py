# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:29:19 2019

@author: 2353588g


It should return the matrix P 
"""
import numpy as np
from scipy.sparse import vstack, hstack
import finite_diff
import build_matrix
"""
This function will make a (nodes x 9) matrix which is then used to make a
sparsed banded matrix. The structure is as follows:
    
    i-1,j+1      i,j+1    1+1,j+1
     (0-α)      (1-β)      (2-γ)
    i-1,j        i,j      1+1,j  
     (3-δ)      (4-ε)      (5-ψ) 
    i-1,j-1      i,j-1    1+1,j-1
     (6-φ)      (7-χ)      (8-ζ)  
      
where the number between () is the vector position of each component.



#%% Required initial values:
s = 2
w = np.linspace(-s,s,2*s+1)
xs,ys = np.meshgrid(w,w)
grid = (xs,np.flipud(ys)) 


e = np.array([10,5,5,5,10,5,5,5,15])
e = np.array([10,5,0,5,10,0,0,0,15])
tensor = {(i,j):e for i,j in zip(grid[0].flat,grid[1].flat)}


print('ok')
#%% for the components:

omega = 2
mu_o = 1
epsilon_o = 2

cost = omega ** 2 * mu_o * epsilon_o
ß = 1
# to make things more clear, I'll name the indices
exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz = (0,1,2,3,4,5,6,7,8)

dx = 1
dy = 1
"""
#%%


def P(tensor,grid,dx,dy):
    """
    Permittivity tensor = tensor
    """   
    
    node_n = tuple(tensor.keys())
    
    #Preallocation
    Pxx = np.zeros((len(node_n),9))
    Pxy = np.zeros((len(node_n),9))
    Pyx = np.zeros((len(node_n),9))
    Pyy = np.zeros((len(node_n),9))

    #%% SORVOLIAMO QUI EPR ORA
    per_dx2, per_dy2, per_dxy, per_dx, per_dy = finite_diff.finite_diff(tensor,grid,dx,dy)

    """ make the 9 diagonals for the 4 matrices """
    #add indices to make it easier to read
    exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz = (0,1,2,3,4,5,6,7,8)
    
    
    
    #%% they have to disappeaer soon
    omega = 2
    mu_o = 1
    epsilon_o = 2

    cost = omega ** 2 * mu_o * epsilon_o
    ß = 1

    #%% definition of function check
    def check(val,tensor):
        cont = 0
        lista = []
        for i in (-0.1,0,0.1):
            for j in (-0.1,0,0.1):
                lista.append((round(val[0]+i,1),round(val[1]+j,1)))        
        #print(lista)
        for a in lista:
            if a in tensor:
                cont += 1
        #print(cont)
        return cont
    
    #%% make matrix P
    conteggio_elementi = 0
    
    #%% Make Pxx
    
    for l,c in enumerate(node_n):
        E_o     = cost*tensor[c][exx]+ 1/tensor[c][ezz]*(per_dx2[c][exx] \
                    + per_dxy[c][eyx])-ß**2
        alpha   = - (tensor[c][eyx]/tensor[c][ezz])/(4*dx*dy) 
        beta    = 1/dy**2 + 1/tensor[c][ezz]*per_dx[c][eyx]*1/dy
        gamma   = (tensor[c][eyx]/tensor[c][ezz])/(4*dx*dy)
        delta   = tensor[c][exx]/tensor[c][ezz]/dx**2
        epsi    = -2*tensor[c][exx]/tensor[c][ezz]/dx**2 \
                    - 1/tensor[c][ezz] * (2 * per_dx[c][exx] + per_dy[c][eyx])*1/dx \
                    - 1/tensor[c][ezz] * per_dx[c][eyx]/dy -2/dy**2 + E_o
        psi     = tensor[c][exx]/tensor[c][ezz]/dx**2 + \
                    1/tensor[c][ezz] * (2 * per_dx[c][exx] + per_dy[c][eyx])*1/dx 
        phi     = (tensor[c][eyx]/tensor[c][ezz])/(4*dx*dy)
        chi     = 1/dy**2
        zeta    = -(tensor[c][eyx]/tensor[c][ezz])/(4*dx*dy) 
        
        #print(l,c)
        Pxx[l,0] = alpha    if check((c[0]-dx,c[1]+dy),node_n) > 0  else Pxx[l,0]
        #print(np.round(c[0]-dx,1),np.round(c[1]+dy,1),Pxx[l,0])
        Pxx[l,1] = beta     if check((c[0],c[1]+dy),node_n) > 0     else Pxx[l,1]
        Pxx[l,2] = gamma    if check((c[0]+dx,c[1]+dy),node_n) > 0  else Pxx[l,2]
        Pxx[l,3] = delta    if check((c[0]-dx,c[1]),node_n) > 0     else Pxx[l,3]
        Pxx[l,4] = epsi
        Pxx[l,5] = psi      if check((c[0]+dx,c[1]),node_n) > 0     else Pxx[l,5]
        Pxx[l,6] = phi      if check((c[0]-dx,c[1]-dy),node_n) > 0  else Pxx[l,6]
        Pxx[l,7] = chi      if check((c[0],c[1]-dy),node_n) > 0     else Pxx[l,7]
        Pxx[l,8] = zeta     if check((c[0]+dx,c[1]-dy),node_n) > 0  else Pxx[l,8]
        #print(alpha,beta,gamma,delta,epsi,psi,phi,chi,zeta)
    h = 0
    for el in np.nditer(Pxx):
        if el != 0:
            h += 1
    conteggio_elementi += h    
    print('number of elements in Pxx is {}'.format(h))   
    print('Pxx ok')
    
    #%% Make Pxy 
    for l,c in enumerate(node_n):
        E_o     = cost*tensor[c][exy]+ 1/tensor[c][ezz]*(per_dx2[c][exy] \
                    + per_dxy[c][eyy])
        alpha   = (1-tensor[c][eyy]/tensor[c][ezz])/(4*dx*dy) 
        beta    = 1/tensor[c][ezz]*per_dx[c][eyy]*1/dy
        gamma   = - (1-tensor[c][eyy]/tensor[c][ezz])/(4*dx*dy)
        delta   = tensor[c][exy]/tensor[c][ezz]/dx**2
        epsi    = -2*tensor[c][exy]/tensor[c][ezz]/dx**2 \
                    - 1/tensor[c][ezz] * (2 * per_dx[c][exy] + per_dy[c][eyy])*1/dx \
                    - 1/tensor[c][ezz] * per_dx[c][eyy]/dy + E_o
        psi     = tensor[c][exy]/tensor[c][ezz]/dx**2 + \
                    1/tensor[c][ezz] * (2 * per_dx[c][exy] + per_dy[c][eyy])*1/dx 
        phi     = - (1-tensor[c][eyy]/tensor[c][ezz])/(4*dx*dy)
        chi     = 0
        zeta    = (1-tensor[c][eyy]/tensor[c][ezz])/(4*dx*dy) 
        
        Pxy[l,0] = alpha    if check((c[0]-dx,c[1]+dy),node_n) > 0  else Pxy[l,0]
        Pxy[l,1] = beta     if check((c[0],c[1]+dy),node_n) > 0     else Pxy[l,1]
        Pxy[l,2] = gamma    if check((c[0]+dx,c[1]+dy),node_n) > 0  else Pxy[l,2]
        Pxy[l,3] = delta    if check((c[0]-dx,c[1]),node_n) > 0     else Pxy[l,3]
        Pxy[l,4] = epsi
        Pxy[l,5] = psi      if check((c[0]+dx,c[1]),node_n) > 0     else Pxy[l,5]
        Pxy[l,6] = phi      if check((c[0]-dx,c[1]-dy),node_n) > 0  else Pxy[l,6]
        Pxy[l,7] = chi      if check((c[0],c[1]-dy),node_n) > 0     else Pxy[l,7]
        Pxy[l,8] = zeta     if check((c[0]+dx,c[1]-dy),node_n) > 0  else Pxy[l,8]
    
    h = 0
    for el in np.nditer(Pxy):
        if el != 0:
            h += 1
    conteggio_elementi += h
    print('number of elements in Pxy is {}'.format(h))  
    print('Pxy ok')    
    #%% Make Pyx 
    for l,c in enumerate(node_n):
        E_o     = cost*tensor[c][eyx]+ 1/tensor[c][ezz]*(per_dxy[c][exx] \
                    + per_dy2[c][eyx])
        alpha   = (1-tensor[c][exx]/tensor[c][ezz])/(4*dx*dy) 
        beta    = tensor[c][eyx]/tensor[c][ezz]/dy**2 \
                    + 1/tensor[c][ezz]*(per_dx[c][exx]+2*per_dy[c][eyx])*1/dy
        gamma   = - (1-tensor[c][exx]/tensor[c][ezz])/(4*dx*dy)
        delta   = 0
        epsi    = -2*tensor[c][eyx]/tensor[c][ezz]/dx**2 \
                    - 1/tensor[c][ezz] * (2 * per_dy[c][eyx] + per_dx[c][exx])*1/dy \
                    - 1/tensor[c][ezz] * per_dy[c][exx]/dx + E_o
        psi     = 1/tensor[c][ezz] * per_dy[c][exx]/dx 
        phi     = - (1-tensor[c][exx]/tensor[c][ezz])/(4*dx*dy)
        chi     = tensor[c][eyx]/tensor[c][ezz]/dy**2
        zeta    = (1-tensor[c][exx]/tensor[c][ezz])/(4*dx*dy) 
        
        Pyx[l,0] = alpha    if check((c[0]-dx,c[1]+dy),node_n) > 0  else Pyx[l,0]
        Pyx[l,1] = beta     if check((c[0],c[1]+dy),node_n) > 0     else Pyx[l,1]
        Pyx[l,2] = gamma    if check((c[0]+dx,c[1]+dy),node_n) > 0  else Pyx[l,2]
        Pyx[l,3] = delta    if check((c[0]-dx,c[1]),node_n) > 0     else Pyx[l,3]
        Pyx[l,4] = epsi
        Pyx[l,5] = psi      if check((c[0]+dx,c[1]),node_n) > 0     else Pyx[l,5]
        Pyx[l,6] = phi      if check((c[0]-dx,c[1]-dy),node_n) > 0  else Pyx[l,6]
        Pyx[l,7] = chi      if check((c[0],c[1]-dy),node_n) > 0     else Pyx[l,7]
        Pyx[l,8] = zeta     if check((c[0]+dx,c[1]-dy),node_n) > 0  else Pyx[l,8]    
    h = 0
    for el in np.nditer(Pyx):
        if el != 0:
            h += 1
    conteggio_elementi += h
    print('number of elements in Pxy is {}'.format(h))  
    print('Pyx ok')
    #%% Make Pyy 
    for l,c in enumerate(node_n):
        E_o     = cost*tensor[c][eyy]+ 1/tensor[c][ezz]*(per_dxy[c][exy] \
                    + per_dy2[c][eyy])-ß**2
        alpha   = - (tensor[c][exy]/tensor[c][ezz])/(4*dx*dy) 
        beta    = tensor[c][eyy]/tensor[c][ezz]/dy**2 \
                    + 1/tensor[c][ezz]*(per_dx[c][exy]+2*per_dy[c][eyy])*1/dy
        gamma   = (tensor[c][exy]/tensor[c][ezz])/(4*dx*dy)
        delta   = 1/dx**2
        epsi    = -2*tensor[c][eyy]/tensor[c][ezz]/dx**2 -2/dx**2\
                    - 1/tensor[c][ezz] * (2 * per_dy[c][eyy] + per_dx[c][exy])*1/dy \
                    - 1/tensor[c][ezz] * per_dy[c][exy]/dx + E_o
        psi     = 1/tensor[c][ezz] * per_dy[c][exy]/dx +1/dx**2 
        phi     = (tensor[c][exy]/tensor[c][ezz])/(4*dx*dy)
        chi     = tensor[c][eyy]/tensor[c][ezz]/dy**2
        zeta    = - (tensor[c][exy]/tensor[c][ezz])/(4*dx*dy)
        
        Pyy[l,0] = alpha    if check((c[0]-dx,c[1]+dy),node_n)  else Pyy[l,0]
        Pyy[l,1] = beta     if check((c[0],c[1]+dy),node_n)     else Pyy[l,1]
        Pyy[l,2] = gamma    if check((c[0]+dx,c[1]+dy),node_n)  else Pyy[l,2]
        Pyy[l,3] = delta    if check((c[0]-dx,c[1]),node_n)     else Pyy[l,3]
        Pyy[l,4] = epsi
        Pyy[l,5] = psi      if check((c[0]+dx,c[1]),node_n)     else Pyy[l,5]
        Pyy[l,6] = phi      if check((c[0]-dx,c[1]-dy),node_n)  else Pyy[l,6]
        Pyy[l,7] = chi      if check((c[0],c[1]-dy),node_n)     else Pyy[l,7]
        Pyy[l,8] = zeta     if check((c[0]+dx,c[1]-dy),node_n)  else Pyy[l,8]
    
    h = 0
    for el in np.nditer(Pyy):
        if el != 0:
            h += 1
    conteggio_elementi += h
    print('number of elements in Pyy is {}'.format(h))  
    print('Pyy ok')
    #%% make block matrix
    #
    #Tbh I don't know if it is easier to build the matrix first and then sparse it
    #or if it's better to actually create il with diagonal conditions. 
    #Either way it looks quite expensive ><
    s = int((np.sqrt(len(node_n))-1)/2)
    PXX = build_matrix.build_matrix(s,Pxx).diagonal_position()
    PXY = build_matrix.build_matrix(s,Pxy).diagonal_position()
    PYX = build_matrix.build_matrix(s,Pyx).diagonal_position()
    PYY = build_matrix.build_matrix(s,Pyy).diagonal_position()
    
    print('PXX : {} sparse matrix with {} stored elements.'.format(PXX.shape, PXX.nnz))
    print('PXY : {} sparse matrix with {} stored elements.'.format(PXY.shape, PXY.nnz))
    print('PYX : {} sparse matrix with {} stored elements.'.format(PYX.shape, PYX.nnz))
    print('PYY : {} sparse matrix with {} stored elements.'.format(PYY.shape, PYY.nnz))
    print('sparse ok')
    #%% Stack the matrices to get:

    """
                                ┌          ┐
    [PXX] [PXY]      [PXX PXY]      | PXX  PXY |
                -->             --> |          |
    [PYX] [PYY]      [PYX PYY]      | PYX  PYY |
                                    └          ┘
    """
    """ I AM LOSING ELEMENTS!!!!"""
    """
    No I am not. It's just because at the beginning, PXX PXY return the number of
    values non-zero, but also the significant zeros. When I combine them, I end up
    just taking the non zero elements. This is shown in different way, I've done it
    actually printing these values so don't worry. To count non zero values I've
    just used
    
    k = 0
    for i in PXX.flat:
        if i != 0:
            k += 1

    """
    
    PXXPXY = hstack([PXX,PXY])
    PYXPYY = hstack([PYX,PYY])
    
    print('h-stak ok')
    """
    done for s = 1, which means -1/0/+1
    PXX
    <9x9 sparse matrix of type '<class 'numpy.float64'>'
    	with 61 stored elements (9 diagonals) in DIAgonal format>  		++++ 39 non zero
    array([[ 5.03, -0.22,  0.  ,  1.  , -0.11,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.11,  5.11, -0.22,  0.11,  1.  , -0.11,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.11,  5.53,  0.  ,  0.11,  1.  ,  0.  ,  0.  ,  0.  ],
           [ 1.  ,  0.11,  0.  ,  4.78,  0.11,  0.  ,  1.  , -0.11,  0.  ],
           [-0.11,  1.  ,  0.11,  0.11,  4.78,  0.11,  0.11,  1.  , -0.11],
           [ 0.  , -0.11,  0.67,  0.  ,  0.11,  5.11,  0.  ,  0.11,  1.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ,  0.11,  0.  ,  4.86,  0.11,  0.  ],
           [ 0.  ,  0.  ,  0.  , -0.11,  1.  ,  0.11,  0.11,  4.78,  0.11],
           [ 0.  ,  0.  ,  0.  ,  0.  , -0.11,  0.67,  0.  ,  0.11,  5.03]])
    
    PXY
    <9x9 sparse matrix of type '<class 'numpy.float64'>'				++++ 39 non zero
    	with 61 stored elements (9 diagonals) in DIAgonal format>		bands:  9 + 2*(8 + 5 + 6 + 7) = 88 
    array([[15.78, -0.22,  0.  ,  0.  ,  0.11,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.22, 16.  , -0.22, -0.11,  0.  ,  0.11,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.22, 16.67,  0.  , -0.11,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  , -0.11,  0.  , 15.44,  0.22,  0.  ,  0.  ,  0.11,  0.  ],
           [ 0.11,  0.  , -0.11,  0.22, 15.56,  0.22, -0.11,  0.  ,  0.11],
           [ 0.  ,  0.11, -0.44,  0.  ,  0.22, 16.11,  0.  , -0.11,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  , -0.11,  0.  , 15.56,  0.22,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.11,  0.  , -0.11,  0.22, 15.56,  0.22],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.11, -0.44,  0.  ,  0.22, 16.  ]])
    
    
    PXXPXY
    <9x18 sparse matrix of type '<class 'numpy.float64'>'
    	with 88 stored elements in COOrdinate format>
    array([[ 5.03, -0.22,  0.  ,  1.  , -0.11,  0.  ,  0.  ,  0.  ,  0.  , 15.78, -0.22,  0.  ,  0.  ,  0.11,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.11,  5.11, -0.22,  0.11,  1.  , -0.11,  0.  ,  0.  ,  0.  ,  0.22, 16.  , -0.22, -0.11,  0.  ,  0.11,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.11,  5.53,  0.  ,  0.11,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.22, 16.67,  0.  , -0.11,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 1.  ,  0.11,  0.  ,  4.78,  0.11,  0.  ,  1.  , -0.11,  0.  ,  0.  , -0.11,  0.  , 15.44,  0.22,  0.  ,  0.  ,  0.11,  0.  ],
           [-0.11,  1.  ,  0.11,  0.11,  4.78,  0.11,  0.11,  1.  , -0.11,  0.11,  0.  , -0.11,  0.22, 15.56,  0.22, -0.11,  0.  ,  0.11],
           [ 0.  , -0.11,  0.67,  0.  ,  0.11,  5.11,  0.  ,  0.11,  1.  ,  0.  ,  0.11, -0.44,  0.  ,  0.22, 16.11,  0.  , -0.11,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ,  0.11,  0.  ,  4.86,  0.11,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.11,  0.  , 15.56,  0.22,  0.  ],
           [ 0.  ,  0.  ,  0.  , -0.11,  1.  ,  0.11,  0.11,  4.78,  0.11,  0.  ,  0.  ,  0.  ,  0.11,  0.  , -0.11,  0.22, 15.56,  0.22],
           [ 0.  ,  0.  ,  0.  ,  0.  , -0.11,  0.67,  0.  ,  0.11,  5.03,  0.  ,  0.  ,  0.  ,  0.  ,  0.11, -0.44,  0.  ,  0.22, 16.  ]])
    """
    
    #k = 0
    #
    #
    #def counter(ko,M):
    #    for i in M.toarray().flat:
    #        if i != 0:
    #            ko += 1
    #    assert ko > 0
    #    return ko
    #            
    #k = counter(k,PXX) 
    #k = counter(k,PXY) 
    #k = counter(k,PYX)
    #k = counter(k,PYY)
    #
    #
    #l = 0
    P = vstack([PXXPXY,PYXPYY])
    print('Matrix P - built')
    #print('Number of elements: {}'.format(conteggio_elementi))
    return P, conteggio_elementi
#%% I WANT TO print this P just to see how it looks like
# THIS IS OK for s = 5, so max 242 elements.

"""
import matplotlib.pyplot as plt

U = P.toarray()

with np.nditer(U,op_flags=['readwrite']) as it:
    for x in it:
        if x[...] != 0:
            x[...] = 1

#plt.pcolor(U)

# get dimensions of U
a,b = U.shape

w = np.linspace(-a/2,a,a+1)
xr,yr = np.meshgrid(w,w)
fig = plt.figure(figsize=(10,10))
plt.pcolor(xr,np.flipud(yr),U,cmap = 'Purples') 

    
#%% Solve P with eigenvalues
# Get larger 2 eigenvalues/vectors

from scipy.sparse.linalg import eigs
evals, evecs = eigs(P,2,which='LM')

# supposing a dz = 2 (step)
# Solution for y(z) with z = dz = 2 and considering yo as previous calculated
# field while y'(0) = 0 for now. 

dz = 2
yo = np.ones([50,1])
ß = 50

rad = np.sqrt(ß**2+evals[0]**2)

y2 = yo/2*(1-ß/rad)*np.exp(1j*(ß+rad)*dz) \
        + yo/2*(1+ß/rad)*np.exp(1j*(ß-rad)*dz)


# return y
"""
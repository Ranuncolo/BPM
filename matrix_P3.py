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

"""



def P(tensor,grid,dx,dy,n):
    """
    Permittivity tensor = tensor
    """   
    
    node_n = tuple(tensor.keys())
    ran = np.arange(0,len(node_n))
    lista = tuple(zip(ran,node_n))
    
    #limits
    neg_lim = node_n[0][0] - dx/2
    pos_lim = node_n[0][1] + dx/2
    print(neg_lim,pos_lim)
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
    
    for l,c in enumerate(lista):
        xy = c[1]
        E_o     = cost*tensor[xy][exx]+ 1/tensor[xy][ezz]*(per_dx2[xy][exx] \
                    + per_dxy[xy][eyx])-ß**2
        alpha   = - (tensor[xy][eyx]/tensor[xy][ezz])/(4*dx*dy) 
        beta    = 1/dy**2 + 1/tensor[xy][ezz]*per_dx[xy][eyx]*1/dy
        gamma   = (tensor[xy][eyx]/tensor[xy][ezz])/(4*dx*dy)
        delta   = tensor[xy][exx]/tensor[xy][ezz]/dx**2
        epsi    = -2*tensor[xy][exx]/tensor[xy][ezz]/dx**2 \
                    - 1/tensor[xy][ezz] * (2 * per_dx[xy][exx] + per_dy[xy][eyx])*1/dx \
                    - 1/tensor[xy][ezz] * per_dx[xy][eyx]/dy -2/dy**2 + E_o
        psi     = tensor[xy][exx]/tensor[xy][ezz]/dx**2 + \
                    1/tensor[xy][ezz] * (2 * per_dx[xy][exx] + per_dy[xy][eyx])*1/dx 
        phi     = (tensor[xy][eyx]/tensor[xy][ezz])/(4*dx*dy)
        chi     = 1/dy**2
        zeta    = -(tensor[xy][eyx]/tensor[xy][ezz])/(4*dx*dy) 
        

        Pxx[l,0] = alpha    if l-n-1 in ran and c[1][0] - dx > neg_lim  else Pxx[l,0]
        Pxx[l,1] = beta     if l-n   in ran                             else Pxx[l,1]
        Pxx[l,2] = gamma    if l-n+1 in ran and c[1][0] + dx < pos_lim  else Pxx[l,2]
        Pxx[l,3] = delta    if l-1   in ran and c[1][0] - dx > neg_lim  else Pxx[l,3]
        Pxx[l,4] = epsi
        Pxx[l,5] = psi      if l+1   in ran and c[1][0] + dx < pos_lim  else Pxx[l,5]
        #print(l+1,c[1][0],c[1][0]+dx)
        Pxx[l,6] = phi      if l+n-1 in ran and c[1][0] - dx > neg_lim  else Pxx[l,6]
        Pxx[l,7] = chi      if l+n   in ran                             else Pxx[l,7]
        Pxx[l,8] = zeta     if l+n+1 in ran and c[1][0] + dx < pos_lim  else Pxx[l,8]
        
    h = 0
    for el in np.nditer(Pxx):
        if el != 0:
            h += 1
    conteggio_elementi += h    
    print('number of elements in Pxx is {}'.format(h))   
    #print(Pxx)
    print('Pxx ok')
    
    #%% Make Pxy 
    for l,c in enumerate(lista):
        xy = c[1]
        E_o     = cost*tensor[xy][exy]+ 1/tensor[xy][ezz]*(per_dx2[xy][exy] \
                    + per_dxy[xy][eyy])
        alpha   = (1-tensor[xy][eyy]/tensor[xy][ezz])/(4*dx*dy) 
        beta    = 1/tensor[xy][ezz]*per_dx[xy][eyy]*1/dy
        gamma   = - (1-tensor[xy][eyy]/tensor[xy][ezz])/(4*dx*dy)
        delta   = tensor[xy][exy]/tensor[xy][ezz]/dx**2
        epsi    = -2*tensor[xy][exy]/tensor[xy][ezz]/dx**2 \
                    - 1/tensor[xy][ezz] * (2 * per_dx[xy][exy] + per_dy[xy][eyy])*1/dx \
                    - 1/tensor[xy][ezz] * per_dx[xy][eyy]/dy + E_o
        psi     = tensor[xy][exy]/tensor[xy][ezz]/dx**2 + \
                    1/tensor[xy][ezz] * (2 * per_dx[xy][exy] + per_dy[xy][eyy])*1/dx 
        phi     = - (1-tensor[xy][eyy]/tensor[xy][ezz])/(4*dx*dy)
        chi     = 0
        zeta    = (1-tensor[xy][eyy]/tensor[xy][ezz])/(4*dx*dy) 
        print(tensor[xy][eyy],tensor[xy][ezz],dx,dy,zeta)
        
        Pxy[l,0] = alpha    if l-n-1 in ran and c[1][0] - dx > neg_lim  else Pxy[l,0]
        Pxy[l,1] = beta     if l-n   in ran                             else Pxy[l,1]
        Pxy[l,2] = gamma    if l-n+1 in ran and c[1][0] + dx < pos_lim  else Pxy[l,2]
        Pxy[l,3] = delta    if l-1   in ran and c[1][0] - dx > neg_lim  else Pxy[l,3]
        Pxy[l,4] = epsi
        Pxy[l,5] = psi      if l+1   in ran and c[1][0] + dx < pos_lim  else Pxy[l,5]
        Pxy[l,6] = phi      if l+n-1 in ran and c[1][0] - dx > neg_lim  else Pxy[l,6]
        Pxy[l,7] = chi      if l+n   in ran                             else Pxy[l,7]
        Pxy[l,8] = zeta     if l+n+1 in ran and c[1][0] + dx < pos_lim  else Pxy[l,8]
    h = 0
    for el in np.nditer(Pxy):
        if el != 0:
            h += 1
    conteggio_elementi += h
    print('number of elements in Pxy is {}'.format(h))  
    print('Pxy ok')    
    #%% Make Pyx 
    for l,c in enumerate(lista):
        xy = c[1]
        E_o     = cost*tensor[xy][eyx]+ 1/tensor[xy][ezz]*(per_dxy[xy][exx] \
                    + per_dy2[xy][eyx])
        alpha   = (1-tensor[xy][exx]/tensor[xy][ezz])/(4*dx*dy) 
        beta    = tensor[xy][eyx]/tensor[xy][ezz]/dy**2 \
                    + 1/tensor[xy][ezz]*(per_dx[xy][exx]+2*per_dy[xy][eyx])*1/dy
        gamma   = - (1-tensor[xy][exx]/tensor[xy][ezz])/(4*dx*dy)
        delta   = 0
        epsi    = -2*tensor[xy][eyx]/tensor[xy][ezz]/dx**2 \
                    - 1/tensor[xy][ezz] * (2 * per_dy[xy][eyx] + per_dx[xy][exx])*1/dy \
                    - 1/tensor[xy][ezz] * per_dy[xy][exx]/dx + E_o
        psi     = 1/tensor[xy][ezz] * per_dy[xy][exx]/dx 
        phi     = - (1-tensor[xy][exx]/tensor[xy][ezz])/(4*dx*dy)
        chi     = tensor[xy][eyx]/tensor[xy][ezz]/dy**2
        zeta    = (1-tensor[xy][exx]/tensor[xy][ezz])/(4*dx*dy) 
        
        Pyx[l,0] = alpha    if l-n-1 in ran and c[1][0] - dx > neg_lim  else Pyx[l,0]
        Pyx[l,1] = beta     if l-n   in ran                             else Pyx[l,1]
        Pyx[l,2] = gamma    if l-n+1 in ran and c[1][0] + dx < pos_lim  else Pyx[l,2]
        Pyx[l,3] = delta    if l-1   in ran and c[1][0] - dx > neg_lim  else Pyx[l,3]
        Pyx[l,4] = epsi
        Pyx[l,5] = psi      if l+1   in ran and c[1][0] + dx < pos_lim  else Pyx[l,5]
        Pyx[l,6] = phi      if l+n-1 in ran and c[1][0] - dx > neg_lim  else Pyx[l,6]
        Pyx[l,7] = chi      if l+n   in ran                             else Pyx[l,7]
        Pyx[l,8] = zeta     if l+n+1 in ran and c[1][0] + dx < pos_lim  else Pyx[l,8]
    h = 0
    for el in np.nditer(Pyx):
        if el != 0:
            h += 1
    conteggio_elementi += h
    print('number of elements in Pxy is {}'.format(h))  
    print('Pyx ok')
    #%% Make Pyy 
    for l,c in enumerate(lista):
        xy = c[1]
        E_o     = cost*tensor[xy][eyy]+ 1/tensor[xy][ezz]*(per_dxy[xy][exy] \
                    + per_dy2[xy][eyy])-ß**2
        alpha   = - (tensor[xy][exy]/tensor[xy][ezz])/(4*dx*dy) 
        beta    = tensor[xy][eyy]/tensor[xy][ezz]/dy**2 \
                    + 1/tensor[xy][ezz]*(per_dx[xy][exy]+2*per_dy[xy][eyy])*1/dy
        gamma   = (tensor[xy][exy]/tensor[xy][ezz])/(4*dx*dy)
        delta   = 1/dx**2
        epsi    = -2*tensor[xy][eyy]/tensor[xy][ezz]/dx**2 -2/dx**2\
                    - 1/tensor[xy][ezz] * (2 * per_dy[xy][eyy] + per_dx[xy][exy])*1/dy \
                    - 1/tensor[xy][ezz] * per_dy[xy][exy]/dx + E_o
        psi     = 1/tensor[xy][ezz] * per_dy[xy][exy]/dx +1/dx**2 
        phi     = (tensor[xy][exy]/tensor[xy][ezz])/(4*dx*dy)
        chi     = tensor[xy][eyy]/tensor[xy][ezz]/dy**2
        zeta    = - (tensor[xy][exy]/tensor[xy][ezz])/(4*dx*dy)
        
        Pyy[l,0] = alpha    if l-n-1 in ran and c[1][0] - dx > neg_lim  else Pyy[l,0]
        Pyy[l,1] = beta     if l-n   in ran                             else Pyy[l,1]
        Pyy[l,2] = gamma    if l-n+1 in ran and c[1][0] + dx < pos_lim  else Pyy[l,2]
        Pyy[l,3] = delta    if l-1   in ran and c[1][0] - dx > neg_lim  else Pyy[l,3]
        Pyy[l,4] = epsi
        Pyy[l,5] = psi      if l+1   in ran and c[1][0] + dx < pos_lim  else Pyy[l,5]
        Pyy[l,6] = phi      if l+n-1 in ran and c[1][0] - dx > neg_lim  else Pyy[l,6]
        Pyy[l,7] = chi      if l+n   in ran                             else Pyy[l,7]
        Pyy[l,8] = zeta     if l+n+1 in ran and c[1][0] + dx < pos_lim  else Pyy[l,8]
    
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
    
    PXXPXY = hstack([PXX,PXY])
    PYXPYY = hstack([PYX,PYY])
    
    print('h-stak ok')
    
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

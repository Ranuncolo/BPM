# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:31:05 2019

@author: 2353588g
"""

import numpy as np
from scipy.sparse import dia_matrix
import time

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


""" Required values for now """

#s = size of the testing grid:
s = 100
w = np.linspace(-s,s,2*s+1)
xs,ys = np.meshgrid(w,w)
grid = (xs,np.flipud(ys)) 
# why do we need this flipup? It is just because I need to start from the 
# top-left corner or the grid to solve it or I am gonna have a weird matrix. 



# permittivity tensor, since it's not importat its values are None. I just
# care about the keys, which are the grid nodes coordinates
tensor = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}


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
    #print(l,c,Pxx[l])


    

C = Pxx

"""
After having tried several function, and having found that diag an the other
functions works with rows instead of columns, I have to move my vectors in 
order to use all the values. So I use the function roll which moves a number
of element from the bottom to the top. 
Look at this picture to better understand how diag works:
    
    https://scipy-lectures.org/advanced/scipy_sparse/dia_matrix.html
    
    offset: row

     2:  9
     1:  --10------
     0:  1  . 11  .
    -1:  5  2  . 12
    -2:  .  6  3  .
    -3:  .  .  7  4
         ---------8
    
    This is everything but elegant, and I hate it. 
"""    

t1 = time.time()
a1 = np.roll(C[:,0],-2*s-2)
a2 = np.roll(C[:,1],-2*s-1)
a3 = np.roll(C[:,2],-2*s)
a4 = np.roll(C[:,3],-1)
a5 = C[:,4]
a6 = np.roll(C[:,5],1)
a7 = np.roll(C[:,6],2*s)
a8 = np.roll(C[:,7],2*s+1)
a9 = np.roll(C[:,8],2*s+2)

# Store everuthing in data: these are my diagonals
data = [a1,a2,a3,a4,a5,a6,a7,a8,a9]

#store offset vector (in rows)
diag = np.array([-2*s-2,-2*s-1,-2*s,-1,0,1,2*s,2*s+1,2*s+2])

#call the sparse matrix. Use toarray() to actually see it rn
D = dia_matrix((data,diag),(s*2+1,2*s+1)).toarray()
print(D)
print(time.time()-t1)

## WAY TwO, instead of roll I am going to delete some elements so that
# the numeber of elements in each diagonal is the right one, always.
t2 = time.time()

a1 = C[2*s+2:,0]
a2 = C[2*s+1:,1]
a3 = C[2*s:,2]
a4 = C[1:,3]
a5 = C[:,4]
a6 = C[:-1,5]
a7 = C[:-2*s,6]
a8 = C[:-2*s-1,7]
a9 = C[:-2*s-2,8]

from scipy.sparse import diags
diagonals = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
O = diags(diagonals,diag).toarray()
print(O)
print(time.time() - t2)

""" Equivalent methods, I wonder which one is quicker >< """

""" 
apparently for bigger s the first one is quicker than the second one O-o 
which is somehow unexpected (?)

but ok! I'll trust it. 
"""
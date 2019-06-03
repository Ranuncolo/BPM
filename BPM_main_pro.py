# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:13:02 2019

@author: 2353588G

This should be the complete and main script to call the full BPM
"""


import EMpy
import pylab
import numpy as np
#import whatever as what


# wavelength [m]
λ = 1.55e-6

#size of computation window and waveguide

x, y, z = 1e-6, 1e-6, 10e-6
xw, yw = 0.4e-6, .2e-6

print('Wavelength: ✓\nComputation window: ✓\nWavelength dimensions: ✓')

neig = 2 
tol = 1e-8
boundary = '0000'

#definition of waveguide permitt
e_w = (3.4757**2, 1, 0, -1, 2.4757**2, 0, 0, 1, 3.4757**2) 
exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz = e_w

# this function actually defines where the waveguide is, giving its permittivity
def epsfunc(x_, y_):
    '''Similar to ex_modesolver.py, but using anisotropic eps.'''
    eps = np.zeros((len(x_), len(y_), 5))
    for ix, xx in enumerate(x_):
        for iy, yy in enumerate(y_):
            # The waveguide is actuaply defined here
            if abs(xx - x/2) <= xw/2 and abs(yy - y/2) <= yw/2:
                eps[ix, iy, :] = [exx, exy, eyx, eyy, ezz]
            else:
                a = 1.446**2
                # isotropic
                eps[ix, iy, :] = [a, 0, 0, a, a]
    return eps


"""
So, i couldn't really get how it worked but actyally they are just defining the
grid window  at the beginning in x,y but the waveguide is defined by its e_tensor
inside the epsfunc. So yeah, the waveguide is actually just 240x110 nm, which 
is way more reasonable lol
"""
# Here I am trying to plot it and it is actually true, as I said. For now,
# instead of considering the dx step, it is actually chosen the numeber of points
# I want. This is because arange is not good for NON integer values. However,
# in this way I also return the step dx,dy
xl, dx = np.linspace(0,x,101, retstep = True)
yl, dy = np.linspace(0,y,101, retstep = True)



#% Plot the initial configuration,
print('Initial permittivity tensor: ✓')
print('Plot 1 - Input waveguide and permittivity tensor')
o = epsfunc(xl,yl)
#fig = pylab.figure()
pylab.contourf(o[:,:,0], 50)

#% Call EMpy solver for this waveguide and get Ex_
print('Calling EMpy modesolver:')
solver = EMpy.modesolvers.FD.VFDModeSolver(λ,xl,yl,epsfunc,boundary).solve(neig,tol)
print('EMpy modesolver: ✓\nStoring source field in Ex_o')


Ex_o = abs(solver.modes[0].Ex) # *** why abs?
print('Im getting the abs value, not sure why though --> **WHY ABS**')
pylab.contourf(Ex_o, 50) #Here 50 is for number of lines, the more the circa better


#%% Converting Ex_o into a dictionary because it would be easier to manage (? I think)
#try a meshgrid

x_r,y_r = Ex_o.shape
x_r = np.linspace(0,x_r,x_r+1)
y_r = np.linspace(0,y_r,y_r+1)

#xx,yy = np.meshgrid(x_r,y_r)
xx,yy = np.meshgrid(xl-max(xl)/2,yl-max(yl)/2)
xx = xx*10**9; yy = yy*10**9
#pylab.contourf(xx,yy,Ex_o,50) plot it to see if it's the same, it should lol
yy = np.flipud(yy)

xx = np.around(xx,1); yy = np.around(yy,1)
# And now convert in dictionary:
key = zip(xx.flat,yy.flat)
key = tuple(key)
Ex_o = dict(zip(key,Ex_o.flat))

## To come back, do as follows:
#k = tuple(Ex_o.keys())
#n,m = zip(*k)
#n = np.asarray(n).reshape(99,99)
#m = np.asarray(m).reshape(99,99)
#v = tuple(Ex_o.values())
#v = np.asarray(v).reshape(99,99)
#pylab.contourf(n,m,v,50)
print('Source field stored in Ex_o\nPropagation is starting now.')
print('--------------------------------------------------------------------')
#%%

print('Make our permettivity tensor')

"""
Permittivity tensor should address this problem, with a wider grid and a if
statement

OG COURSE THERE IS THE PROBLEM THAT I AM NO LONGER
CENTRATO PERCHE IL TIPO HA PENSATO DI INIZIARE DA ZERO ZERO ANZICHE PARTIRE
CON LA WAVEGUIDE AL CENTRO >.> VEDERE SE CAMBIA? COMUNQUE IPOTIZZANDO DI STARE
CON I LORO, MI TRSLO IO.

"""
dz = 0 #first step
import permit_tensor
xw = xw*10**9
yw = yw*10**9
wave_space = ((-yw/2,yw/2),(-xw/2,xw/2))
#xx = xx*10**9; yy = yy*10**9
grid = (xx,yy)
#grid = np.around(grid,1) # just one decimal is more than enough, I am in nanometers
#%% 
#initialisation of the dictionary - permittivity
per_tensor  = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}

#%%
e_w = np.asarray(e_w)
per_tensor = permit_tensor.permit_tensor(per_tensor,wave_space,e_w)

# try to plot it

k = 0
a = np.zeros(len(per_tensor))
for key in per_tensor:
    a[k] = per_tensor[key][0]
    k += 1
"""
with np.printoptions(threshold=np.inf):
    print(a)
"""
pylab.contourf(xx,yy,a.reshape(xx.shape))



#%% make P
omega = 2
mu_o = 1
epsilon_o = 2
cost = omega ** 2 * mu_o * epsilon_o
ß = 1

#%% Call the function
################################
dx = dx*10**9
dy = dy*10**9

import matrix_P
P = matrix_P.P(per_tensor,grid,dx,dy)
print('P - sparse matrix: ✓')
print('{} sparse matrix with {} stored elements.'.format(P.shape, P.nnz))

#%% Search for eigenvalues

from scipy.sparse.linalg import eigen
vals, vecs = eigen.eigs(P,2,which='LR',return_eigenvectors=True)
print('Eigenvalues/vectors: ✓')
print('Computed {} eigenvalues each with {} eigenvectors.'.format(len(vals),len(vecs)))

""" 
So, apparently I am still losing values (actually most of my diagonals)
because, I think, most of the values are still zero even if I am not quite 
that sure. I should try anyway with real numbers to see what I get. 
Unfortunately I am not 100% sure what is going to get out.
"""


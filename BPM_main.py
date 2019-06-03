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

#size of computation window, um i guess
x, y, z = 2.48, 2.22, 10

neig = 2 
tol = 1e-8
boundary = '0000'




def epsfunc(x_, y_):
    '''Similar to ex_modesolver.py, but using anisotropic eps.'''
    eps = np.zeros((len(x_), len(y_), 5))
    for ix, xx in enumerate(x_):
        for iy, yy in enumerate(y_):
            # The waveguide is actuaply defined here
            if abs(xx - 0.5e-6) <= .24e-6 and abs(yy - 0.5e-6) <= .11e-6:
                a = 3.4757**2
                b = 1  # some xy value
                # eps_xx, xy, yx, yy, zz
                eps[ix, iy, :] = [a, b, b, a, a]
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
# Here I am trying to plot it and it is actually true, as I said
xl = np.linspace(0,1.00e-6,100)
yl = np.linspace(0,1.00e-6,100)
o = epsfunc(xl,yl)
#fig = pylab.figure()
pylab.contourf(o[:,:,0], 50)

#%% questa super retarded function vuole una funzione (epsfunc) come input >.>
solver = EMpy.modesolvers.FD.VFDModeSolver(λ,xl,yl,epsfunc,boundary).solve(neig,tol)

#%%%
fig = pylab.figure()
fig.add_subplot(1, 3, 1)

#%%

EX = solver.modes[0].Ex
EY = solver.modes[0].Ey
EZ = solver.modes[0].Ez

pylab.contourf(abs(EX), 50)
pylab.contourf(abs(EY), 50)
pylab.contourf(abs(EZ), 50)

#%%
#try a meshgrid
pylab.contourf(abs(EX), 50)
Ex = abs(EX)
x_r,y_r = Ex.shape
x_r = np.linspace(0,x_r,99)
y_r = np.linspace(0,y_r,99)

xx,yy = np.meshgrid(x_r,y_r)
pylab.contourf(xx,yy,Ex,50)

# And now convert in dictionary:

a = zip(xx.flat,yy.flat)
A = tuple(a)
E_x0 = dict(zip(A,Ex.flat))

#come back, 

k = tuple(E_x0.keys())
n,m = zip(*k)
n = np.asarray(n).reshape(99,99)
m = np.asarray(m).reshape(99,99)

v = tuple(E_x0.values())
v = np.asarray(v).reshape(99,99)
pylab.contourf(n,m,v,50)
#%%
"""
So from this I am only interested into E_x0, a dictionary where each (x,y) is
actually a node, and its value of E_field is there saved). Now I can propagate
with my own piece of code.
"""





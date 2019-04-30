# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:56:59 2019

@author: 2353588G
"""

import numpy as np
import matplotlib.pyplot as plt


"""
Making the function - Pade (1,1), Pade(2,2) and real root.
"""

def Pade1(P):
    Pade1 = (P/2)/(1+P/4)
    return Pade1

def Pade2(P):
    Pade2 = (P/2 + P**2/4)/(1+3*P/4+P**2/16)
    return Pade2

def Pade3(P):
    Pade3 = (P/2 + P**2/2 +3*P**3/32)/(1+5*P/4+3*P**2/8 + P**3/64)
    return Pade3

def no_P(P):
    no_P = np.sqrt(1+P)-1
    return no_P

"""
I am spanning from 0 to 10
"""

Po = np.r_[-1:10:1001j]

p0 = no_P(Po)
p1 = Pade1(Po)
p2 = Pade2(Po)
p3 = Pade3(Po)

plt.plot(Po,p0, label = 'real root')
plt.plot(Po,p1, label = 'Pade (1,1)')
plt.plot(Po,p2, label = 'Pade (2,2)')
plt.plot(Po,p3, label = 'Pade (3,3)')
plt.legend()
plt.show()
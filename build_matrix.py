# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:49:21 2019

@author: 2353588g
"""
import numpy as np
from scipy.sparse import dia_matrix

class build_matrix:
    
    def __init__(self,s,C):
        """ For now it is, but it's wrong because I would like to have the grid"""
        self.s = s
        self.C = C #matrix, nodex9
    def diagonal_position(self):
        C = self.C
        s  = self.s
        a1 = np.roll(C[:,0],-2*s-2)
        a2 = np.roll(C[:,1],-2*s-1)
        a3 = np.roll(C[:,2],-2*s)
        a4 = np.roll(C[:,3],-1)
        a5 = C[:,4]
        a6 = np.roll(C[:,5],1)
        a7 = np.roll(C[:,6],2*s)
        a8 = np.roll(C[:,7],2*s+1)
        a9 = np.roll(C[:,8],2*s+2)
        n_nodes = len(a1)
        # Store everuthing in data: these are my diagonals
        data = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
        
        #store offset vector (in rows)
        diag = np.array([-2*s-2,-2*s-1,-2*s,-1,0,1,2*s,2*s+1,2*s+2])

        #D = dia_matrix((data,diag),(s*2+1,2*s+1)).toarray()
        D = dia_matrix((data,diag),(n_nodes,n_nodes))
        return D
    
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:54:54 2019

@author: 2353588g
"""

"""
Function to grid out the waveguide

"""

#width = 50
#height = 120  
#length = 200 
#max_step = 10
def grid_xyz(width,height,length,max_step):
    
    """
    In input, I want to have:   
            - width (along x), 
            - heigth (along y),
            - length (along z).
            - max_step is the maximal size between two nodes
            - min_step (maybe?)
            
            
        - It computes Dx and Dy in order to have them equal. If the span is
            bigger than max_step, then it is divided by 10.
    
        - Dz will be the closest value of Dx/Dy which gives k*Dz = length
            *** this is just for now ***
    """
    
    
    """ D&C algorthm - recursive """
    def dxdy_pitch(width,height):
        rett = [width,height]    
        # remainder is what is left after the division
        remainder = max(rett) % min(rett)
        if remainder == 0:
            dx = rett[0]
            return dx
        else:
            # remainder is now the max edge of my rectangle:
            rett[rett.index(max(rett))] = remainder
            return dxdy_pitch(rett[0],rett[1])

    
    
    dx = dxdy_pitch(width,height)
    while dx >= max_step:
        dx = dx/10
        
    
    Dx = dx
    Dy = dx
    Dz = dx


    # now I get Dz. If it is a integer number of dx, I'll keep dx.
    times = 10  #tries, if it doesn't finish in 10 tries I will stop and
                    # force it to be 1.
    while length % Dz != 0:
        #print(times) #those checks are only for me 
        #print(Dz)
        if times == 0:
            Dz = 1
            break
        times = times - 1
        Dz = round(Dz - 0.1,1) # had to put round or it floated too much lol

    
    
    print(Dx,Dy,Dz)

    return Dx, Dy, Dz


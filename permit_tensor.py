          # -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:01:43 2019
    
@author: 2353588g



This code will assign at each node its permittivity tensor in a hash table
(dictionary). In order to test it, some values will be added like the size of
the waveguide, a random permit_tensor, the thickness of the coating and so on.

For now, two functions are defined here: the real function which assing the 
permittivity tensor to each node (permit_tensor(*arg)) and a second one, quite
similar to the previous one, which will plot the section. The code, in this
second function, its quite similar to the other one and I would ideally be able
to plot the hast table itself. It should be too difficult, and I will probably
change it later on. 
"""


"""
------------------------------------------------------------------------------
TESTING PART
------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt

"""
# Specify waveguide dimensions, coating and their tensors 
width = 100
height = 82

e_w = np.array([3, 1, 0, 1, 3, 0, 0, 0, 2])   
#coating thickness
t = 15
e_c = np.array([3.5, 0.5, 0, 0.5, 3, 0, 0, 0, 2.7])
    
# define their occupied 'space'    
wave_space = ((-width/2,width/2),(-height/2,height/2))
coat_space = ((-width/2,width/2),(height/2,height/2+t))


# make the grid for the nodes, two times as big as the waveguide for now
rr = max(width,height)   
rr = rr*2 
rr = np.linspace(-rr/2,rr/2,rr+1)
xx,yy = np.meshgrid(rr,rr)

# Not sure if it;s convenient but I pile them and then zip. Ideally I may give
# the grid of nodes as an input from somewhere else in the code
grid = (xx,yy)



# Preallocate tensor
# It is a tensor whose keys are combination of (x,y) and None as values.
tensor  = {(i,j):None for i,j in zip(grid[0].flat,grid[1].flat)}
"""
##############################################################################
def permit_tensor(permittivity_tensor,wave,e_w,coat=((0,0),(0,0)),e_c=np.ones(9),k=1):
    
    """ 
    This function will return the permittivity tensor at the k-step, which
    corresponds to the z = k * dz position. 
    The z-step size (dz) is somehow assumed to be global.
    Since I only need a single permittivity tensor at each step, and once 
    the field is computer I can get rid of it, I will (for now) provide it
    both as input and output to spare memory and preallocate it. 
        
    INPUT
    - tensor:   (dict), previous hash table with all the tensors for each node.
                Usually, once the step has finished it is not necessary to have
                it, so it will be overwrite;
    - wave:     (tuple), it is the extremes occupied by the waveguide in the
                following way: ((-x_w/2,x_w/2),(-y_w/2,y_w/2));
    - e_w:      (ndarray), permittivity tensor in the waveguide, it's 3x3;
    - coat:     (tuple), same idea of the wave;
    - e_c:      (ndarray), same as e_w;
    - k:        (int), index of the z position according to z = k * dz;
        
    OUTPUT
    - tensor:   (dict) tensor dictionary at the position k. Every key is a 
                (x,y) position with its tensor epsilon(x,y)
                
                
    ---------------------------------------------------------------------
    
    This means that given this permitt_tensor at each step, I can ask the hash
    table for the permittivity tensor at any poit giving a (x,y) positions! 
    
    for example, if I want to know e_xyz at the position (3,5),
    permitt_tensor[(3,5)] = [[1 2 3]
                            [4 5 6]
                            [7 8 9]]
    
    If you want to know e_zz at this position then it's just

    permitt_tensor[(3,5)][2,2]
    
    NB it's [2,2] because the position in a ndarray is:
        - first index column (from zero)
        - second index row (from zero)                 
        
    NB2 I may not need a 2D-array but just one with [exx,exy,eyx,eyy,ezz]         
    
    ---------------------------------------------------------------------
    """
    
    # let's start iterating along this hash table's keys 
    for key in permittivity_tensor:
        # if inside the waveguide, it's values is e_w --> key:e_w
        if wave[0][0] <= key[0] <= wave[0][1]:
            if wave[1][0] <= key[1] <= wave[1][1]:
                permittivity_tensor[key] = e_w
        #if inside the coating it will be key:e_c
            elif coat[1][0] <= key[1] <= coat[1][1]:
                permittivity_tensor[key] = e_c
        #otherwise it's the identity tensor
            else:
                permittivity_tensor[key] = np.ones(9)
        else:
            permittivity_tensor[key] = np.ones(9)
            
    return permittivity_tensor
            
    
#############################################################################   
def draw_profile(grid,wave,coat):
    """
    This functions retrieves a vector k (actually it's a ndarray) which is 
    whose values depends on the position (x,y) in the grid where k[i] is 
    respectively:
    0 - if in air;
    1 - if in the waveguide;
    2 - if inside the coating.
    
    INPUT
    grid:   (tuple), with the mesh grids in xx and yy;
    wave:   (tuple) is the extremes of the waveguide;
    coat:   (tuple) is the extremes of the coating;
    
    OUTPUT
    k:      (ndarray) with the values for each position
    
    -------------------------------------------------------------------------
    FYI:
    for i,j in zip(grid[0].flat,grid[1].flat):
    print(i,j)
    
    results in
     0 0
     1 0
     ...
     8 9
     9 9
     ------------------------------------------------------------------------
    """
    #preallocate the vector k, as wide as the number of elements in the grid
    length_k = grid[0].size 
    k = np.zeros(length_k) 
    
    
    
    cnt = 0 #potion of vector k
    
    # this make a list of (x,y) position using the iterators in zip.
    # something.flat make it an iterator, and are now combined
    for i,j in zip(grid[0].flat,grid[1].flat):
        # if inside the waveguide, marker is 1
        if wave[0][0] <= i <= wave[0][1]:
            if wave[1][0] <= j <= wave[1][1]:
                k[cnt] = 1
                cnt+=1
        #if inside the coating is 2
            elif coat[1][0] <= j <= coat[1][1]:
                k[cnt] = 2
                cnt+=1
        #otherwise is zero
            else:
                k[cnt] = 0
                cnt+=1
        else:
            k[cnt] = 0
            cnt+=1
        
    return k

##############################################################################    

"""
Let's try them:

tensor_k = permit_tensor(tensor,wave_space,e_w,coat_space,e_c,1)

#call the second function and then plot it
k = draw_profile(grid,wave_space,coat_space)
plt.pcolor(xx,yy,k.reshape(len(xx),len(xx)))




"""

"""
Problem of k in the first function
I need a way to check if, at this position, there is or there's not the coating.
k is the step, ok, but there should be something somewhere which tells me 
yes, consider the coating or nope, there's just air here. 
"""







"""
OLD DISCUSSION

Now I have two options to solve this problem for each step:
    - a switch condition, where I have my basic permitt_tensor (the one of 
        the waveguide (w,h)) as long as I am inside (+-w/2,+-h/2), else I have
        to compute and update the permitt_tensor at the interfaces 
    - a function permitt_tensor which is called at every step dz and here it
        builts up a hash table where I can easily get the tensor at each pos

After a quick thought, I think the second one is probably quicker because
otherwise I should check my position at each step, and it would be very quick
as long as I am inside the waveguide, but quite expensive at the borders. 

So for now I will try with the hash table, then if I'll be told that I am going
in the wrong direction I'll change
"""

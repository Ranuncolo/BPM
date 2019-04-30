# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:47:20 2019

@author: 2353588G
"""

def binary_search(list,item):
    low = 0
    high = len(list) - 1
    
    while low <= high:
        mid = round(0.5*(low+high))
        guess = list[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None

my_list = [1,22,34,111,547,554,1245,1289,1457,1865,12210]

print(binary_search(my_list, 34))
print(binary_search(my_list, 35))

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:05:03 2019

@author: 2353588g
"""

"""
Trying to write an algorithm to sort out an array

This algorithm makes a new array with numbers from the smallest to the highest

"""

# questa funzione va a cercare la posizione del valore piu' piccolo
def find_smallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index

# questa funzione crea un nuovo array ma questa volta sorted
    
def selection_sort(arr):
    new_array = []
    for i in range(len(arr)):
        smallest_i = find_smallest(arr)
        new_array.append(arr.pop(smallest_i))
    return new_array


arrow = [5, 4 , 7, 10, 11, 1, 25]

print(arrow)
print(selection_sort(arrow))   
        

""" 
There's also a RECURSIVE countdown :D
"""


def countdown(i):
    print(i)
    if i <= 0:
        return
    else:
        countdown(i-1)
        

countdown(15)

"""
CALL STACK
 - funzioni che chiamano se stesse in recursive mode 
"""

# fattoriale

def fact(x):
    if x == 1:
        return 1
    else:
        return x * fact(x-1)
    

y = fact(4)
print(y)



# dictionary - hash table, made with check if there's already
voted = {} # make dictionary
def check_vote(name):
    if voted.get(name):
        print("No I'm sorry")
    else:
        voted[name] = True
        print("You can")
        
        
    
"""
Graph, queue and dequeue from pag 107
"""
graph = {}
graph["you"] = ["alice","mara","caram"]
from collections import deque
search_queue = deque()                  # creates new queue
search_queue += graph["you"]            # add all the neighbours to the search

# this is the mango owner

def person_is_seller(name):
    return name[-1] == 'm'              # check if it is ending with m 

while search_queue:                     # while queue is not empty
    person = search_queue.popleft()     # grab the first person off the queue
    if person_is_seller(person):        # check if mango seller
        print("eccola!")
#        return True
    else:
        search_queue += graph[person]   # add their list to the queue
#return False

""" 
Better code, from the book, with checks
"""

def search(name):
    search_queue = deque()
    search_queue += graph[name]
    searched = []
    while search_queue:
        person = search_queue.popleft()
        if not person in searched:
            if person_is_seller(person):
                print("is a mango seller!")
                return True
            else:
                search_queue += graph[person]
                searched.append(person)
    return False

search("you")



    
    
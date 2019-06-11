# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:20:19 2019

@author: 2353588g
"""

import difflib

file1 = 'BPM_main_pro.py'
file2 = 'BPM_main_pro2.py'

ff1 = open(file1).readlines()
ff2 = open(file2).readlines()

for line in difflib.unified_diff(ff1,ff2):
    print(line)
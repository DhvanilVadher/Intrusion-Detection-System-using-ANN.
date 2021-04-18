#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:41:39 2021

@author: dhvanil
"""

import csv
input = open('CombinedWEDFRI.csv', 'r')
output = open('CombinedWEDFRIBenign.csv', 'w')
writer = csv.writer(output)
cnt = 0
for row in csv.reader(input):
    if cnt == 0:
        writer.writerow(row)
        cnt = cnt + 1
    if row[11]=="BENIGN":
        writer.writerow(row)
        
input.close()
output.close()
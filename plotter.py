#!/usr/bin/env python3
# -*- coding : utf8 -*-

from matplotlib import pyplot as plt
import argparse
import csv
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
args = parser.parse_args()                                                        

#colors = ['b','r','g','y','k']
time = []
scatter = []


with open(args.inputFile, 'r') as inputcsv:
    reader = csv.reader(inputcsv)
    bslabel = next(reader)[1]
    for row in reader:
        if bslabel != row[1] and time:
            plt.scatter(time, scatter, label=bslabel)#, color=colors[int(bslabel)])
            bslabel = row[1]
            time = []
            scatter = []

        time.append(int(row[0]))
        scatter.append(float(row[2]))
    
    plt.scatter(time, scatter, label=bslabel)#, color=colors[int(bslabel)])


plt.legend()
plt.show()

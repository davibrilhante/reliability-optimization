#! /usr/bin/env python3
# -*- coding : utf-8 -*-

import json
import numpy as np
from matplotlib import pyplot as plt

plot = []
accuracy = []
not_accurate = []

for i in range(101,201):
    filename = 'test-results/test-optimisation-'+str(i)
    try:
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile)

    except FileNotFoundError:
        continue

    opt_comb = [i for i in data['opt_comb'] if i[1] > data['begin']]
    if not opt_comb:
        opt_comb = [data['opt_comb'][-1]]
    else:
        opt_comb = [data['opt_comb'][data['opt_comb'].index(opt_comb[0]) - 1]] + opt_comb

    if type(data['test_comb'][0]) == type([]):
        bruteforce_comb = [[data['test_comb'][0][n+1]['index'], t] for n, t in enumerate(data['test_comb'][1])  ]
        bruteforce_comb = [[data['test_comb'][0][0]['index'], data['begin'] - data['timeslots']]] + bruteforce_comb
    else:
        bruteforce_comb = [[data['test_comb'][0]['index'], data['begin'] - data['timeslots']]]

    match = 0
    if len(opt_comb) > 1:
        for comb in opt_comb[1:]:
            if comb in bruteforce_comb[1:]:
                match += 1
        accuracy.append(match/(max(len(opt_comb), len(bruteforce_comb))-1))
    else:
        if opt_comb[0][0] == bruteforce_comb[0][0]:
            accuracy.append(1.0)

    if accuracy[-1] != 1.0:
        not_accurate.append([len(accuracy), accuracy[-1], data['instance']])


    
    plot.append(data['diff'])

    if data['diff'] > 1:
        if data['instance'].split('/')[-2:] == ['0.004','11']:
            plt.text(len(plot)+6, data['diff']+500, data['instance'].split('/')[-2:], 
                        ha='left', va='bottom', rotation=45.0)
        else:
            plt.text(len(plot)-3, data['diff']+500, data['instance'].split('/')[-2:], 
                        ha='left', va='bottom', rotation=45.0)
        plt.scatter(len(plot)-1, data['diff'], marker='o', color='r')

print(plot)

mean = np.mean(plot)
median = np.median(plot)

print(len(plot))
print(len(plot)/100)
print(mean)
print(median)

plt.grid(True, which='both')
plt.ylabel('Test - Optimisation Error')
plt.ylim([10e-10,10e13])
plt.xlabel('Instance #N')
plt.xlim([0,110])

plt.hlines(mean, 0, 100, color='g')
plt.text(100, mean+700, 'mean', ha='right', fontweight='bold')

plt.hlines(median, 0, 100, color='purple')
plt.text(1, median+10e-4, 'median',fontweight='bold')

plt.semilogy(plot)
plt.show()

plt.plot(accuracy)
for i in not_accurate:
    plt.text(i[0], i[1], i[2].split('/')[-2:])
    plt.scatter(i[0]-1, i[1], marker='o', color='r')
plt.ylabel('Handover Accuracy')
plt.xlabel('Instance #N')
plt.grid()
plt.show()

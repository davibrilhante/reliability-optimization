#! /usr/bin/env python3
# -*- coding : utf8 -*-

'''
Este código plota o máximo de oportunidades de handover por densidade de 
bloqueio e velocidade
'''

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import json

from plot_utils import load_result
from plot_utils import load_instance
from optimization import snr_processing

def threadHoCounter(instance,tsim,ttt,offset,hysteresis):
    opportunities = 0
    ho_flag = False
    SNR, RSRP = snr_processing(instance['scenario'], 
                                instance['baseStation'],
                                instance['userEquipment'], 
                                instance['channel'], 
                                instance['blockage'])

    for source,_ in enumerate(instance['baseStation']):
        for t in range(ttt,tsim):
            for target,_ in enumerate(instance['baseStation']):
                if source==target:
                    continue

                temp = 0
                init = max(t - ttt,0)
                for k in range(init,t):
                    if RSRP[target][0][k] > RSRP[source][0][k] + offset + hysteresis:
                        temp += 1
                    else:
                        break

                if temp >= ttt:
                    opportunities += 1
                    break
    #result.append(opportunities)

    return opportunities


def main():
    ho_opportunities = {}

    offset = 3
    hysteresis = 0

    vel_params = [(22,160,203647),
            (43,80,101823),
            (64,40,67882)]

    Lambda = [round(i*0.002 + 0.001,3) for i in range(5)]
    delay = [i*2 + 1 for i in range(5)]
    seeds = range(60)
    counter = 0

    for vel, ttt, tsim in vel_params:
        for blockdensity in Lambda:
            data, _ = load_result(
                    'instances/no-interference/opt/{t}/{v}/{l}/750e6/1/'.format(t=ttt,v=vel,l=blockdensity)).load()

            instances = load_instance(
                    'instances/full-scenario/{v}/{l}/'.format(v=vel, l=blockdensity)).load()

            ho_opportunities[vel] = {}
            ho_opportunities[vel][blockdensity] = []

            ninstances = len(data.items())
            #for m, result in data.items():
            nthreads = 5
            rounds = ninstances//nthreads

            for i in range(rounds):
                print(i)
                with Pool(processes=nthreads) as pool:
                    ho_opportunities[vel][blockdensity] += pool.starmap(threadHoCounter,
                    [(instances[str(i*nthreads+j)],tsim,ttt,offset,hysteresis) for j in range(nthreads)])
                
            plt.bar(counter, np.mean(ho_opportunities[vel][blockdensity]))
            counter += 1

    with open('ho-opp','w') as outfile:
        json.dump(ho_opportunities,outfile)

    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

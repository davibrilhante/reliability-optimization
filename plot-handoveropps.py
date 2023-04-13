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
    ntargets = []
    ho_flag = False
    SNR, RSRP = snr_processing(instance['scenario'], 
                                instance['baseStation'],
                                instance['userEquipment'], 
                                instance['channel'], 
                                instance['blockage'])

    for source,_ in enumerate(instance['baseStation']):
        for t in range(ttt,tsim):
            targets = 0
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
                    targets += 1

            if targets>0:
                ntargets.append(targets)

    #result.append(opportunities)

    return opportunities


def threadRsrp(instance):
    samples = []
    SNR, RSRP = snr_processing(instance['scenario'], 
                                instance['baseStation'],
                                instance['userEquipment'], 
                                instance['channel'], 
                                instance['blockage'])
    for bs in RSRP:
        for sample in bs[0]:
            samples.append(sample)

    return samples


def main():
    ho_opportunities = {}

    offset = 3
    hysteresis = 0

    vel_params = [(22,160,203647)]#,
            #(43,80,101823),
            #(64,40,67882)]

    Lambda = [round(i*0.002 + 0.001,3) for i in range(5)]
    delay = [i*2 + 1 for i in range(5)]
    seeds = range(60)
    counter = 0

    for vel, ttt, tsim in vel_params:
        for blockdensity in Lambda:
            print(vel, blockdensity)
            data, _ = load_result(
                    'instances/no-interference/opt/{t}/{v}/{l}/750e6/1/'.format(t=ttt,v=vel,l=blockdensity)).load()

            instances = load_instance(
                    'instances/full-scenario/{v}/{l}/'.format(v=vel, l=blockdensity)).load()

            ho_opportunities[vel] = {}
            ho_opportunities[vel][blockdensity] = []

            ninstances = len(data.items())
            nthreads = 6

            #with Pool(processes=nthreads) as pool:
            #    ho_opportunities[vel][blockdensity] += pool.starmap(threadHoCounter,
            #    [(instances[str(i)],tsim,ttt,offset,hysteresis) for i in range(ninstances)])
                

            with Pool(processes=nthreads) as pool:
                ho_opportunities[vel][blockdensity] += pool.map(threadRsrp,
                        [instances[str(i)] for i in range(ninstances)])

            #print(ho_opportunities[vel][blockdensity])
            #plt.bar(counter, np.mean(ho_opportunities[vel][blockdensity]))
            #counter += 1

    with open('ho-rsrp','w') as outfile:
        json.dump(ho_opportunities,outfile)

    plt.boxplot(ho_opportunities[22].values())
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

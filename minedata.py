#! /usr/bin/env python3
# -*- coding : utf8 -*-
import numpy as np
from json import load, dump
import pandas as pd
from decompressor import decompressor
from optimization import calc_snr, calc_recv

def catch_handover(time, bs, instance, ttt, offset=3, hysteresis =0):
    init = max(time-tau, 0)
    sbs_snr = []
    candidates = []
    for t in range(init,time):
        sbs_snr.append(calc_snr(instance['baseStation'][bs],
                            instance['userEquipment'][0],
                            instance['channel'],
                            instance['blockage'][bs][0][t],
                            t))

    for tbs in instance['baseStation']:
        flag = 0
        if tbs['index'] == bs:
            continue

        index = tbs['index']
        for t in range(init, time):
            tbs_snr = calc_snr(instance['baseStation'][index],
                                instance['userEquipment'][0],
                                instance['channel'],
                                instance['blockage'][index][0][t],
                                t)

            if tbs_snr >= sbs_snr[t-init] + offset + hysteresis:
                flag+=1
            else:
                break

        if flag==ttt:
            candidates.append(index)

    return candidates

def list_stats(init,end,bs,instance):
    rsrp = []
    block = []
    rates = []
    nlos_slots = 0
    interval = (end - init)//5
    for t in range(init,end):
        rsrp.append(calc_recv(instance['baseStation'][bs],
            instance['userEquipment'][0],
            instance['channel'],
            instance['blockage'][bs][0][t]))

        if instance['blockage'][bs][0][t]:
            if nlos_slots != 0:
                block.append(nlos_slots)
            nlos_slots = 0
        else:
            nlos_slots += 1

        past_slots = t - init
        current = past_slots//interval
        if past_slots%interval == 0:
            previous = init + max(current - 1, 0)*interval
            current = init + current*interval
            rates.append(sum(instance['blockage'][bs][0][previous:current])/interval)

    if nlos_slots!=0:
        block.append(nlos_slots)

    return rsrp, block, rates


def calc_dist(t,bs,instance):
    current_x = instance['userEquipment'][0]['position']['x'] + instance['userEquipment'][0]['speed']['x']*3.6*(t/1e3)
    current_y = instance['userEquipment'][0]['position']['y'] + instance['userEquipment'][0]['speed']['y']*3.6*(t/1e3)
    dist = np.hypot(instance['baseStation'][bs]['position']['x'] - current_x,
            instance['baseStation'][bs]['position']['y'] - current_y)

    bs_angle = np.arctan2(instance['baseStation'][bs]['position']['y'] - current_y,
                        instance['baseStation'][bs]['position']['x'] - current_x)

    ue_angle = np.arctan2(current_y,current_x)
    angle = bs_angle - ue_angle

    '''
    if angle > np.pi/2 or angle < -np.pi/2:
        angle = -1
    else:
        angle = 1
    '''

    return dist, angle

if __name__ == '__main__':
    vel_params = [(22,160,203647),
                (43,80,101823),
                (64,40,67882)]

    densities = [round(i*0.002 + 0.001,3) for i in range(5)]

    data = {}

    root = 'instances/'
    base_name = 'optimization-'
    execs = 60
    interval = 1000
    for vel, tau, sim in vel_params:
        for den in densities:
            for n in range(execs):
                # Open the result file
                filename = root+'no-interference/opt/{t}/{v}/{l}/750e6/1/'.format(v=vel, t=tau,l=den)+base_name+'{v}-{t}-{l}-750e6-1-{n}'.format(v=vel, 
                        t=tau,l=den,n=n)
                with open(filename,'r') as jsonfile:
                    result = load(jsonfile)

                # Open the instance file
                filename = root+'full-scenario/{v}/{l}/{n}'.format(v=vel,l=den,n=n)
                with open(filename,'r') as jsonfile:
                    instance = load(jsonfile)
                    decompressor(instance)

                # Iterate over optimization associations
                ho_opportunity = 0
                for bs, init, end, _ in result['association']:
                    try:
                        data[(vel,den,n,ho_opportunity-1,bs)]['chosen']=1
                    except KeyError:
                        pass

                    for t in range(init,end+1):
                        # Identify the handover opportunity
                        candidates = catch_handover(t, bs, instance, tau)

                        if candidates:
                            #iterate over candidates
                            for target in [bs]+candidates:
                                # calculate the metrics
                                tmp = {}
                                tmp['chosen'] = 0
                                if target == bs and t!=end:
                                    tmp['chosen'] = 1

                                ti = max(0,t-interval)
                                tf = t
                                rsrp, block, rate = list_stats(ti, tf, target, instance)
                                tmp['mu_rsrp'] = np.mean(rsrp)
                                tmp['dev_rsrp'] = np.std(rsrp)
                                if block:
                                    tmp['mu_block'] = np.mean(block)
                                    tmp['dev_block'] = np.std(block)
                                    tmp['sum_block'] = sum(block)
                                    tmp['n_blocks'] = len(block)
                                    tmp['percent_block'] = sum(block)/interval
                                    tmp['max_block'] = max(block)
                                else:
                                    tmp['mu_block'] = 0
                                    tmp['dev_block'] = 0
                                    tmp['sum_block'] = 0
                                    tmp['n_blocks'] = 0
                                    tmp['percent_block'] = 0
                                    tmp['max_block'] = 0

                                tmp['rate1'] = rate[0] 
                                tmp['rate2'] = rate[1]
                                tmp['rate3'] = rate[2]
                                tmp['rate4'] = rate[3]
                                tmp['rate5'] = rate[4]
                                tmp['distance'], tmp['direction'] = calc_dist(t,target,instance)
                                tmp['timestamp'] = t
                                data[(vel,den,n,ho_opportunity,target)] = tmp

                            ho_opportunity+=1

    with open('mined_data.json','w') as jsonfile:
        jsonfile = dump(data)

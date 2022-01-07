#! /bin/env python3
# -*- coding : utf8 -*-

from itertools import combinations_with_replacement
from itertools import permutations
import argparse
import json
import numpy as np
from decompressor import decompressor

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
parser.add_argument('-l','--length', type=int, default=2, 
                        help='base station combination sequence length')
parser.add_argument('-t','--timeslots', type=int, default=1000, 
                        help='how many timeslots will by searched')
parser.add_argument('-b','--begin', type=int, default=0, 
                        help='beginning timeslot to be searched')
parser.add_argument('--ttt', type=int, default=640, 
                        help='System time to trigger')
parser.add_argument('--initialize', type=int, nargs=3)

args = parser.parse_args()


def todb(x : float) -> float:
    return 10*np.log10(x)

def tolin(x : float) -> float:
    return 10**(x/10)

def load_inputFile(inputFile, beginning = 0):
    network = []
    nodes = []
    print('Loading Instance...')

    with open(inputFile) as json_file:
        try:
            data = json.load(json_file)
        except:
            sys.exit()

        decompressor(data)

        scenario = data['scenario']
        channel = data['channel']
        LOS = data['blockage']
        #gamma = data['gamma']
        for p in data['baseStation']:
            network.append(p)
        for p in data['userEquipment']:
            nodes.append(p)
     # Resource blocks attribution
    R = []
    for bs in network:
        bw_per_rb = 12*bs['subcarrierSpacing'] #12 subcarriers per resouce block times 120kHz subcarrier spacing
        R.append(bw_per_rb*bs['resourceBlocks'])

    return scenario, channel, LOS, network, nodes, R

def snr_processing(scenario, network, nodes, channel, LOS, beginning=0):
    SNR = []

    print('Preprocessing...')

    # SNR evaluation
    print('SNR Evaluation...')
    for m, bs in enumerate(network):
        SNR.append([])
        for n,ue in enumerate(nodes):
            SNR[m].append([])
            #Adding the time dependency
            for t in range(scenario['simTime']):
                if LOS[m][n][beginning+t] == 1:
                    los = True
                else:
                    los = False
                SNR[m][n].append(calc_snr(bs,ue,channel,los,t))

    return SNR


def calc_recv(base : dict, user : dict, channel : dict, los : bool, t=0) -> float:
    # Evaluating the new position according with the vehicle speed (Change for vectorial speed)
    new_position_x = user['position']['x'] + (user['speed']['x']/3.6)*(t*1e-3)
    new_position_y = user['position']['y'] + (user['speed']['y']/3.6)*(t*1e-3)
    distance = np.hypot(base['position']['x'] - new_position_x, base['position']['y'] - new_position_y)
    wavelength = 3e8/base['frequency']
 
    bs_antenna_gain = 15
    ue_antenna_gain = 15
    exponent = channel['lossExponent']
 
    #pl_0 = 20*np.log10(4*np.pi/wavelength)
 
    if los:
        path_loss = 61.4 + 10*2*np.log10(distance) #+ np.random.normal(0,5.8)
    else:
        path_loss = 72 + 10*2.92*np.log10(distance) #+ np.random.normal(0,8.7)
 
    #path_loss = pl_0 + 10*exponent*np.log10(distance) #- np.random.normal(0,8.96)
    return base['txPower'] + bs_antenna_gain + ue_antenna_gain - path_loss
 
 
def calc_snr(base : dict, user : dict, channel : dict, los : bool, t=0) -> float:
    noise_power = channel['noisePower']
 
    return 10**((calc_recv(base, user, channel, los, t) - noise_power)/10)


def gen_combninations(network : list, length : int):
    #return combinations_with_replacement(network, length)
    return permutations(network, length)

def check_handover(s_bs : dict, t_bs : dict, SNR : list, start : int, tau : int, offset : int, hysteresys : int) -> bool:
    if s_bs != t_bs:
        handover = 0
        if start > tau:
            for k in range(start - tau, start):
                serving = todb(SNR[s_bs['index']][0][k])
                target = todb(SNR[t_bs['index']][0][k])

                balance = target - (serving + offset + 2*hysteresys)

                if balance >= 0:
                    handover += 1

            if handover >= tau:
                return True

            else:
                return False

        else:
            return False
    else:
        return False

def calc_objvalue(combination : list, ho_event : list, SNR : list, nodes : list, n : int, start : int, end : int) -> float:
    objval = 0

    if end not in ho_event:
        ho_event.append(end)

    for bs, interval in zip(combination, ho_event):
        #print(bs, start, interval)
        for t in range(start, interval):
            snr = SNR[bs['index']][0][t]
            if snr >= tolin(15): #nodes[n]['threshold']):
                objval += snr
                if t in nodes[n]['packets']:
                    objval += 1

        start = interval + 1

    return objval

def eval_comb(combination : list, ho_events : list, SNR : list, nodes : list, start : int, end : int, tau : int, recursion : bool):

    serv_bs = combination[0]
    final_seq = []
    obj_value = 0

    for targ_bs in combination[1:]:
        for t in range(start, end):
            events = []
            bs_seq = [serv_bs]
            if t - start >= tau - args.initialize[1]:
                if check_handover(serv_bs, targ_bs, SNR, t, tau, 3, 0):
                    if (end - start > 1) and len(combination[2:]) > 1:
                        t_bs, event = eval_comb(combination[1:], events, SNR, nodes, t, end, tau, True)
                        bs_seq += t_bs
                        events += event

                        if recursion:
                            return bs_seq, events

                    else: 
                        if recursion:
                            return [targ_bs], [t]
                        else:
                            bs_seq.append(targ_bs)
                            events.append(t)

                    current_objval = calc_objvalue(bs_seq, events, SNR, nodes, 0, start, end)
                    
                    if obj_value < current_objval :
                        obj_value = current_objval
                        final_seq = bs_seq
                        ho_events = events

                else:
                    continue

        if t >= end - 1:
            break

    return combination, ho_events, obj_value


if __name__=="__main__":
    scenario, channel, LOS, network, nodes, R = load_inputFile(args.inputFile)
 
    m_bs = len(network)
    n_ue = len(nodes)
 
    combinations = gen_combninations(network, args.length)
    SNR = snr_processing(scenario, network, nodes, channel, LOS)

    obj_value = list() 
    pre_value = 0
    assoc_time = 0
    for n, comb in enumerate(combinations):
        '''
        serv_bs = comb[0]
        ho_events = list()
        start = args.begin
        end = start+args.timeslots
        for targ_bs in comb[1:]:
            for t in range(start, end):
                if t > args.ttt:
                    if check_handover(serv_bs, targ_bs, SNR, t, args.ttt, 3, 0):
                        ho_events.append(t)                        
                        start = t+1
                        serv_bs = targ_bs
                        break
                    else:
                        continue
            if t >= end - 1:
                break
        ho_events.append(end)
        if len(ho_events) < args.length:
            comb = comb [:len(ho_events)]
        '''
        if args.initialize:
            last_bs = list([network[args.initialize[0]]])
            comb = last_bs + list(comb)

            assoc_time = args.initialize[1]
            pre_value = args.initialize[2]

        serv_bs = comb[0]
        ho_events = list()
        start = args.begin #- min(args.ttt, assoc_time)
        #if start < 0:
        #    start = 0

        end = start+args.timeslots

        comb, ho_events, comb_objval = eval_comb(comb, ho_events, SNR, nodes, start, end, args.ttt, False)

        '''
        ho_events.append(end)
        if len(ho_events) < args.length:
            comb = comb [:len(ho_events)]
        '''

        if ho_events:
            #obj_value.append([comb, calc_objvalue(comb, ho_events, SNR, nodes, args.begin, args.begin+args.timeslots), ho_events])
            obj_value.append([comb, pre_value + comb_objval, ho_events])

        else:
            y_var = 0
            for t in range(start, end):
                if t in nodes[0]['packets']:
                    y_var += 1
            obj_value.append([serv_bs, pre_value + y_var + sum(SNR[serv_bs['index']][0][start:end]), [end]])

        #print(n, obj_value[-1])

    best = max(obj_value, key=lambda x: x[1])
    print(best)

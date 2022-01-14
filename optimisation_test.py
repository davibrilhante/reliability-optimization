#! /bin/env python3
# -*- coding : utf8 -*-

from itertools import combinations_with_replacement
from itertools import permutations
import argparse
import json
import numpy as np
from decompressor import decompressor
from optimization import test_api

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
parser.add_argument('-s','--seed', type=int, default=1, 
                        help='seed')


args = parser.parse_args()

np.random.seed(args.seed)

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
                if s_bs['index'] == 3 and t_bs['index'] == 5:
                    print(k, SNR[s_bs['index']][0][k], SNR[t_bs['index']][0][k])
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

def check_disconnection(bs : dict, tolerance : int, SNR : list, slot : int) -> bool:
    if todb(SNR[bs['index']][0][slot]) < tolerance:
        return True

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
    disconnected_interval = 0

    print('serv',serv_bs, start)
    for targ_bs in combination[1:]:
        for t in range(start, end):
            events = []
            bs_seq = [serv_bs]
            if t - start >= tau - args.initialize[1]:
                # IF THERE IS A HANDOVER OPPORTUNITY
                if check_handover(serv_bs, targ_bs, SNR, t, tau, 3, 0):
                    # IF STILL THERE IS TIME AND BASE STATIONS, CHECK FOR FURTHER HANDOVERS
                    if (end - start > 1) and len(combination[2:]) > 1:
                        # CALL THE FUNCTION RECURSIVELY TO CHECK FOR HANDOVERS
                        t_bs, event = eval_comb(combination[1:], events, SNR, nodes, t, end, tau, True)

                        bs_seq += t_bs
                        events += event

                        # AFTER RECURSIONS, RETURN THE SEQUENCE OF BS AND WHEN HANDOVER TO THEM
                        if recursion:
                            return bs_seq, events

                    # IF THERE IS NO TIME OR MORE BS, DESPITE A HANDOVER IS POSSIBLE
                    # THEN RETURN THE LAST BS AND TIME SLOT
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

    if recursion:
        return [serv_bs], [t]

    else:
        return combination, ho_events, obj_value


if __name__=="__main__":

    seed = str(np.random.randint(0, 60))
    Lambda = "{0:.3f}".format(np.random.choice([0.001*i + 0.001 for i in range(10)]))
    args.inputFile = 'instances/full-scenario/22/'+Lambda+'/'+seed
    
    args.timeslots = np.random.randint(0, 500)
    args.begin = np.random.randint(0, 203000 - args.timeslots)
    print('instances/full-scenario/22/'+Lambda+'/'+seed, args.begin, args.begin+args.timeslots)

    if not args.initialize:
        result, SNR, network, nodes = test_api(args.inputFile, args.begin, args.begin + args.timeslots, args.ttt)

        if result['status']:
            args.initialize = [result['last_bs'], result['assoc_time'], result['pre_objval']]
            print(args.initialize)

        else:
            print('Infeasible')
            exit()

    m_bs = len(network)
    n_ue = len(nodes)
 
    #combinations = gen_combninations(network, args.length)
    combinations = [[network[5], network[1]]]
    #SNR = snr_processing(scenario, network, nodes, channel, LOS)

    obj_value = list() 
    pre_value = 0
    assoc_time = 0

    print(result['assoc'])
    print(result['obj'])

    for n, comb in enumerate(combinations):
        if args.initialize:
            last_bs = list([network[args.initialize[0]]])
            comb = last_bs + list(comb)

            assoc_time = args.initialize[1]
            pre_value = args.initialize[2]

        serv_bs = comb[0]
        ho_events = list()
        start = args.begin

        end = start+args.timeslots

        comb, ho_events, comb_objval = eval_comb(comb, ho_events, SNR, nodes, start, end, args.ttt, False)


        if ho_events:
            obj_value.append([comb, pre_value + comb_objval, ho_events])

        else:
            y_var = 0
            for t in range(start, end):
                if t in nodes[0]['packets']:
                    y_var += 1
            obj_value.append([serv_bs, pre_value + y_var + sum(SNR[serv_bs['index']][0][start:end]), [end]])


    #print(obj_value)
    best = max(obj_value, key=lambda x: x[1])
    print(best)


    diff = abs(result['obj'] - best[1])
    print(diff)

    result_dict = {
        'instance': args.inputFile,
        'begin': args.begin,
        'timeslots': args.timeslots,
        'opt_objvalue': result['obj'],
        'opt_comb': result['assoc'],
        'test_objvalue': best[1],
        'test_comb': [best[0],best[2]],
        'diff': diff
    }

    #filename = Lambda+'-'+seed+'-'+str(args.begin)+'-'+str(args.begin+args.timeslots)
    filename = 'test-optimisation'+str(args.seed)
    with open(filename, 'w') as jsonfile:
        json.dump(result_dict, jsonfile, indent=4)

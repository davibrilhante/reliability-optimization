#! /usr/bn/env python3
# -*- coding : utf8 -*-

import numpy as np
from os import listdir, path
from json import load
from json import decoder
from scipy.stats import norm
from decompressor import decompressor
import sys
from os import get_terminal_size


def todb(x : float) -> float:
    return 10*np.log10(x)

def tolin(x : float) -> float:
    return 10**(x/10)

def progress_bar(count, total, head=''):
    columns = get_terminal_size()[0]

    ratio = round(count/total, 2)
    percentage = int(100*ratio)
    
    foot = '{percent:03d}%'.format(percent=percentage)
    barlen = columns - len(foot+head) - 6

    formatter = ''
    if barlen > 0:
        bar = '='*(int(ratio*barlen)-1)+'>'+' '*int((1-ratio)*barlen)
        formatter = '\r{head} [{bar}] {foot}'.format(head=head,bar=bar,foot=foot)


    sys.stdout.write(formatter)
    sys.stdout.flush()

class load_result:
    def __init__(self, rootdir):
        self.rootdir = rootdir

    #List available metrics
    def get_metrics(self, sample):
        return list(sample.keys())
        

    # This method process the files on a dir to extract results
    # given a string separator and a int or slice as index
    def load(self, separator = '-', index = -1, relax=True):
        data = {}
        for filename in listdir(self.rootdir):
            name = filename.split(separator)[index]
            if path.isfile(self.rootdir+filename):
                filename = self.rootdir+filename
                try:
                    with open(filename, 'r') as jsonfile:
                        data[name] = load(jsonfile)
                except decoder.JSONDecodeError:
                    if relax:
                        print(filename)
                    else:
                        exit()

        key = list(data.keys())[0]
        return data, self.get_metrics(data[key])


class extract_metric:
    def __init__(self, data):
        self.data = data

    def raw(self, metric):
        metric_array = []
        for instance in self.data.keys():
            try:
                metric_array.append(self.data[instance][metric])
            except KeyError as error:
                print('Metric does not exist!')
                exit()

        return metric_array
    
    def mean(self, metric):
        return np.mean(self.raw(metric))

    def stdev(self, metric):
        return np.std(self.raw(metric))

    def var(self, metric):
        return np.var(self.raw(metric))

    def confinterval(self, metric, alpha=0.95):
        n = len(self.data)
        ci = alpha + (1-alpha)/2
        sigma = self.stdev(metric)

        return norm.ppf(ci)*((sigma/(n**0.5)))

    def errorplot(self, metric, **kwargs):
        alpha = None
        for key, value in kwargs.items():
            if key == 'alpha':
                alpha = value

        if alpha is not None:
            return self.mean(metric), self.confinterval(metric, alpha)

        else:
            return self.mean(metric), self.confinterval(metric)

class load_instance:
    def __init__(self, rootdir):
        self.rootdir = rootdir


    def load(self, separator = '-', index = -1):
        data = {}
        for filename in listdir(self.rootdir):
            name = filename.split(separator)[index]
            if path.isfile(self.rootdir+filename):
                filename = self.rootdir+filename
                with open(filename, 'r') as jsonfile:
                    data[name] = load(jsonfile)

                decompressor(data[name])

        return data


def calc_avg_blockage(scenario, result):
    blk = 0
    episodes = 0
    los = False
    for assoc in result['association']:
        for t in range(assoc[1], assoc[2]):
            if scenario['blockage'][assoc[0]][0][t] == 0:
                blk += 1
                # was LoS but now changed to NLoS, a new blockage episode
                if los:
                    episodes += 1
                    los = False
            else:
                los = True

    if episodes > 0:
        return blk, blk/scenario['scenario']['simTime'], episodes, blk/episodes
    else:
        return blk, blk/scenario['scenario']['simTime'], episodes, 0

def calc_gap(data, simtime, hit):
    assoc_time = 0
    for assoc in data['association']:
        assoc_time += assoc[2] - assoc[1]

    total_hit = data['handover']*hit

    if len(assoc) > 3:
        return ((simtime - assoc_time) + total_hit)/simtime
    else:
        return (simtime - assoc_time)/simtime

def calc_bsdist(scenario, result):
    dist = []
    vel_x = scenario['userEquipment'][0]['speed']['x']
    vel_y = scenario['userEquipment'][0]['speed']['y']
    
    for assoc in result['association']:
        bs = assoc[0]
        init = assoc[1]
        end = assoc[2]
        for t in range(init, end):
            ue_x = vel_x*t*1e-3/3.6
            ue_y = vel_y*t*1e-3/3.6
            bs_x = scenario['baseStation'][bs]['position']['x']
            bs_y = scenario['baseStation'][bs]['position']['y']
            bs_dist = np.hypot(ue_x-bs_x, ue_y-bs_y)

            dist.append(bs_dist) 

    return np.mean(dist)


def calc_handoverfailures(data, scenario, ttt, hit=68, t310=1000, n310=1, t311=1, n311=1):
    failures = 0
    vel_x = scenario['userEquipment'][0]['speed']['x']
    vel_y = scenario['userEquipment'][0]['speed']['y']

    bs_antenna_gain = 15
    ue_antenna_gain = 15

    rrcprocdelay = 5
    hodecisiondelay = 15
    x2delay = 5
    x2procdelay = 5
    admcontroldelay = 20
    rrctxdelay = 5
    hocmddelay = 15

    state2 = rrcprocdelay + hodecisiondelay + 2*(x2delay+x2procdelay) + admcontroldelay + rrctxdelay + hocmddelay

    

    prev = []
    for n, assoc in enumerate(data['association']):
        counter310 = 0
        timer310 = 0
        counter311 = 0
        bs = assoc[0]
        bs_power = scenario['baseStation'][bs]['txPower']
        noise_power = scenario['channel']['noisePower']

        init = 0
        if n > 0:
            init = prev[2]

        for t in range(init,assoc[2]):
            ue_x = vel_x*t*1e-3/3.6
            ue_y = vel_y*t*1e-3/3.6
            bs_x = scenario['baseStation'][bs]['position']['x']
            bs_y = scenario['baseStation'][bs]['position']['y']
            bs_dist = np.hypot(ue_x-bs_x, ue_y-bs_y)

            if scenario['blockage'][bs][0][t] == 1:
                path_loss = 61.4 + 10*2*np.log10(bs_dist)
            else:
                path_loss = 72 + 10*2.92*np.log10(bs_dist)

            snr = (bs_power + bs_antenna_gain + ue_antenna_gain - path_loss) - noise_power 

            if snr >= -6:
                counter311 += 1

            if snr <-6 and counter310 > 0:
                timer310 +=1

            if snr <= -8:
                counter310 += 1
                timer310 +=1

            if timer310 > 0 and (t == (assoc[2] - state2)):
                failure += 1
                break

            if timer310 >= t310 and (t > (assoc[2] - state2 - ttt)) and  (t < (assoc[2] - state2)):
                failure += 1
                break

            if t <= init + hit and counter310 > 0:
                failure += 1
                break


        prev = assoc



    return failures




def calc_pingpongs(data, min_tos=1000):
    pingpongs = 0
    n_assoc = len(data['association'])

    for n, assoc in enumerate(data['association']):
        '''
        if n>0 and n < n_assoc-1:
            nxt = data['association'][n+1]
            tos = assoc[2] - assoc[1]
 
            if (prev[0] == nxt[0] and
                    assoc[0] != nxt[0] and
                    tos <= min_tos):
                pingpongs += 1

        prev = assoc
        '''

        prev = n
        backtime = 0
        acc_pingpongs = 0
        while prev>0:
            backtime += data['association'][prev][1] - data['association'][prev-1][1]
            if  backtime <= min_tos:
                if assoc[0] == data['association'][prev-1][0]:
                    pingpongs += 1
            prev -= 1


    return pingpongs/data['handover']


def calc_diffs(data):
    snr_diff = []
    similarity = []
    snr_dict = {}
    similarity_dict = {}
    for vel in data['opt'].keys():
        snr_dict[vel] = {} 
        similarity_dict[vel] = {}
        for ttt in data['opt'][vel].keys():
            snr_dict[vel][ttt] = []
            similarity_dict[vel][ttt] = []

            for var in data['opt'][vel][ttt].keys():
                for instances in data['opt'][vel][ttt][var].keys():
                    try:
                        snr_diff.append(tolin(data['opt'][vel][ttt][var][instances]['snr']) - 
                                tolin(data['base'][vel][ttt][var][instances]['sinr']))

                        similarity.append(calc_similarity(data['opt'][vel][ttt][var][instances]['association'],
                                        data['base'][vel][ttt][var][instances]['association']))
                    except KeyError:
                        pass


                snr_dict[vel][ttt].append(np.mean(snr_diff))
                similarity_dict[vel][ttt].append(np.mean(similarity))

    return snr_dict, similarity_dict


def calc_similarity(a_instance, b_instance):
    equals = 0
    for x in a_instance:
        for y in b_instance:
            factor  = 0
            if x[1] < y[1]:
                if x[2] < y[1]:
                    continue

                elif x[2] >= y[1] and x[2] <= y[2]:
                    factor = x[2] - y[1]
                    #print(x[0],y[0],x[1],'<',y[1],x[2],'<=',y[2])

                elif x[2] > y[2]:
                    factor = y[2] - y[1]
                    #print(x[0],y[0],x[1],'<',y[1],x[2],'>',y[2])

            elif x[1] > y[1]:
                if x[1] > y[2]:
                    continue

                elif x[1] < y[2] :
                    if x[2] <= y[2]:
                        factor = x[2] - x[1]
                        #print(x[0],y[0],x[1],'>',y[1],x[2],'<=',y[2])

                    elif x[2] > y[2]:
                        factor = y[2] - x[1]
                        #print(x[0],y[0],x[1],'>',y[1],x[2],'>',y[2])

            if x[0]==y[0]:
                equals += factor

            else:
                equals += 0

    return equals/x[2]

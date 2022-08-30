#! /usr/bn/env python3
# -*- coding : utf8 -*-

import numpy as np
from os import listdir, path
from json import load
from scipy.stats import norm
from decompressor import decompressor


def todb(x : float) -> float:
    return 10*np.log10(x)

def tolin(x : float) -> float:
    return 10**(x/10)

class load_result:
    def __init__(self, rootdir):
        self.rootdir = rootdir

    #List available metrics
    def get_metrics(self, sample):
        return list(sample.keys())
        

    # This method process the files on a dir to extract results
    # given a string separator and a int or slice as index
    def load(self, separator = '-', index = -1):
        data = {}
        for filename in listdir(self.rootdir):
            name = filename.split(separator)[index]
            if path.isfile(self.rootdir+filename):
                filename = self.rootdir+filename
                with open(filename, 'r') as jsonfile:
                    data[name] = load(jsonfile)



        return data, self.get_metrics(data['0'])


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

    return blk, blk/scenario['scenario']['simTime'], episodes, blk/episodes

def calc_gap(data, simtime, hit):
    assoc_time = 0
    for assoc in data['association']:
        assoc_time += assoc[2] - assoc[1]

    total_hit = data['handover']*hit

    if len(assoc) > 3:
        return (simtime - assoc_time) + total_hit
    else:
        return (simtime - assoc_time)

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


def calc_diffs(data):
    snr_diff = []
    similarity = []
    snr_dict = {}
    similarity_dict = {}
    for vel in data['opt'].keys():
        snr_dict[vel] = []
        similarity_dict[vel] = []

        for var in data['opt'][vel].keys():
            for instances in data['opt'][vel][var].keys():
                snr_diff.append(tolin(data['opt'][vel][var][instances]['snr']) - 
                        tolin(data['base'][vel][var][instances]['sinr']))

                similarity.append(calc_similarity(data['opt'][vel][var][instances]['association'],
                                data['base'][vel][var][instances]['association']))


            snr_dict[vel].append(np.mean(snr_diff))
            similarity_dict[vel].append(np.mean(similarity))

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

#! /usr/bn/env python3
# -*- coding : utf8 -*-

import numpy as np

from os import listdir, path
from json import load
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as lines
import matplotlib.patches as mpatches
from scipy.stats import norm
from matplotlib import use
from decompressor import decompressor

def calc_bsdist(instance, simtime, vel, network):
    dist_list = []
    for assoc in instance['association']:
        bs = assoc[0]
        init = assoc[1]
        end = assoc[2]
        for t in range(init, end):
            ue_x = vel*t*1e-3/3.6
            ue_y = vel*t*1e-3/3.6
            bs_x = network[bs]['position']['x']
            bs_y = network[bs]['position']['y']
            bs_dist = np.hypot(ue_x-bs_x, ue_y-bs_y)

            dist_list.append(bs_dist) 

    return np.mean(dist_list)

def todb(x : float) -> float:
    return 10*np.log10(x)

def tolin(x : float) -> float:
    return 10**(x/10)

plot_dict = {
        (22,1280):{
            'label':'v=30km/h,$\\tau$=1280',
            'marker':'^',
            'size':10,
            'color':'blue'
            },

        (43,640):{
            'label':'v=60km/h,$\\tau$=640',
            'marker':'+',
            'size':10,
            'color':'green'
            },
        (64,480):{
            'label':'v=90km/h,$\\tau$=480',
            'marker':'o',
            'size':10,
            'color':'red'
            },
        'similarity':{
            'ylabel':'Overlap Similarity'
            },
        'trigger_diff':{
            'ylabel':'Handover Trigger Time Difference'

            },
        'dist_diff':{
            'ylabel':'Average Compared BS Distance '
            },
        'snr_diff':{
            'ylabel':'Average SNR Difference '
            },

        'opt_uedistbs':{
            'ylabel':'Average BS Distance '
            }
        }

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

    return tolin(calc_recv(base, user, channel, los, t) - noise_power)

def calc_bssnr(instance, bs, init, end):
    try:
        with open(instance) as json_file:
            data = load(json_file)
    except Exception as e:
        print(e)
        print(10)
        exit()

    scenario = data['scenario']
    channel = data['channel']
    LOS = data['blockage']
    for p in data['baseStation']:
        network.append(p)
    for p in data['userEquipment']:
        nodes.append(p)

    dist_list = []
    for t in range(init, end):
        if LOS[bs][0][t] == 1:
            los = True
        else:
            los = False

        dist_list.append(calc_snr(network[bs], nodes[0], channel, los, t))

    return np.mean(dist_list)

def assoc_to_list(assoc):

    if len(assoc) >=3:
        assoc = [i[:3] for i in assoc]

    if assoc[0][1] != 0:
        assoc_list = [assoc[0][0] for i in range(assoc[1][1])]
    else:
        assoc_list = [None for i in range(assoc[0][1])]

    prev_end = assoc[0][1]

    for bs, init, end in assoc[1:]:
        for t in range(prev_end,end):
            assoc_list.append(bs)
        prev_end = end


    return assoc_list


if __name__ == '__main__':
    plt.rc('text', usetex=True)                                                 
    plt.rc('font', family='serif', size=13)
    use('PS')

    instance_file = 'instances/full-scenario/22/0.001/0'

    network = []
    nodes = []

    try:
        with open(instance_file) as json_file:
            data = load(json_file)
    except Exception as e:
        print(e)
        print(10)
        exit()

    decompressor(data)

    scenario = data['scenario']
    channel = data['channel']
    LOS = data['blockage']
    for p in data['baseStation']:
        network.append(p)
    for p in data['userEquipment']:
        nodes.append(p)

    line_plot = True
    
    vel_ttt =[(22,1280,203647),
                (43,640,101823),
                (64,480,67882)]


    metrics = {}
    metrics['similarity'] = {}
    metrics['trigger_diff'] = {}
    metrics['dist_diff'] = {}
    metrics['snr_diff'] = {}
    metrics['opt_uedistbs'] = {}
    metrics['base_uedistbs'] = {}
    lambdas = [round(2e-3*i + 1e-3, 3) for i in range(5)]
    delays = [2*i + 1 for i in range(5)]
    capacity = '750e6'

    base = {}
    opt = {}
    
    for vel, ttt, simtime in vel_ttt:
        metrics['similarity'][vel,ttt] = {} 
        metrics['trigger_diff'][vel,ttt] = {}
        metrics['dist_diff'][vel,ttt] = {}
        metrics['snr_diff'][vel,ttt] = {}
        metrics['opt_uedistbs'][vel,ttt] = {}
        metrics['base_uedistbs'][vel,ttt] = {}
        for block in lambdas:

            metrics['similarity'][vel,ttt][block] = []
            metrics['trigger_diff'][vel,ttt][block] = []
            metrics['dist_diff'][vel,ttt][block] = []
            metrics['snr_diff'][vel,ttt][block] = [] 
            metrics['opt_uedistbs'][vel,ttt][block] = []
            metrics['base_uedistbs'][vel,ttt][block] = []

            line_data = []
            diff_data = []
            dist_diff = []
            for delay in delays:
                opt[vel,ttt], opt_metrics = load_result('instances/no-interference/opt/{tau}/{vel}/{Lambda}/{cap}/{Del}/'.format(
                    tau=ttt,
                    vel=vel,
                    Lambda=block,
                    cap=capacity,
                    Del=delay)).load()

                base[vel,ttt], base_metrics = load_result('instances/no-interference/base/{tau}/{vel}/{Lambda}/{cap}/{Del}/'.format(
                    tau=ttt,
                    vel=vel,
                    Lambda=block,
                    cap=capacity,
                    Del=delay)).load()

                for instance in base[vel,ttt].keys():
                    #instance_file = 'instances/full-scenario/{vel}/{block}/{inst}'.format(vel=vel, block=block, inst=instance)

                    #opt_assoc = assoc_to_list(opt[vel,ttt][instance]['association'])
                    #base_assoc = assoc_to_list(base[vel,ttt][instance]['association'])
                    opt_assoc = opt[vel,ttt][instance]['association']
                    base_assoc = base[vel,ttt][instance]['association']

                    '''
                    network = []
                    nodes = []

                    try:
                        with open(instance_file) as json_file:
                            data = load(json_file)
                    except Exception as e:
                        print(e)
                        print(10)
                        exit()
                    
                    decompressor(data)

                    scenario = data['scenario']
                    channel = data['channel']
                    LOS = data['blockage']
                    for p in data['baseStation']:
                        network.append(p)
                    for p in data['userEquipment']:
                        nodes.append(p)

                    snr_list = []

                    base_counter = 0
                    opt_counter = 0
                    for t in range(simtime):
                        if t > base_assoc[base_counter][2]:
                            base_counter += 1
                        base_bs = base_assoc[base_counter][0]

                        if LOS[base_bs][0][t] == 1:
                            los = True
                        else:
                            los = False
                        snr_base = calc_snr(network[base_bs],nodes[0], channel, los, t)



                        if t > opt_assoc[opt_counter][2]:
                            opt_counter += 1
                        opt_bs = opt_assoc[opt_counter][0]
                        if LOS[opt_bs][0][t] == 1:
                            los = True
                        else:
                            los = False
                        snr_opt = calc_snr(network[opt_bs],nodes[0], channel, los, t)

                        snr_list.append(snr_opt - snr_base)
                    metrics['snr_diff'][vel,ttt][block].append(np.mean(snr_list))

                    '''



                    minlen = min(len(opt_assoc),len(base_assoc))
                    maxlen = max(len(opt_assoc),len(base_assoc))
                    equals = 0

                    #metrics['opt_uedistbs'][vel,ttt][block].append(calc_bsdist(opt[vel,ttt][instance],simtime, vel, network))
                    #metrics['base_uedistbs'][vel,ttt][block].append(calc_bsdist(base[vel,ttt][instance],simtime, vel, network))

                    
                    #for x,y in zip(opt_assoc[:maxlen],base_assoc[:maxlen]):
                    for x in opt_assoc:
                        for y in base_assoc:
                            factor  = 0
                            bs_dist = np.hypot(network[x[0]]['position']['x'] - network[y[0]]['position']['x'], 
                                    network[x[0]]['position']['y'] - network[y[0]]['position']['y'])

                            if x[1] < y[1]:
                                if x[2] < y[1]:
                                    #pass
                                    continue

                                elif x[2] >= y[1] and x[2] <= y[2]:
                                    factor = x[2] - y[1]
                                    print(x[0],y[0],x[1],'<',y[1],x[2],'<=',y[2])

                                elif x[2] > y[2]:
                                    factor = y[2] - y[1]
                                    print(x[0],y[0],x[1],'<',y[1],x[2],'>',y[2])

                            elif x[1] > y[1]:
                                if x[1] > y[2]:
                                    #pass
                                    continue

                                elif x[1] < y[2] :
                                    if x[2] <= y[2]:
                                        factor = x[2] - x[1]
                                        print(x[0],y[0],x[1],'>',y[1],x[2],'<=',y[2])

                                    elif x[2] > y[2]:
                                        factor = y[2] - x[1]
                                        print(x[0],y[0],x[1],'>',y[1],x[2],'>',y[2])

                            if x[0]==y[0]:
                                equals += factor

                            else:
                                equals += 0#factor*(1 - (bs_dist/1350))
                        
                        '''
                            #snr_opt = calc_snr(network[x[0]],nodes[0], channel, los : bool, t=0)
                            #snr_base = calc_snr()

                        #if x[0] == y[0]:
                        if x == y:
                            equals += 1
                            #difference = x[2] - y[2]
                            #diff_data.append(difference/simtime)
                        else:
                            equals += 1 - (bs_dist/1350)
                        '''

                        dist_diff.append(bs_dist)


                    line_data.append(equals/x[2])

            print(list(metrics['opt_uedistbs'][vel,ttt].keys()))

            metrics['similarity'][vel,ttt][block] = line_data
            metrics['trigger_diff'][vel,ttt][block] = diff_data
            metrics['dist_diff'][vel,ttt][block] = dist_diff
            

    for metric in metrics.keys():
        for vel, ttt, simtime in vel_ttt:
            if line_plot:
                if metric == 'base_uedistbs' and metric == 'opt_uedistbs':
                    pass
                print(metric, vel)

                plt.plot(lambdas, [np.mean(metrics[metric][vel,ttt][b]) for b in lambdas], 
                        label=plot_dict[vel,ttt]['label'],
                        color=plot_dict[vel,ttt]['color'],
                        marker=plot_dict[vel,ttt]['marker'],
                        markeredgecolor=plot_dict[vel,ttt]['color'],
                        markersize=plot_dict[vel,ttt]['size'],
                        markerfacecolor='None',
                        linewidth=0.5)

                if metric == 'opt_uedistbs':

                    plt.plot(lambdas, [np.mean(metrics['base_uedistbs'][vel,ttt][b]) for b in lambdas], 
                            label=plot_dict[vel,ttt]['label'],
                            color=plot_dict[vel,ttt]['color'],
                            marker=plot_dict[vel,ttt]['marker'],
                            markeredgecolor=plot_dict[vel,ttt]['color'],
                            markersize=plot_dict[vel,ttt]['size'],
                            markerfacecolor='None',
                            linestyle='--',
                            linewidth=0.5)

                    #line.set_dashes((15,10,15,10))


                plot_name = metric+'-line-plot.eps'

                plt.legend()



            else:
                boxplot_array = []
                for n, block in enumerate(lambdas):
                    for vel, ttt, _ in vel_ttt:
                        boxplot_array.append(metrics[metric][vel,ttt][block])

                bplot = plt.boxplot(boxplot_array,
                            widths=0.25,
                            patch_artist=True,
                            vert=True,
                            positions = [0.3*i + n for n in range(1,6) for i in range(-1,2)]) 
                plt.xticks(range(1,6),lambdas)

                colors = ['pink', 'lightblue', 'lightgreen']
                labels = ['v=30km/h,$\\tau$=1280','v=60km/h,$\\tau$=640','v=90km/h,$\\tau$=480']
                handles = []
                for color,label in zip(colors, labels):
                    handles.append(mpatches.Patch(color=color, label=label))

                plt.legend(handles=handles, 
                        ncol=3,
                        bbox_to_anchor=(-0.15, 1.15),
                        columnspacing=0.5,
                        loc='upper left')

                colors *= 5
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)


                plot_name = metric+'-box-plot.eps'

        plt.xlabel('Blockage Density $\lambda$ [objects/$m^2$]')
        plt.ylabel(plot_dict[metric]['ylabel'])
        plt.grid()
        plt.savefig('test2-'+plot_name,
                dpi=300,                                                    
                bbox_inches="tight",                                        
                transparent=True,                                           
                format="eps")

        plt.show()

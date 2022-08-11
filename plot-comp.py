#! /usr/bn/env python3
# -*- coding : utf8 -*-

import numpy as np

from os import listdir, path
from json import load
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as lines
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm
from matplotlib import use
from decompressor import decompressor
from import 


metric_dict = {
                'rsrp' : {
                            'title' : 'Average Received Signal Reference Power',
                            'ylabel': 'RSRP [dBm]',
                            'ypercent' : False
                        },
                'gap' : {
                            'title' : 'Average Interruption Time',
                            'ylabel': 'Interruption Time  [ms]',
                            'ypercent' : False 
                    },

                'partDelay':{
                            'title' : 'Average Packets delivered before delay expiration ',
                            'ylabel': 'Delivery Rate [\%]',
                            'ypercent' : True
                },
                'handover' : {
                            'title' : 'Average Number of handovers',
                            'ylabel': ' Handovers',
                            'ypercent' : False 
                    },
                'pingpong' : {
                            'title' : 'Average Ping-Pong Handover Rate',
                            'ylabel': 'Ping-pongs [\%]',
                            'ypercent' : True 
                    },
                'handoverRate' : {
                            'title' : 'Average Handover Rate',
                            'ylabel': 'Handovers per second',
                            'ypercent' : False 
                    },
                'capacity' : {
                            'title' : 'Average Shannon Capacity',
                            'ylabel': 'Capacity [bps]',
                            'ypercent' : False 
                    },
                'deliveryRate': {
                            'title' : 'Average packet delivery rate',
                            'ylabel': 'Delivery Rate [\%]',
                            'ypercent' : True
                    },
                'delay' : {
                            'title' : 'Average Packet Delay',
                            'ylabel': 'Delay [ms]',
                            'ypercent' : False 
                },
                'bs_dist' : {
                            'title' : 'Average Distant to serving BS',
                            'ylabel': 'Average Distance',
                            'ypercent' : False 
                }
            }


plotter = {
        (22,1280):{
            'label':'v=30km/h,$\\tau$=1280',
            'marker':'^',
            'size':10,
            'color':'blue'
            },
        (22,640):{
            'label':'v=30km/h,$\\tau$=640',
            'marker':'v',
            'size':10,
            'color':'orange'
            },
        (43,640):{
            'label':'v=60km/h,$\\tau$=640',
            'marker':'+',
            'size':10,
            'color':'green'
            },
        (64,640):{
            'label':'v=90km/h,$\\tau$=640',
            'marker':'s',
            'size':10,
            'color':'cyan'
            },
        (64,320):{
            'label':'v=60km/h,$\\tau$=320',
            'marker':'*',
            'size':6,
            'color':'gray'
            },
        (64,480):{
            'label':'v=90km/h,$\\tau$=480',
            'marker':'o',
            'size':10,
            'color':'red'
            },
        'opt':{
            'linestyle':'-',
            'linewidth':3
            },
        'base':{
            'linestyle':'--',
            'linewidth':3
            }
        }

param_dict = {
        'block':{
            'xaxis':'Blockage Density $\lambda$ [objects/$m^2$]',
            'xticks':[round(2e-3*i + 1e-3, 3) for i in range(5)],
            'xticks_style':'sci',
            'title':'\nCapacity = 750Mbps, Delay = 1ms'
            },
        'capacity':{
            'xaxis':'Required Capacity [bps]',
            'xticks':[750e6], #['1e6','100e6','500e6'],
            'xticks_style':'',
            'title':'\nBlockage Density $\lambda$ = 0.005, Delay = 1ms'
            },
        'delay':{
            'xaxis':'Required Delay [ms]',
            'xticks':[2*i+1 for i in range(5)], #['1','5','10','50'],
            'xticks_style':'',
            'title':'\nBlockage Density $\lambda$ = 0.005, Capacity = 750Mbps'
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
        



def calc_gap(data, simtime = 203647, hit = 68):

    on_intervals = []
    for variable, var_dict in data['variables'].items():
        #Gets the x variables and the intervals when they are 1
        varname = variable.split('_')[0]
        if varname == 'x':
            try:
                for interval in var_dict['1.0']:
                    on_intervals.append(interval[1] - interval[0])
            except KeyError:
                pass

    # subtracts the online intervals from the total time to get the
    # the periods when the ue was disconnected
    offline = simtime - (sum(on_intervals) + len(on_intervals) - 1)


    try:
        outofsync = data['gap'] #+ data['handover']*hit

    except:
        outofsync = 0
        #gap = data['handover']*hit

    total_hit = data['handover']*hit

    outofsync = outofsync + offline     

    gap = outofsync + total_hit

    
    return gap, total_hit, outofsync


def handover_table(data):
    table = {}

    for instance, values in data.items():
        table[instance] = []

        for tup in values['association']:
            table[instance].append(tup[0])
            #table[instance].append(tup[2])
    
    return table


def similarity_calc(opt_table, base_table, simil_func, weight_func):
    similarities = []
    #prob = probability_calc(opt_table, base_table) 

    for instance in opt_table.keys():
        similarities.append(0)
        '''
        if len(opt_table[instance]) < len(base_table[instance]):
            for i in range(len(base_table[instance])-len(opt_table[instance])):
                opt_table[instance].append(None)
        elif len(opt_table[instance]) > len(base_table[instance]):
            for i in range(len(opt_table[instance])-len(base_table[instance])):
                base_table[instance].append(None)
        '''

        max_len = min(len(opt_table[instance]), len(base_table[instance]))
        prob = probability_calc(opt_table, base_table) 
        #Instance similarity
        for n, values in enumerate(zip(opt_table[instance[:max_len]], base_table[instance][:max_len])):
            x = values[0]
            y = values[1]
            if x == y:
                similarities[-1] += simil_func(x,y,n,prob,True)*weight_func(x,y,n,prob,True)
            else:
                similarities[-1] += simil_func(x,y,n,prob,False)*weight_func(x,y,n,prob,False)
        
    return np.mean(similarities)

#def frequency_calc(opt_table, base_table):
def probability_calc(opt_table, base_table):
    frequencies = {}

    table = {
            'opt':opt_table,
            'bas':base_table
            }

    attr_counter = 0
    while attr_counter < max(len(opt_table), len(base_table)):
        frequencies[attr_counter] = {}
        for seg in table.keys():
            for values in table[seg].values():
                #print(seg,values)
                try:
                    frequencies[attr_counter][values[attr_counter]] += 1
                except KeyError:
                    frequencies[attr_counter][values[attr_counter]] = 1
                except IndexError:
                    pass

        total = sum(frequencies[attr_counter].values())

        for attr_value in frequencies[attr_counter].keys():
            frequencies[attr_counter][attr_value] /= total

        attr_counter += 1

    return frequencies

def overlap_sim(x,y,n,prob,state):
    if state:
        return 1
    else:
        return 0

def eskin_sim(x,y,n,prob,state):
    if state:
        return 1
    else:
        nb = len(prob[n].keys())
        return nb**2/(2 + nb**2)

def of_sim(x,y,n,prob,state):
    if state:
        return 1
    else:
        return 1/(1+np.log(prob[n][x])*np.log(prob[n][y]))

def lin_sim(x,y,n,prob,state):
    if state:
        return 2*np.log(prob[n][x])
    else:
        #print(x,y)
        return 2*np.log(prob[n][x]+prob[n][y])

def overlap_weight(x,y,n,prob,state):
    n_attr = len(prob.keys())
    return 1/n_attr

def lin_weight(x,y,n,prob,state):
    sum_prob_logs = 0
    for i in prob.keys():
        try:
            sum_prob_logs += np.log(prob[i][x]) + np.log10(prob[i][y])
        except KeyError:
            pass

    return 1/sum_prob_logs 


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


def calc_snr2(rsrp : float, channel : dict) -> float:
    noise_power = channel['noisePower']

    return tolin(rsrp - noise_power)


def calc_bsdist(instance, simtime, vel, network):
    dist_list = []
    for assoc in instance['association']:
        bs = assoc[0]
        init = assoc[1]
        end = assoc[2]
        for t in range(init, end):
            ue_x = vel*t*1e-3/3.6
            ue_y = vel*t*1e-3/3.6
            bs_x = network[bs]['x']
            bs_y = network[bs]['y']
            bs_dist = np.hypot(ue_x-bs_x, ue_y-bs_y)

            dist_list.append(bs_dist) 

    return np.mean(dist_list)


def calc_bssnr(instance, result, simtime, vel, network):
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
    for assoc in instance['association']:
        bs = assoc[0]
        init = assoc[1]
        end = assoc[2]
        for t in range(init, end):
            if LOS[m][n][beginning+t] == 1:
                los = True
            else:
                los = False

            dist_list.append(calc_snr(network[bs], nodes[0], channel, los, t))

    return np.mean(dist_list)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    use('PS')

    xfmt = ScalarFormatter()
    xfmt.set_powerlimits((-1,3))

    plot_opt = {}
    plot_base = {}

    #vel_ttt =[(22,1280),(22,960),(43,640),(64,480)]
    vel_ttt =[(22,1280,203647),
                #(22,640,203647),
                (43,640,101823),
                #(64,640,67882),
                (64,480,67882)]
                #(64,320,67882)]

    variables = {
            'block':['{0:.3f}'.format(0.002*i + 0.001) for i in range(5)],
            #'capacity':['750e6'], #['1e6', '100e6', '500e6'],#, '1000e6'],
            'delay':[2*i+1 for i in range(5)]#1,5,10,50]
            }

    fixed = {
            'block':0.005,
            'capacity':'750e6',
            'delay':1
            }

    ### Create base data
    try:
        with open('instances/full-scenario/22/0.001/0') as json_file:
            data = load(json_file)
    except Exception as e:
        print(e)
        print(10)
        exit()

    network = {}
    for p in data['baseStation']:
        network[p['index']] = p['position']

    plt_OOS_opt = {}
    plt_OOS_base = {}

    similarity = {}

    for var in variables.keys():
        print(var)
        opt = {}
        base= {}
        for vel, ttt, simtime in vel_ttt:
            plot_opt[vel,ttt] = {}
            plot_base[vel,ttt] = {}

            plt_OOS_opt[vel,ttt] = []
            plt_OOS_base[vel,ttt] = []

            similarity[vel,ttt] = []

            for value in variables[var]:

                spec = [0,0,0]

                if var == 'block':
                    spec[0] = value
                elif var == 'capacity':
                    spec[1] = value
                elif var == 'delay':
                    spec[2] = value
                
                n = 0
                for fix,val  in fixed.items():
                    if fix != var:
                        spec[n] = val
                    n += 1

                opt[vel,ttt], opt_metrics = load_result('instances/no-interference/opt/{tau}/{vel}/{Lambda}/{cap}/{Del}/'.format(
                    tau=ttt,
                    vel=vel,
                    Lambda=spec[0],
                    cap=spec[1],
                    Del=spec[2])).load()

                base[vel,ttt], base_metrics = load_result('instances/no-interference/bas/{tau}/{vel}/{Lambda}/{cap}/{Del}/'.format(
                    tau=ttt,
                    vel=vel,
                    Lambda=spec[0],
                    cap=spec[1],
                    Del=spec[2])).load()

                opt_table = handover_table(opt[vel,ttt])
                base_table = handover_table(base[vel,ttt])
                opt_metrics.append('bs_dist')
                base_metrics.append('bs_dist')

                for instance in opt[vel,ttt].keys():
                    try:
                        opt[vel,ttt][instance]['gap'], opt[vel,ttt][instance]['HIT'], opt[vel,ttt][instance]['outofsync'] = calc_gap(opt[vel,ttt][instance],simtime)
                        opt[vel,ttt][instance]['bs_dist'] = calc_bsdist(opt[vel,ttt][instance],simtime, vel, network)
                    except KeyError as error:
                        print(spec[0], spec[1], spec[2],vel,ttt, instance, error)

                for instance in base[vel,ttt].keys():
                    try:
                        base[vel,ttt][instance]['HIT'] = sum(base[vel,ttt][instance]['log']['HIT'])
                        base[vel,ttt][instance]['bs_dist'] = calc_bsdist(base[vel,ttt][instance],simtime, vel, network)
                    except KeyError:
                        base[vel,ttt][instance]['HIT'] = 0

                    if type(base[vel,ttt][instance]['outofsync']) == type([]):
                        temp = 0
                        for OOS_interval in base[vel,ttt][instance]['outofsync']:
                            try:
                                temp += OOS_interval[1] - OOS_interval[0]
                            except IndexError:
                                temp += 1

                        base[vel,ttt][instance]['outofsync'] = base[vel,ttt][instance]['gap']-base[vel,ttt][instance]['HIT']



                plt_OOS_opt[vel,ttt].append(extract_metric(opt[vel,ttt]).mean('outofsync'))
                plt_OOS_base[vel,ttt].append(extract_metric(base[vel,ttt]).mean('outofsync'))

                for metric in base_metrics:
                    if metric in opt_metrics and metric != 'association':
                        try:
                            plot_opt[vel,ttt][metric].append(extract_metric(opt[vel,ttt]).errorplot(metric))
                            plot_base[vel,ttt][metric].append(extract_metric(base[vel,ttt]).errorplot(metric))


                        except KeyError:
                            plot_opt[vel,ttt][metric] = []
                            plot_base[vel,ttt][metric] = []

                            plot_opt[vel,ttt][metric].append(extract_metric(opt[vel,ttt]).errorplot(metric))
                            plot_base[vel,ttt][metric].append(extract_metric(base[vel,ttt]).errorplot(metric))

                #similarity[vel,ttt].append(similarity_calc(opt_table, base_table, overlap_sim, overlap_weight))
                #similarity[vel,ttt].append(similarity_calc(opt_table, base_table, eskin_sim, overlap_weight))
                similarity[vel,ttt].append(similarity_calc(opt_table, base_table, lin_sim, lin_weight))



        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif', size=13)

        #for fix,val  in fixed.items():
        #    if var != fix:
        for metric in base_metrics:
            print(var, metric)
            if metric in opt_metrics and metric not in ['association', 'HIT','outofsync']:
                locs = param_dict[var]['xticks'] #np.arange(len(variables[var]))
                fig = plt.figure(figsize=(6,10), dpi=300)
                ax = fig.add_subplot(1,1,1)

                for vel, ttt, s in vel_ttt:
                    tag = ', v={vel}, $\\tau$={ttt}'.format(vel=vel, ttt=ttt)

                    plot_array_opt = np.transpose(plot_opt[vel,ttt][metric])
                    plot_array_base = np.transpose(plot_base[vel,ttt][metric])


                    #ax.errorbar(labels, plot_array_opt[0], yerr=plot_array_opt[1], label='opt'+tag, marker='^')
                    #ax.errorbar(labels, plot_array_base[0], yerr=plot_array_base[1], label='base'+tag, marker='o')

                    ax.errorbar(locs, plot_array_opt[0], yerr=plot_array_opt[1], 
                            #label=plotter[vel,ttt]['label'], 
                            capsize=5.0,
                            markersize=10.0,
                            marker=plotter[vel,ttt]['marker'], 
                            markeredgecolor=plotter[vel,ttt]['color'], 
                            markerfacecolor='None',#plotter[vel,ttt]['color'],
                            color=plotter[vel,ttt]['color'],
                            linestyle=plotter['opt']['linestyle'],
                            linewidth=0.5)

                    line, _, __ = ax.errorbar(locs, plot_array_base[0], yerr=plot_array_base[1],
                            #label=plotter[vel,ttt]['label'], 
                            capsize=5.0,
                            markersize=10.0,
                            marker=plotter[vel,ttt]['marker'], 
                            markeredgecolor=plotter[vel,ttt]['color'], 
                            markerfacecolor='None',#plotter[vel,ttt]['color'],
                            color=plotter[vel,ttt]['color'],
                            linestyle=plotter['base']['linestyle'],
                            linewidth=0.5)
                    line.set_dashes((15,10,15,10))


                    if metric_dict[metric]['ypercent']:
                        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                    #else:
                    #    ax.set_yticks()


                ax.set_ylabel(metric_dict[metric]['ylabel'], fontsize=30)
                #ax.set_title(metric_dict[metric]['title']+param_dict[var]['title'])

                ax.set_xlabel(param_dict[var]['xaxis'], fontsize=30)
                ax.set_xticks(locs,param_dict[var]['xticks'], fontsize=30)

                plt.yticks(fontsize=30)
                #ax.xaxis.set_major_formatter(xfmt)
                #ax.ticklabel_format(axis='x',style=param_dict[var]['xticks_style'],scilimits=(0,2))
                #ax.set_xticklabels(param_dict[var]['xticks'])
                #plt.xticks(np.arange(len(param_dict[var]['xticks'])),param_dict[var]['xticks'])
                        

                legend_elements = [lines.Line2D([0],[0], color='w',
                        marker=plotter[v,t]['marker'], 
                        markersize=plotter[v,t]['size'], 
                        markeredgecolor=plotter[v,t]['color'], 
                        markerfacecolor='None') #plotter[v,t]['color']) 
                        for v,t,s  in vel_ttt]+[
                        lines.Line2D([0],[0],color='k',linestyle=plotter['opt']['linestyle']),
                        lines.Line2D([0],[0],color='k',linestyle=plotter['base']['linestyle'])]

                legend_labels = [plotter[v,t]['label'] for v,t,s in vel_ttt]+['opt','base']
                #plt.tight_layout()
                plt.legend(legend_elements,
                        legend_labels) 
                        #ncol=2, 
                        #bbox_to_anchor=(0.5, 0.15),
                        #loc='upper center')
                plt.grid()
                #'''
                plt.savefig('comparison-opt-base-{var}-{metric}.eps'.format(var=var,metric=metric),
                            dpi=300,
                            bbox_inches="tight",
                            transparent=True,
                            format="pdf")
                #'''
                #plt.show()
                plt.clf()
                plt.cla()
                plt.close()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)

        for vel, ttt, s in vel_ttt:
            locs = np.arange(len(variables[var]))
            ax2.plot(locs, similarity[vel,ttt], 
                    label=plotter[vel,ttt]['label'], 
                    marker=plotter[vel,ttt]['marker'], 
                    color=plotter[vel,ttt]['color'])

        ax2.set_ylabel('Similarity')#metric_dict[metric]['ylabel'])
        ax2.set_title('Lin Similarity Index')

        ax2.set_xlabel(param_dict[var]['xaxis'])
        ax2.set_xticks(locs,param_dict[var]['xticks'])

        legend_elements = [lines.Line2D([0],[0], color='w',
                marker=plotter[v,t]['marker'], 
                markersize=plotter[v,t]['size'], 
                markeredgecolor=plotter[v,t]['color'], 
                markerfacecolor=plotter[v,t]['color']) for v,t,s  in vel_ttt]

        legend_labels = [plotter[v,t]['label'] for v,t,s in vel_ttt]
            
        plt.legend()#legend_elements,legend_labels, ncol=3, loc='upper center')
        plt.grid()
        plt.savefig('similarity2-lin-{var}.eps'.format(var=var,metric=metric),
                    dpi=300,
                    bbox_inches="tight",
                    transparent=True,
                    format="pdf")
        plt.show()

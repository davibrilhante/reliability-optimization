#! /usr/bn/env python3
# -*- coding : utf8 -*-

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter
from matplotlib import use
from json import dump, load
from argparse import ArgumentParser

from plot_utils import load_result
from plot_utils import load_instance
from plot_utils import extract_metric
from plot_utils import calc_avg_blockage
from plot_utils import calc_gap
from plot_utils import calc_bsdist
from plot_utils import calc_diffs
from plot_utils import progress_bar
from plot_utils import calc_handoverfailures
from plot_utils import calc_pingpongs



parser = ArgumentParser()
parser.add_argument('-n','--new',action='store_true')
parser.add_argument('-S','--speed',action='store_false')
parser.add_argument('-s','--savefig',action='store_true')

args = parser.parse_args()

newplot_flag = args.new
normal = args.speed
savefig = args.savefig

metrics_dict = {
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
                'hofailure' : {
                            'title' : 'Average Number of handover failures',
                            'ylabel': 'failures',
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
                'gap' : {
                            'title' : 'Average Interruption Time',
                            'ylabel': 'Interruption [ms]',
                            'ypercent' : False 
                },
                'blockage' : {
                            'title' : 'Average Blockage Duration',
                            'ylabel': 'Blockage [ms]',
                            'ypercent' : False 
                },
                'blk_ratio' : {
                            'title' : 'Average Blockage Duration',
                            'ylabel': 'Blockage [\%]',
                            'ypercent' : True
                },
                'blk_episodes' : {
                            'title' : 'Average Blockage Episodes',
                            'ylabel': 'Blockage Episodes',
                            'ypercent' : False
                },
                'blk_duration' : {
                            'title' : 'Average Blockage Episode Duration',
                            'ylabel': 'Blockage [ms]',
                            'ypercent' : False
                },
                'bs_dist' : {
                            'title' : 'Average Distance to serving BS',
                            'ylabel': 'Average Distance',
                            'ypercent' : False 
                },
                'snr_diff' : {
                            'title' : 'Average SNR Difference Between\n Optimization and Baseline',
                            'ylabel': 'Average SNR Differente',
                            'ypercent': False 
                },
                'similarity':{
                            'ylabel':'Overlap Similarity',
                            'title': 'Overlap BS Association Similarity',
                            'ypercent':False
                }
            }

plotter = {
        (22,160):{
            'label':'v=30km/h,$\\tau$=160',
            'marker':'^',
            'size':10,
            'color':'blue'
            },
        (43,160):{
            'label':'v=60km/h,$\\tau$=160',
            'marker':'+',
            'size':10,
            'color':'green'
            },
        (43,80):{
            'label':'v=60km/h,$\\tau$=80',
            'marker':'s',
            'size':10,
            'color':'green'
            },
        (64,640):{
            'label':'v=90km/h,$\\tau$=160',
            'marker':'o',
            'size':10,
            'color':'red'
            },
        (64,480):{
            'label':'v=90km/h,$\\tau$=160',
            'marker':'o',
            'size':10,
            'color':'red'
            },
        (64,320):{
            'label':'v=90km/h,$\\tau$=160',
            'marker':'o',
            'size':10,
            'color':'red'
            },
        (64,160):{
            'label':'v=90km/h,$\\tau$=160',
            'marker':'o',
            'size':10,
            'color':'red'
            },
        (64,80):{
            'label':'v=90km/h,$\\tau$=80',
            'marker':'*',
            'size':10,
            'color':'red'
            },
        (64,40):{
            'label':'v=90km/h,$\\tau$=40',
            'marker':'v',
            'size':10,
            'color':'red'
            },
        0.001:{
            'label':'$\\lambda$=0.001',
            'marker':'o',
            'size':10,
            'color':'red'
            },
        0.003:{
            'label':'$\\lambda$=0.003',
            'marker':'+',
            'size':10,
            'color':'green'
            },
        0.005:{
            'label':'$\\lambda$=0.005',
            'marker':'^',
            'size':10,
            'color':'blue'
            },
        0.007:{
            'label':'$\\lambda$=0.007',
            'marker':'s',
            'size':10,
            'color':'orange'
            },
        0.009:{
            'label':'$\\lambda$=0.009',
            'marker':'*',
            'size':10,
            'color':'black'
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

def plot_from_dict(plot_dict, xticks, metrics, lines):
    for metric in metrics:
        for n, plot in enumerate(plot_dict.keys()):
            if plot == "diff":
                continue

            for line in lines:
                transposed = np.transpose(plot_dict[plot][line][metric])
                means = transposed[0]
                intervals = transposed[1]

                if n ==0:
                    plt.errorbar(xticks,means,yerr=intervals,
                            label=plotter[line]['label'],
                            color=plotter[line]['color'],
                            capsize=5.0,
                            marker=plotter[line]['marker'],
                            markersize=10,
                            markeredgecolor=plotter[line]['color'],
                            markerfacecolor='None',
                            linewidth=0.5
                            )
                else:
                    line, _, __ = plt.errorbar(xticks,means,yerr=intervals,
                            color=plotter[line]['color'],
                            capsize=5.0,
                            marker=plotter[line]['marker'],
                            markersize=10,
                            markeredgecolor=plotter[line]['color'],
                            markerfacecolor='None',
                            linewidth=0.5
                            )
                    line.set_dashes((15,10,15,10))

        plt.ylabel(metrics_dict[metric]['ylabel'])
        plt.title(metrics_dict[metric]['title'])
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    use('PS')

    xfmt = ScalarFormatter()
    xfmt.set_powerlimits((-1,3))

    plot_dict = {
            'opt':{},
            'base':{}
            }

    data_dict = {
            'opt':{},
            'base':{}
            }


    ''' 
    vel_params = [(22,1280,203647),
                (43,640,101823),
                (64,480,67882)]
    '''

    vel_params = [
                (22,160,203647),
                #(43,160,101823),
                (43,80,101823),
                #(64,640,67882),
                #(64,480,67882),
                #(64,320,67882),
                #(64,160,67882),
                #(64,80,67882),
                (64,40,67882)]

    x_ticks = [round(i*0.002 + 0.001,3) for i in range(5)]#np.transpose(x_params)[0]
    delay = [i*2+1 for i in range(5)]
    #x_params = [(round(i*0.002 + 0.001, 3),j*2 + 1) for i in range(5) for j in range(5)] #[(0.001,1),(0.003,1),(0.005,1),(0.007,1),(0.009,1)]
    #x_params = [(0.001,1),(0.003,1),(0.005,1),(0.007,1),(0.009,1)]
    x_params = [(x_ticks[i],j*2 + 1) for i in range(5) for j in range(5)] #[(0.001,1),(0.003,1),(0.005,1),(0.007,1),(0.009,1)]
    metric_diff = ['snr_diff', 'similarity']
    
    no_plot = metric_diff

    if newplot_flag:
        total = len(plot_dict.keys())*len(vel_params)*len(x_ticks)*len(delay)*60
        counter = 0
        for plot in plot_dict.keys():
            for vel, ttt, simtime in vel_params:
                try:
                    data_dict[plot][vel][ttt] = {}
                except KeyError:
                    data_dict[plot][vel] = {}
                    data_dict[plot][vel][ttt] = {}

                try:
                    plot_dict[plot][vel][ttt] = {}
                except KeyError:
                    plot_dict[plot][vel] = {}
                    plot_dict[plot][vel][ttt] = {}

                for var in x_ticks:
                    data_dict[plot][vel][ttt][var] = {}
                    #data_dict[plot][vel][var], _ = load_result('instances/no-interference/{plot}/{tau}/{vel}/{Lambda}/{cap}/{Del}/'.format(
                    for l in delay:
                        #print(var,l)

                        tmp_dict, _ = load_result('instances/no-interference/{plot}/{tau}/{vel}/{Lambda}/{cap}/{Del}/'.format(
                            plot=plot,
                            tau=ttt,
                            vel=vel,
                            Lambda=var,
                            cap='750e6',
                            Del=l)).load()

                        scenario = load_instance('instances/full-scenario/{vel}/{Lambda}/'.format(vel=vel, Lambda=var)).load()
                        #for instance in data_dict[plot][vel][var].keys():
                        for instance in tmp_dict.keys():
                            counter += 1
                            progress_bar(counter,total,head='{plot}, {tau:03d}, {vel}, {Lambda}, {Del}, {inst:02d}'.format(
                                plot=plot,
                                tau=ttt,
                                vel=vel,
                                Lambda=var,
                                Del=l,
                                inst=int(instance)))

                            key = int(instance) + (l//2)*int(len(tmp_dict.keys()))
                            #blk, blk_ratio, episodes, blk_duration = calc_avg_blockage(scenario[instance],data_dict[plot][vel][var][instance])
                            blk, blk_ratio, episodes, blk_duration = calc_avg_blockage(scenario[instance],tmp_dict[instance])

                            data_dict[plot][vel][ttt][var][key] = tmp_dict[instance].copy()

                            data_dict[plot][vel][ttt][var][key]['blockage'] = blk 
                            data_dict[plot][vel][ttt][var][key]['blk_ratio'] = blk_ratio
                            data_dict[plot][vel][ttt][var][key]['blk_episodes'] = episodes
                            data_dict[plot][vel][ttt][var][key]['blk_duration'] = blk_duration
                            data_dict[plot][vel][ttt][var][key]['gap'] = calc_gap(tmp_dict[instance],scenario[instance]['scenario']['simTime'],68) 
                            #data_dict[plot][vel][var[0]][key]['gap'] = calc_gap(data_dict[plot][vel][var][instance],scenario[instance]['scenario']['simTime'],68) 
                            data_dict[plot][vel][ttt][var][key]['bs_dist'] = calc_bsdist(scenario[instance],tmp_dict[instance]) 
                            #data_dict[plot][vel][var[0]][key]['bs_dist'] = calc_bsdist(scenario[instance],data_dict[plot][vel][var][instance]) 
                            data_dict[plot][vel][ttt][var][key]['hofailure'] = calc_handoverfailures(tmp_dict[instance],scenario[instance],ttt) 
                            data_dict[plot][vel][ttt][var][key]['pingpong'] = calc_pingpongs(tmp_dict[instance]) 
                        

                    for metric in metrics_dict.keys():
                        if metric not in no_plot:
                            try:
                                plot_dict[plot][vel][ttt][metric].append(extract_metric(data_dict[plot][vel][ttt][var]).errorplot(metric))
                            except KeyError:
                                plot_dict[plot][vel][ttt][metric] = []
                                plot_dict[plot][vel][ttt][metric].append(extract_metric(data_dict[plot][vel][ttt][var]).errorplot(metric))

        plot_dict['diff'] = {}
        snr_diff,similarity = calc_diffs(data_dict)
        plot_dict['diff']['snr_diff'] = snr_diff
        plot_dict['diff']['similarity'] = similarity

        print(plot_dict)

        with open('temp.json','w') as jsonfile:
            dump(plot_dict,jsonfile)

        lines = np.transpose(vel_params)[0]
        plot_from_dict(plot_dict,x_ticks,metrics_dict.keys(),lines)



    else:
        #plot_from_file()
        data = {}
        with open('temp.json') as jsonfile:
            data = load(jsonfile)

        if normal:
            #x_ticks = np.transpose(x_params)[0]
            x_ticks = [round(i*0.002 + 0.001,3) for i in range(5)]
            lines = vel_params #np.transpose(vel_params)
        else:
            #lines = np.transpose(x_params)[0]
            lines = [round(i*0.002 + 0.001,3) for i in range(5)]
            x_ticks = np.transpose(vel_params)


        for metric in metrics_dict:
            for n, plot in enumerate(data.keys()):
                print(plot, metric)
                if plot == 'diff':
                    for m, line in enumerate(lines):
                        if metric in metric_diff:
                            if normal:
                                plt.plot(x_ticks,data[plot][metric][str(line[0])][str(line[1])],
                                label=plotter[(line[0],line[1])]['label'],
                                color=plotter[(line[0],line[1])]['color'],
                                marker=plotter[(line[0],line[1])]['marker'],
                                markersize=10,
                                markeredgecolor=plotter[(line[0],line[1])]['color'],
                                markerfacecolor='None',
                                linewidth=0.5
                                )
                                prefix = 'block'
                                
                            else:
                                pass

                elif metric not in metric_diff:
                    for m, line in enumerate(lines):
                        if normal:
                            transposed = np.transpose(data[plot][str(line[0])][str(line[1])][metric])
                            means = transposed[0]
                            intervals = transposed[1]
                            prefix = 'block'

                        else:
                            means = [data[plot][str(vel)][str(line[1])][metric][m][0] for vel in x_ticks]
                            intervals = [data[plot][str(vel)][str(line[1])][metric][m][1] for vel in x_ticks]
                            prefix = 'vel'

                        if n ==0:
                            plt.errorbar(x_ticks,means,yerr=intervals,
                            label=plotter[(line[0],line[1])]['label'],
                            color=plotter[(line[0],line[1])]['color'],
                            capsize=5.0,
                            marker=plotter[(line[0],line[1])]['marker'],
                            markersize=10,
                            markeredgecolor=plotter[(line[0],line[1])]['color'],
                            markerfacecolor='None',
                            linewidth=0.5
                            )
                        else:
                            line, _, __ = plt.errorbar(x_ticks,means,yerr=intervals,
                            color=plotter[(line[0],line[1])]['color'],
                            capsize=5.0,
                            marker=plotter[(line[0],line[1])]['marker'],
                            markersize=10,
                            markeredgecolor=plotter[(line[0],line[1])]['color'],
                            markerfacecolor='None',
                            linewidth=0.5
                            )
                            line.set_dashes((15,10,15,10))

            if metrics_dict[metric]['ypercent']:
                plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


            handles, labels = plt.gca().get_legend_handles_labels()
            elements = [mlines.Line2D([0],[0],color='k',linestyle=plotter['opt']['linestyle']),
                        mlines.Line2D([0],[0],color='k',linestyle=plotter['base']['linestyle'])]


            plt.ylabel(metrics_dict[metric]['ylabel'])
            plt.title(metrics_dict[metric]['title'])
            plt.legend(handles+elements,labels+['opt','base'],)
            plt.grid()
            if savefig:
                plt.savefig('{prefix}-{metric}.eps'.format(prefix=prefix,metric=metric),
                        dpi=300,
                        bbox_inches="tight",
                        transparent=True,
                        format="pdf")
            plt.show()
            plt.close()
            plt.clf()
            plt.cla()

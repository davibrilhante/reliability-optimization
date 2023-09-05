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


parser = ArgumentParser()
parser.add_argument('-s','--savefig',action='store_true')

args = parser.parse_args()

savefig = args.savefig

metrics_dict = {
                'pingpong' : {
                            'title' : 'Average Ping-Pong Handover Rate',
                            'ylabel': 'Ping-pongs [\%]',
                            'ypercent' : True 
                    },
                'blk_duration' : {
                            'title' : 'Average Blockage Episode Duration',
                            'ylabel': 'Blockage [ms]',
                            'ypercent' : False
                },
                'blk_episodes' : {
                            'title' : 'Average Number of Blockage Episode',
                            'ylabel': '\# Blockage Episodes',
                            'ypercent' : False
                },
                'blk_ratio' : {
                            'title' : 'Average Percentual Time Blocked',
                            'ylabel': 'Percentual Blockage [\%]',
                            'ypercent' : True
                },
                'partDelay':{
                            'title' : 'Average Packets delivered before delay expiration ',
                            'ylabel': 'Delivery Rate [\%]',
                            'ypercent' : True
                },
                'capacity' : {
                            'title' : 'Average Shannon Capacity',
                            'ylabel': 'Capacity [bps]',
                            'ypercent' : False 
                },
                'bs_dist' : {
                            'title' : 'Average Distant to serving BS',
                            'ylabel': 'Average Distance',
                            'ypercent' : False 
                },
                'handoverRate' : {
                            'title' : 'Average Handover Rate',
                            'ylabel': 'Handovers per second',
                            'ypercent' : False 
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
        'opt':{
            'linestyle':'-',
            'linewidth':3
            },
        'base':{
            'linestyle':'--',
            'linewidth':3
            }
        }

if __name__ == "__main__":
    data = {}
    with open('temp.json') as jsonfile:
        data = load(jsonfile)

    metric1 = 'handoverRate'
    metric2 = 'pingpong'
    
    metric1 = 'capacity'
    metric2 = 'partDelay'



    metric1 = 'blk_duration'
    metric2 = 'blk_episodes'


    width = 17.0 
    ratio = 0.5 
    height = width*ratio

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=26)
    plt.rc('figure',figsize=[width,height])
    #use('ps')
    use('pgf')
    plt.rc('pgf',texsystem='pdflatex')
    plt.rc('pgf',rcfonts=False)
    #plt.rc('axes.formatter',use_mathtext=True)
    plt.rc('figure',dpi=300)

    fig, (ax1, ax2) = plt.subplots(1,2)

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

    x_ticks = [round(i*0.002 + 0.001,3) for i in range(5)]
    lines = vel_params #np.transpose(vel_params)

    for plot in ['opt', 'base']:
        for m, line in enumerate(lines):

            transposed = np.transpose(data[plot][str(line[0])][str(line[1])][metric1])
            means = transposed[0]
            intervals = transposed[1]
            prefix = 'block'

            if plot == 'opt':
                ax1.errorbar(x_ticks,means,yerr=intervals,
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
                Line, _, __ = ax1.errorbar(x_ticks,means,yerr=intervals,
                color=plotter[(line[0],line[1])]['color'],
                capsize=5.0,
                marker=plotter[(line[0],line[1])]['marker'],
                markersize=10,
                markeredgecolor=plotter[(line[0],line[1])]['color'],
                markerfacecolor='None',
                linewidth=0.5
                )
                Line.set_dashes((15,10,15,10))

            transposed = np.transpose(data[plot][str(line[0])][str(line[1])][metric2])
            means = transposed[0]
            intervals = transposed[1]
            prefix = 'block'

            if plot == 'opt':
                ax2.errorbar(x_ticks,means,yerr=intervals,
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
                Line, _, __ = ax2.errorbar(x_ticks,means,yerr=intervals,
                color=plotter[(line[0],line[1])]['color'],
                capsize=5.0,
                marker=plotter[(line[0],line[1])]['marker'],
                markersize=10,
                markeredgecolor=plotter[(line[0],line[1])]['color'],
                markerfacecolor='None',
                linewidth=0.5
                )
                Line.set_dashes((15,10,15,10))

    #if metrics_dict[metric]['ypercent']:
    #    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


    handles, labels = ax1.get_legend_handles_labels()
    elements = [mlines.Line2D([0],[0],color='k',linestyle=plotter['opt']['linestyle']),
                mlines.Line2D([0],[0],color='k',linestyle=plotter['base']['linestyle'])]


    ax1.set_ylabel(metrics_dict[metric1]['ylabel'])
    ax1.set_xlabel('Blockage Density $\lambda$ [objects/$m^2$]')
    ax1.set_xticks(x_ticks)
    if metrics_dict[metric1]['ypercent']:
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, is_latex=True))
    ax1.grid()

    ax2.set_ylabel(metrics_dict[metric2]['ylabel'])
    ax2.set_xlabel('Blockage Density $\lambda$ [objects/$m^2$]')
    ax2.set_xticks(x_ticks)
    if metrics_dict[metric2]['ypercent']:
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, is_latex=True))
    ax2.grid()

    loc_hor = 1.20
    loc_vert = 0.8

    plt.legend([handles[0],elements[0],handles[1],elements[1],handles[2]],
            [labels[0],'Optimisation',labels[1],'Baseline',labels[2]],
            bbox_to_anchor=(loc_vert, loc_hor), 
            loc='upper right', 
            ncol=3, 
            columnspacing=0.5,
            borderaxespad=0.)
    plt.subplots_adjust(wspace=0.4)

    if savefig:
        plt.savefig('{prefix}-{m1}-{m2}.pgf'.format(prefix=prefix,m1=metric1,m2=metric2),
                dpi=300,
                bbox_inches="tight",
                transparent=True,
                format="pgf")
    #plt.show()
    plt.close()
    plt.clf()
    plt.cla()

    fig, (ax1, ax2) = plt.subplots(1,2)

    metric1 = 'blk_duration'
    metric1 = 'capacity'
    metric2 = 'partDelay'

    metric1 = 'blk_ratio'
    metric2 = 'bs_dist'

    for plot in ['opt', 'base']:
        for m, line in enumerate(lines):

            transposed = np.transpose(data[plot][str(line[0])][str(line[1])][metric1])
            means = transposed[0]
            intervals = transposed[1]
            prefix = 'block'

            if plot == 'opt':
                ax1.errorbar(x_ticks,means,yerr=intervals,
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
                Line, _, __ = ax1.errorbar(x_ticks,means,yerr=intervals,
                color=plotter[(line[0],line[1])]['color'],
                capsize=5.0,
                marker=plotter[(line[0],line[1])]['marker'],
                markersize=10,
                markeredgecolor=plotter[(line[0],line[1])]['color'],
                markerfacecolor='None',
                linewidth=0.5
                )
                Line.set_dashes((15,10,15,10))


    ax1.set_ylabel(metrics_dict[metric1]['ylabel'])
    ax1.set_xlabel('Blockage Density $\lambda$ [objects/$m^2$]')
    ax1.set_xticks(x_ticks)
    if metrics_dict[metric1]['ypercent']:
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, is_latex=True))
    ax1.grid()

    x_ticks = [i*2 + 1 for i in range(5)]
    for plot in ['opt', 'base']:
        for m, line in enumerate(lines):
            transposed = np.transpose(data[plot][str(line[0])][str(line[1])][metric2])
            means = transposed[0]
            intervals = transposed[1]
            prefix = 'block'

            if plot == 'opt':
                ax2.errorbar(x_ticks,means,yerr=intervals,
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
                Line, _, __ = ax2.errorbar(x_ticks,means,yerr=intervals,
                color=plotter[(line[0],line[1])]['color'],
                capsize=5.0,
                marker=plotter[(line[0],line[1])]['marker'],
                markersize=10,
                markeredgecolor=plotter[(line[0],line[1])]['color'],
                markerfacecolor='None',
                linewidth=0.5
                )
                Line.set_dashes((15,10,15,10))

    #if metrics_dict[metric]['ypercent']:
    #    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


    handles, labels = ax1.get_legend_handles_labels()
    elements = [mlines.Line2D([0],[0],color='k',linestyle=plotter['opt']['linestyle']),
                mlines.Line2D([0],[0],color='k',linestyle=plotter['base']['linestyle'])]



    ax2.set_ylabel(metrics_dict[metric2]['ylabel'])
    #ax2.set_xlabel('Delay Tolerance [ms]')
    ax2.set_xlabel('Blockage Density $\lambda$ [objects/$m^2$]')
    ax2.set_xticks(x_ticks)
    if metrics_dict[metric2]['ypercent']:
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, is_latex=True))
    ax2.grid()

    plt.legend([handles[0],elements[0],handles[1],elements[1],handles[2]],
            [labels[0],'Optimisation',labels[1],'Baseline',labels[2]],
            bbox_to_anchor=(loc_vert, loc_hor), 
            loc='upper right', 
            ncol=3, 
            columnspacing=0.5,
            borderaxespad=0.)
    plt.subplots_adjust(wspace=0.4)

    if savefig:
        plt.savefig('{prefix}-{m1}-{m2}.pgf'.format(prefix=prefix,m1=metric1,m2=metric2),
                dpi=300,
                bbox_inches="tight",
                transparent=True,
                format="pgf")


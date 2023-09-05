#! /usr/bin/env python3
# -*- coding : utf8 -*-

'''
Este código plota o máximo de oportunidades de handover por densidade de 
bloqueio e velocidade
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mtick                                               
from matplotlib import use                                                      
from multiprocessing import Pool
import json

from plot_utils import load_result
from plot_utils import load_instance
from optimization import snr_processing

meta = {
        (22,160):{
            'label':'v=30 km/h',
            'marker':'^',
            'size':10,
            'color':'blue'
            },
        (43,80):{
            'label':'v=60 km/h',
            'marker':'s',
            'size':10,
            'color':'green'
            },
        (64,40):{
            'label':'v= 90km/h',
            'marker':'v',
            'size':10,
            'color':'red'
            },
        }

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

    width = 17.0 
    ratio = 0.5 
    height = width*ratio

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=26)
    #plt.rc('figure',figsize=[width,height])
    #use('ps')
    #use('pgf')
    plt.rc('pgf',texsystem='pdflatex')
    plt.rc('pgf',rcfonts=False)
    #plt.rc('axes.formatter',use_mathtext=True)
    #plt.rc('figure',dpi=300)

    ho_opportunities = {}

    offset = 3
    hysteresis = 0

    vel_params =[(22,160,203647),
            (43,80,101823),
            (64,40,67882)]

    Lambda = [round(i*0.002 + 0.001,3) for i in range(5)]
    delay = [i*2 + 1 for i in range(5)]
    seeds = list(range(60))
    np.random.shuffle(seeds)

    outdict = {}

    for vel, ttt, tsim in vel_params:
        outdict[vel] = {}
        outdict[vel]['mean'] = []
        outdict[vel]['ci'] = []
        outdict[vel]['rsrp'] = []

        for blockdensity in Lambda:
            print(vel, blockdensity)
            diff = []
            rsrp = []
            for seed in seeds[:20]:
                filename = 'instances/hodata/{v}/{t}/{l}/{s}'.format(
                        v=vel,
                        t=ttt,
                        l=blockdensity,
                        s=seed
                        )
                with open(filename) as infile:
                    data = json.load(infile)

                rsrp += data['sliced']
                counter = 0
                sample = 0
                for n in data['ntargets']:
                    for j in data['diff'][counter:counter+n]:
                        if j > 3:
                            sample += 1
                            break
                    counter += n
                diff.append(sample/tsim)

            outdict[vel]['mean'].append(np.mean(diff))
            outdict[vel]['ci'].append(np.std(diff)*1.96/np.sqrt(len(diff)))
            outdict[vel]['rsrp'].append(rsrp)
        
        print(outdict[vel]['mean'])
        plt.errorbar(range(5), outdict[vel]['mean'],
                yerr=outdict[vel]['ci'],
                label=meta[(vel,ttt)]['label'],
                color=meta[(vel,ttt)]['color'],
                marker=meta[(vel,ttt)]['marker'],
                markersize=10,
                markeredgecolor=meta[(vel,ttt)]['color'],
                markerfacecolor='None',
                capsize=5.0,
                linewidth=0.5)
    plt.xticks(range(5),Lambda)
    plt.xlabel('Blockage Density $\lambda$\n')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, is_latex=True))
    plt.ylabel('$\%$ of timeslots')
    plt.title('$\%$ of timeslots that \n TBS RSRP $>$ SBS RSRP + offset')
    plt.legend()
    plt.grid()
    '''
    plt.savefig('timeslots-diff.png',
                dpi=300,
                bbox_inches="tight",
                transparent=True)

    '''
    plt.show()

    for vel, ttt, tsim in vel_params:
        medianprops = dict(linestyle='-', linewidth=2.5, color='orange')
        meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')

        elements = [mlines.Line2D([0],[0],color='orange',linestyle='-',linewidth=2.5),
                    mlines.Line2D([0],[0],color='purple',linestyle='--',linewidth=2.5)]

        plt.boxplot(outdict[vel]['rsrp'], notch=False,showfliers=False, showmeans=True, 
                meanline=True, medianprops=medianprops, meanprops=meanlineprops)
        plt.xticks(range(1,6),Lambda)
        plt.xlabel('Blockage Density $\lambda$\n')
        plt.ylabel('RSRP [dBm]')
        plt.ylim(-100,-20)
        plt.title('RSRP variation for 350m range for each BS\n')
        plt.legend(elements,['Median','Mean'],
                bbox_to_anchor=(0.5,1.06),
                ncol=2,
                loc='center',
                borderpad=.1,
                labelspacing=1)
        plt.grid()
        plt.savefig('rsrp-scenario-{v}-350.png'.format(v=vel),
                dpi=300,
                bbox_inches="tight",
                transparent=True)
        plt.show()


    '''
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
            boxplot = []
            lineplot = []
            others['source'].append([])
            others['target'].append([])
            others['neighbors'].append([])
            others['leaving'].append([])
                #boxplot += data['rsrp'] 
                boxplot += data['diff'] 
                lineplot += data['ntargets']
                others['source'][-1] += data['source']
                others['target'][-1] += data['target']
                others['neighbors'][-1] += data['neighbors']
                others['leaving'][-1] += data['leaving']

            #plt.boxplot(boxplot)
            plotter.append(boxplot)
            biplotter.append(lineplot)



    with open('ho-rsrp','w') as outfile:
        json.dump(ho_opportunities,outfile)


    print(len(others['target'][0]), len(others['target'][1]),
            len(others['target'][2]),len(others['target'][3]),
            len(others['target'][4]))

    medianprops = dict(linestyle='-', linewidth=2.5, color='orange')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')

    elements = [mlines.Line2D([0],[0],color='orange',linestyle='-',linewidth=2.5),
                mlines.Line2D([0],[0],color='purple',linestyle='--',linewidth=2.5)]



    plt.boxplot(plotter, notch=False,showfliers=False, showmeans=True, 
            meanline=True, medianprops=medianprops, meanprops=meanlineprops)
    plt.ylabel('Difference RSRP [dB]')
    plt.xticks(range(1,6),Lambda)
    plt.xlabel('Blockage Density $\lambda$')
    plt.title('SBS-TBS RSRP Difference Boxplot for 30 Km/h\n\n')
    plt.legend(elements,['Median','Mean'],
            bbox_to_anchor=(0.5,1.1),
            ncol=2,
            loc='center')
    plt.grid()
    plt.show()

    plt.boxplot(biplotter, notch=False, showfliers=False, showmeans=True,
            meanline=True, medianprops=medianprops, meanprops=meanlineprops)
    plt.ylabel('\# Target BS')
    plt.xticks(range(1,6),Lambda)
    plt.xlabel('Blockage Density $\lambda$')
    plt.title('Number of Target BS for 30Km/h')
    plt.legend(elements,['Median','Mean'])
    plt.grid()
    plt.savefig('number-targets.png',
                dpi=300,
                bbox_inches="tight",
                transparent=True)
    plt.show()

    b = np.arange(1,6)
    a = np.ones(5)*0.5
    x = np.concatenate((b-a,b,b+a))
    #y = np.concatenate((others['source'],others['target']),axis=1)
    y = [j for i in zip(others['leaving'],others['target']) for j in i]
    #y = [j for i in zip(others['source'],others['neighbors']) for j in i]


    plt.boxplot(y,showfliers=False, showmeans=True, meanline=True, 
            medianprops=medianprops, meanprops=meanlineprops)#, data=others['source'])
#    plt.boxplot(others['source'],showfliers=False, showmeans=True, meanline=True, 
#            medianprops=medianprops, meanprops=meanlineprops)#, data=others['source'])
#    plt.xticks(b,Lambda)
#    plt.show()
#    plt.boxplot(others['neighbors'],showfliers=False, showmeans=True, meanline=True,
#            medianprops=medianprops, meanprops=meanlineprops)
#    plt.xticks(b,Lambda)
#    plt.show()
#    plt.boxplot(others['target'], showfliers=False, showmeans=True, meanline=True,
#            medianprops=medianprops, meanprops=meanlineprops)
#    plt.xticks(b,Lambda)
#    plt.show()
    #plt.ylabel('\# Target BS')
    plt.xticks(b*2-np.ones(5)*0.5,Lambda)
    plt.xlabel('Blockage Density $\lambda$')
    plt.ylabel('RSRP [dB]')
    plt.title('Source and Target BS RSRP for 30Km/h\n\n')
    plt.legend(elements,['Median','Mean'],
            bbox_to_anchor=(0.5,1.1),
            ncol=2,
            loc='center')
    plt.grid()
    plt.savefig('rsrp-source-targets.png',
                dpi=300,
                bbox_inches="tight",
                transparent=True)
    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot([np.median(i) for i in others['leaving']],label='source')
    ax1.plot([np.median(i) for i in others['target']],label='target')
    ax1.set_ylabel('RSRP [dB]')


    ax2 = ax1.twinx()
    ax2.plot([np.median(i) for i in plotter],label='diff',color='g')
    ax2.set_ylabel('RSRP difference [dB]',color='g')

    plt.xticks(range(5),Lambda)
    plt.xlabel('Blockage Density $\lambda$\n')
    plt.grid()
    plt.legend()
    plt.show()

            '''

if __name__ == '__main__':
    main()

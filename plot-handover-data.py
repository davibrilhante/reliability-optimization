import numpy as np
from scipy.stats import norm
from json import load, dump
import argparse
from os import listdir, path
from decompressor import decompressor
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from matplotlib import use



from optimization import todb, tolin, load_inputFile, calc_recv, beta_processing, snr_processing


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

                jsonfile.close()



        return data, self.get_metrics(data['0'])


parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')

args = parser.parse_args()



if __name__ == "__main__":
    labels = ['{0:.3f}'.format(0.002*i + 0.001) for i in range(5)]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=13)
    use('PS')


    line_dict = {
        'opt':{
            'label':'optimal',
            'color':'blue',
            'marker':'^'
            },
        'bas':{
            'label':'baseline',
            'color':'orange',
            'marker':'o'
            },
        'block': {
            'title': 'Average period the UE had LoS blocked',
            'ylabel':'Time under Blockage',
            'linestyle': '-',
            'multiplot': False,
            'relative': True,
            'figname':'average-block-period.eps'
            },
        'rsrp_min': {
            'title': 'Average Maximum and Minimum RSRP',
            'ylabel':'RSRP [dBm]',
            'linestyle': '-',
            'multiplot': True,
            'relative': False,
            'figname':'average-max-min-rsrp.eps'
            },
        'rsrp_max': {
            'title': 'Average Maximum and Minimum RSRP',
            'ylabel':'RSRP [dBm]',
            'linestyle': '--',
            'multiplot': False,
            'relative': False,
            'figname':'average-max-min-rsrp.eps'
            },
        'rsrp_delta': {
            'title': 'Average RSRP Variation',
            'ylabel':'$\Delta$ RSRP [dB]',
            'linestyle': '-',
            'multiplot': False,
            'relative': False,
            'figname':'average-delta-rsrp.eps'
            },
        'anchor_average': {
            'title': 'Average Anchor BS RSRP',
            'ylabel':'RSRP [dB]',
            'linestyle': '-',
            'multiplot': True,
            'relative': False,
            'figname':'average-anchor-rsrp.eps'
            },
        'nonanchor_average': {
            'title': 'Average Non-Anchor BS RSRP',
            'ylabel':'RSRP [dB]',
            'linestyle': '--',
            'multiplot': False,
            'relative': False,
            'figname':'average-nonanchor-rsrp.eps'
            },
        'anchor_diff': {
            'title': 'Average RSRP Difference between anchor\nand Non Anchor BS',
            'ylabel':'Difference [dB]',
            'linestyle': '-',
            'multiplot': False,
            'relative': False,
            'figname':'average-diff-anchor-rsrp.eps'
            },
        'anchor_interval': {
            'title': 'Average Association Interval with Anchor BS',
            'ylabel':'Average association interval',
            'linestyle': '-',
            'multiplot': False,
            'relative': True,
            'figname':'average-anchor-interval.eps'
            },
        'nonanchor_interval': {
            'title': 'Average Association Interval with Non-Anchor BS',
            'ylabel':'Average association interval',
            'linestyle': '-',
            'multiplot': False,
            'relative': True,
            'figname':'average-nonanchor-interval.eps'
            },
        'rsrp_decay': {
            'title': 'Average timeslots the BS is outperformed',
            'ylabel':'Outperformed Slots',
            'linestyle': '-',
            'multiplot': False,
            'relative': True,
            'figname':'average-rsrp-decay.eps'
            },
        'rsrp_transition': {
            'title': 'Average Number of Blockage Episodes\n(blockage duration ~ TTT)',
            'ylabel':'Number of Blockage Episodes',
            'linestyle': '-',
            'multiplot': False,
            'relative': False,
            'figname':'average-block-episodes-2.eps'
            },
        'interval_average': {
            'title': 'Average Association Interval',
            'ylabel':'Average association interval',
            'linestyle': '-',
            'multiplot': False,
            'relative': True,
            'figname':'average-association-interval.eps'
            }
        }

    results = {
        'opt':{},
        'bas':{}
        }

    offset = 3
    hysteresis = 0
    tau = 640

    with open('no-blockage-test','r') as jsonfile:
        anchor = load(jsonfile) 
    jsonfile.close()

    for data_type in results.keys():
        results[data_type]['block'] = {}
        results[data_type]['rsrp_max'] = {}
        results[data_type]['rsrp_min'] = {}
        results[data_type]['rsrp_delta'] = {} 
        results[data_type]['interval_average'] = {}
        results[data_type]['anchor_average'] = {}
        results[data_type]['anchor_interval'] = {}
        results[data_type]['anchor_diff'] = {}
        results[data_type]['nonanchor_average'] = {}
        results[data_type]['nonanchor_interval'] = {}
        results[data_type]['rsrp_decay'] = {}
        results[data_type]['rsrp_transition'] = {}

        for index, Lambda in enumerate(labels):
            results[data_type]['block'][Lambda] = [] 
            results[data_type]['rsrp_max'][Lambda] = [] 
            results[data_type]['rsrp_min'][Lambda] = [] 
            results[data_type]['rsrp_delta'][Lambda] = [] 
            results[data_type]['interval_average'][Lambda] = [] 
            results[data_type]['anchor_average'][Lambda] = [] 
            results[data_type]['anchor_interval'][Lambda] = []
            results[data_type]['nonanchor_average'][Lambda] = [] 
            results[data_type]['anchor_diff'][Lambda] = [] 
            results[data_type]['nonanchor_interval'][Lambda] = []
            results[data_type]['rsrp_decay'][Lambda] = []
            results[data_type]['rsrp_transition'][Lambda] = []

            # Load simulation results to data
            data, metrics = load_result(f'instances/compare/{data_type}/22/{Lambda}/').load()

            rootdir = 'instances/'
            for filename in listdir(rootdir):
                fullpath = rootdir+filename
                name = filename.split('-')[0:2]

                if path.isfile(fullpath) and name == ['handover','dataset']:
                    l_index = filename.split('-')[2]
                    s_index = filename.split('-')[3]

                    seeds_allowed = ['1', '2', '3', '4', '5', '6']

                    if l_index == str(index) and s_index in seeds_allowed:
                        fullpath = rootdir+filename
                        print(fullpath)

                        # load instance data to inst_data
                        with open(fullpath, 'r') as jsonfile:
                            inst_data = load(jsonfile)

                        jsonfile.close()
                        for seed in inst_data[Lambda].keys():
                            ndecays = 0
                            #rsrp_transition = 0
                            rsrp_accumulator = []
                            interval = []
                            for assoc in data[seed]['association']:
                                servingbs = assoc[0]
                                init = assoc[1]
                                try:
                                    end = assoc[2]
                                except IndexError:
                                    continue

                                for t in range(init, end):
                                    '''
                                    if t > 0 and (inst_data[Lambda][seed]['RSRP'][servingbs][0][t] < 
                                                inst_data[Lambda][seed]['RSRP'][servingbs][0][t-1] - tolin(10)):
                                        rsrp_transition += 1
                                    '''


                                    for m in range(len(inst_data[Lambda][seed]['RSRP'])):
                                        if (inst_data[Lambda][seed]['RSRP'][servingbs][0][t] <
                                                inst_data[Lambda][seed]['RSRP'][m][0][t]):
                                            ndecays += 1
                                            break


                                        
                                rsrp_accumulator += inst_data[Lambda][seed]['RSRP'][servingbs][0][init:end]
                                interval.append(end-init)

                            results[data_type]['rsrp_max'][Lambda].append(max(rsrp_accumulator)) 
                            results[data_type]['rsrp_min'][Lambda].append(min(rsrp_accumulator))
                            results[data_type]['rsrp_delta'][Lambda].append(
                                    results[data_type]['rsrp_max'][Lambda][-1] 
                                    - results[data_type]['rsrp_min'][Lambda][-1])
                            results[data_type]['interval_average'][Lambda].append(np.mean(interval))
                            results[data_type]['rsrp_decay'][Lambda].append(ndecays)
                            #results[data_type]['rsrp_transition'][Lambda].append(rsrp_transition)

                            temp = []
                            temp2 = []
                            diff = []
                            nonanchor_interval = 0
                            anchor_interval = 0

                            for anchorassoc in anchor['association']:
                                anchorbs = anchorassoc[0]
                                anchor_init = anchorassoc[1]
                                anchor_end = anchorassoc[2]

                                temp += inst_data[Lambda][seed]['RSRP'][anchorbs][0][anchor_init:anchor_end]

                                s1 = set(range(anchor_init,anchor_end))

                                for assoc in data[seed]['association']:
                                    servingbs = assoc[0]
                                    init = assoc[1]
                                    try:
                                        end = assoc[2]
                                    except IndexError:
                                        continue

                                    s2 = set(range(init,end))
                                    intersection = s1.intersection(s2)
                                    if intersection:
                                        if servingbs == anchorbs:
                                            anchor_interval += len(intersection)
                                        else:
                                            temp2 += inst_data[Lambda][seed]['RSRP'][servingbs][0][init:min(end, anchor_end)]
                                            nonanchor_interval += len(intersection)

                                        for t in intersection:
                                            diff.append(inst_data[Lambda][seed]['RSRP'][servingbs][0][t] - 
                                                    inst_data[Lambda][seed]['RSRP'][anchorbs][0][t])

                                    if init > anchor_end:
                                        break

                            results[data_type]['anchor_interval'][Lambda].append(anchor_interval)
                            results[data_type]['anchor_average'][Lambda].append(np.mean(temp))
                            results[data_type]['nonanchor_average'][Lambda].append(np.mean(temp2))
                            results[data_type]['anchor_diff'][Lambda].append(np.mean(diff))
                            results[data_type]['nonanchor_interval'][Lambda].append(nonanchor_interval)



                        

            rootdir = f'instances/full-scenario/22/{Lambda}/'
            for filename in listdir(rootdir):
                fullpath = rootdir+filename

                if path.isfile(fullpath):
                    with open(fullpath, 'r') as jsonfile:
                        rawinstance = load(jsonfile)
                        decompressor(rawinstance)
                    
                    jsonfile.close()
                    
                    blocked_time = 0
                    blockage_eps = 0
                    unblocked = 0
                    blocked = 0
                    for assoc in data[filename]['association']:
                        servingbs = assoc[0]
                        init = assoc[1]
                        try:
                            end = assoc[2]
                        except IndexError:
                            continue

                        ### There is a bug here!
                        blocked_time += sum(rawinstance['blockage'][servingbs][0][init:end])

                        for t in range(init,end):
                            if t > 0:
                                if rawinstance['blockage'][servingbs][0][t] == 0:
                                    unblocked += 1                                    
                                    blocked = 0

                                elif rawinstance['blockage'][servingbs][0][t] == 1:
                                    blocked += 1

                                    if unblocked > 640 and blocked > 640:
                                        blockage_eps += 1
                                        unblocked = 0

                    results[data_type]['block'][Lambda].append(len(rawinstance['blockage'][0][0]) - 
                                                                blocked_time)
                    results[data_type]['rsrp_transition'][Lambda].append(blockage_eps)


            for m in results[data_type].keys(): 
                factor = 1
                if line_dict[m]['relative']:
                    factor = 1/203647

                mean = factor*np.mean(results[data_type][m][Lambda])
                sigma = factor*np.std(results[data_type][m][Lambda])
                n = len(results[data_type][m][Lambda])
                alpha = 0.99
                ci = alpha + (1-alpha)/2
                conf_interval = norm.ppf(ci)*((sigma/(n**0.5))) 

                results[data_type][m][Lambda] = [mean,conf_interval]


    for m in results[data_type].keys():
        if m in ['rsrp_max', 'nonanchor_average']:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Blockage Density $\lambda$ [objects/$m^2$]')
        ax.set_ylabel(line_dict[m]['ylabel'])
        ax.set_title(line_dict[m]['title'])

        if m in ['rsrp_min']:
            ax.set_ylim(-100,-20)

        elif m in ['rsrp_delta']:
            ax.set_ylim(50,70)

        elif m in ['anchor_average']:
            ax.set_ylim(-70,-40)

        if line_dict[m]['relative']:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        for data_type in results.keys():
            values = list(results[data_type][m].values())
            values = np.transpose(values)

            ax.errorbar(results[data_type][m].keys(),values[0], 
                        yerr=values[1],
                        label=line_dict[data_type]['label'],
                        color=line_dict[data_type]['color'],
                        marker=line_dict[data_type]['marker'],
                        linestyle=line_dict[m]['linestyle'])

            if line_dict[m]['multiplot']:
                if m == 'rsrp_min':
                    auxiliar = 'rsrp_max'

                elif m == 'anchor_average':
                    auxiliar = 'nonanchor_average'

                values = list(results[data_type][auxiliar].values())
                values = np.transpose(values)

                ax.errorbar(results[data_type][auxiliar].keys(),values[0],
                        yerr=values[1],
                        #label=line_dict[data_type]['label'],
                        color=line_dict[data_type]['color'],
                        marker=line_dict[data_type]['marker'],
                        linestyle=line_dict[auxiliar]['linestyle'])



        
        plt.legend()
        #plt.tight_layout()
        plt.grid()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()
        plt.savefig(line_dict[m]['figname'], 
                    dpi=300, 
                    bbox_inches='tight', 
                    transparent=True, 
                    #pad_inches=0.1, 
                    format='pdf')

        plt.close(fig)

    print('min diff:')
    print(results['opt']['rsrp_min']['0.001'][0] - results['opt']['rsrp_min']['0.009'][0])
    print('max diff:')
    print(results['opt']['rsrp_max']['0.001'][0] - results['opt']['rsrp_max']['0.009'][0])

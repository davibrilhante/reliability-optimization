#! /usr/bn/env python3
# -*- coding : utf8 -*-

import numpy as np

from os import listdir, path
from json import load
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm
from matplotlib import use


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




if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=13)
    use('PS')

    plot_opt = {}
    plot_base = {}

    labels = ['{0:.3f}'.format(0.002*i + 0.001) for i in range(5)]

    plt_OOS_opt = []
    plt_OOS_base = []

    for Lambda in labels:
        opt, opt_metrics = load_result('instances/compare/opt/22/{Lambda}/'.format(Lambda=Lambda)).load()
        base, base_metrics = load_result('instances/compare/bas/22/{Lambda}/'.format(Lambda=Lambda)).load()

        for instance in opt.keys():
            opt[instance]['gap'], opt[instance]['HIT'], opt[instance]['outofsync'] = calc_gap(opt[instance])

        for instance in base.keys():
            try:
                base[instance]['HIT'] = sum(base[instance]['log']['HIT'])
            except KeyError:
                base[instance]['HIT'] = 0

            if type(base[instance]['outofsync']) == type([]):
                temp = 0
                for OOS_interval in base[instance]['outofsync']:
                    try:
                        temp += OOS_interval[1] - OOS_interval[0]
                    except IndexError:
                        temp += 1

                base[instance]['outofsync'] = base[instance]['gap']-base[instance]['HIT']



        plt_OOS_opt.append(extract_metric(opt).mean('outofsync'))
        plt_OOS_base.append(extract_metric(base).mean('outofsync'))

        for metric in base_metrics:
            if metric in opt_metrics and metric != 'association':
                try:
                    plot_opt[metric].append(extract_metric(opt).errorplot(metric))
                    plot_base[metric].append(extract_metric(base).errorplot(metric))


                except KeyError:
                    plot_opt[metric] = []
                    plot_base[metric] = []

                    plot_opt[metric].append(extract_metric(opt).errorplot(metric))
                    plot_base[metric].append(extract_metric(base).errorplot(metric))



    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=13)

    for metric in base_metrics:
        if metric in opt_metrics and metric not in ['association', 'HIT','outofsync']:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            plot_array_opt = np.transpose(plot_opt[metric])
            plot_array_base = np.transpose(plot_base[metric])

            ax.errorbar(labels, plot_array_opt[0], yerr=plot_array_opt[1], label='opt', marker='^')
            ax.errorbar(labels, plot_array_base[0], yerr=plot_array_base[1], label='base', marker='o')

            ax.set_ylabel(metric_dict[metric]['ylabel'])
            ax.set_title(metric_dict[metric]['title'])

            ax.set_xlabel('Blockage Density $\lambda$ [objects/$m^2$]')

            if metric_dict[metric]['ypercent']:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            if metric == 'gap':

                x_fill = labels + labels[len(labels) -1::-1]
                y_fill = plt_OOS_base + [0 for i in labels]

                ax.plot(plt_OOS_base, 'orange', ls='--', label='base ou of sync')
                ax.fill(x_fill, y_fill, 'orange', alpha = 0.5)

                x_fill = labels + labels[len(labels) -1::-1]
                y_fill = plt_OOS_opt + [0 for i in labels]

                ax.plot(plt_OOS_opt, 'b', ls='--', label='opt out of sync')
                ax.fill(x_fill, y_fill, 'b', alpha = 0.5)

                

            #plt.tight_layout()
            plt.legend()
            plt.grid()
            plt.savefig('comparison-opt-base-'+metric+'.eps',
                        dpi=300,
                        bbox_inches="tight",
                        transparent=True,
                        format="pdf")
            plt.show()
                    

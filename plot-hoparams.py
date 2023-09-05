import numpy as np

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter
from matplotlib import use
from json import dump, load
from argparse import ArgumentParser
from ast import literal_eval

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

args = parser.parse_args()

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


if __name__ == "__main__":
    vel_params = [(22,160,203647),
            (43,80,101823),
            (64,40,67882)]

    Lambda = [round(i*0.004 + 0.001,3) for i in range(3)]
    offset = range(1,5)
    hysteresis = [0, 0.25]
    seeds = range(60)

    total = len(vel_params)*len(Lambda)*len(offset)*len(hysteresis)*60
    no_plot = ['association','log']

    if args.new:

        out_dict = {}
        plot_dict = {}

        for vel, ttt, simtime in vel_params:
            out_dict[vel] = {}
            plot_dict[vel] = {}
            out_dict[vel][ttt] = {}
            plot_dict[vel][ttt] = {}
            for l in Lambda:
                out_dict[vel][ttt][l] = {}
                plot_dict[vel][ttt][l] = {}
                for ofs  in offset:
                    for hyst in hysteresis:
                        margin = ofs + 2*hyst
                        out_dict[vel][ttt][l][margin] = {}
                        plot_dict[vel][ttt][l][margin] = {}

                        print(vel, l, ofs, hyst)
                        tmp_dict, _ = load_result(
                        'instances/ho-params/{vel}/{tau}/{Lambda}/{of}/{hy}/'.format(
                            tau=ttt,
                            vel=vel,
                            Lambda=l,
                            hy=hyst,
                            of=ofs)).load()

                        for instance in tmp_dict.keys():
                            key = int(instance) #+ (dly//2)*int(len(tmp_dict.keys()))
                            print(key)
                            out_dict[vel][ttt][l][margin][key] = tmp_dict[instance].copy()

                            out_dict[vel][ttt][l][margin][key]['gap'] = calc_gap(tmp_dict[instance],simtime,68)
                            out_dict[vel][ttt][l][margin][key]['pingpong'] = calc_pingpongs(tmp_dict[instance])

                        key = list(out_dict[vel][ttt][l][margin].keys())[0]
                        for metric in out_dict[vel][ttt][l][margin][key].keys():
                            if metric not in no_plot:
                                try:
                                    plot_dict[vel][ttt][l][margin][metric].append(extract_metric(out_dict[vel][ttt][l][margin]).errorplot(metric))
                                except KeyError:
                                    plot_dict[vel][ttt][l][margin][metric] = []
                                    plot_dict[vel][ttt][l][margin][metric].append(extract_metric(out_dict[vel][ttt][l][margin]).errorplot(metric))

        with open('hoparams-out-1.json','w') as outfile:
            dump(plot_dict, outfile)

    else:
        try:
            with open('hoparams-out-1.json') as infile:
                pred = load(infile)

        except FileNotFoundError:
            exit()

        try:
            with open('temp.json') as infile:
                data = load(infile)
        except FileNotFoundError:
            exit()


        metrics = ['rsrp','handover','handoverRate','delay',
                'pingpong','gap','blockage','deliveryRate',
                'blk_ratio','blk_episodes','blk_duration','partDelay']
        lines = ['pred', 'pred','pred','opt','base']
        
        for metric in metrics:
            for vel, ttt, _ in vel_params:
                for n, line in enumerate(lines):
                    vel = str(vel)
                    ttt = str(ttt)
                    if line == 'pred':
                        window = str(pred_w[n])
                        plotter = np.transpose([pred[vel][ttt][str(i)][window][metric] for i in Lambda])
                        lbl='pred,w={w}'.format(w=window)
                        mean = plotter[0][0]
                        conf_int = plotter[1][0]
                    else:
                        plotter = np.transpose(data[line][vel][ttt][metric])
                        lbl=line
                        mean = plotter[0]
                        conf_int = plotter[1]


                    plt.errorbar(Lambda,mean,yerr=conf_int,label=lbl)

                plt.title('Velocity = {v}, TTT = {t}'.format(v=vel,t=ttt))
                plt.ylabel(metrics_dict[metric]['ylabel'])
                plt.xlabel('Blockage Density $\lambda$ [objects/$m^2$]')
                plt.grid()
                plt.legend()
                plt.savefig('{m}-{v}-{t}.png'.format(m=metric,v=vel,t=ttt))
                plt.show()

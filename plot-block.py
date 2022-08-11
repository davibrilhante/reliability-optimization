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


parser = argparse.ArgumentParser()
parser.add_argument('-d','--distribution', action='store_true')
parser.add_argument('-H','--histogram', action='store_true')
parser.add_argument('-a','--average',action='store_true')

args = parser.parse_args()

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


def pdf2cdf(pdf):
    length = len(pdf)
    cdf = [0 for i in range(length)]

    for i in range(length):
        cdf[i] = sum(pdf[:i])

    return cdf

def smooth(data, length):
    box = np.ones(length)/length
    data_smooth = np.convolve(data, box, mode='same')
    return data_smooth


if __name__ == "__main__":
    labels = ['{0:.3f}'.format(0.002*i + 0.001) for i in range(5)]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=13)
    #use('PS')


    results={
                'serving': {},
                'all':{}
            }

    neighbourhood = {
            0: [1,2],
            1: [0,2,3,4],
            2: [0,1,3],
            3: [1,2,4,6],
            4: [1,3,5,6],
            5: [4,6,7],
            6: [3,4,5,7,8],
            7: [5,6,8,9,10],
            8: [6,7,9],
            9: [7,8,10,11],
            10: [7,9,11,12],
            11: [9,10,12],
            12: [10,11]
            }

    #for data_type in ['opt', 'bas']:
    for index, Lambda in enumerate(labels):
        #data, metrics = load_result(f'instances/compare/{data_type}/22/{Lambda}/').load()
        
        rootdir = f'instances/full-scenario/22/{Lambda}/'

        serving_blockage_intervals = [] 
        all_blockage_intervals = [] 
        blocked_basestations = []
        
        for filename in listdir(rootdir):

            fullpath = rootdir+filename

            if path.isfile(fullpath):

                with open(fullpath, 'r') as jsonfile:

                    rawinstance = load(jsonfile)
                    decompressor(rawinstance)
                
                jsonfile.close()

                '''
                for assoc in data[filename]['association']:
                    servingbs = assoc[0]
                    init = assoc[1]
                    try:
                        end = assoc[2]
                    except IndexError:
                        continue
                    
                    # The sum() has the LoS timeslots counted and the (end-init),
                    # the whole interval. So, the subtraction of those two gives
                    # the sum of NLOS timeslots
                    assoc_blockage_interval = (end - init) - sum(rawinstance['blockage'][servingbs][0][init:end])
                    serving_blockage_intervals.append(assoc_blockage_interval)
                '''
                m_bs = len(rawinstance['blockage'])

                for m, bslist in enumerate(rawinstance['blockage']):
                    neighbourhood[m] = [i for i in range(m_bs) if i != m]
                    neighbour_state = []
                    for n, neighbour in enumerate(neighbourhood[m]):
                        neighbour_state.append([])

                    for uelist in bslist:
                        for t, state in enumerate(uelist):
                            try:
                                blocked_basestations[t].append(0)
                            except IndexError:
                                blocked_basestations.append([0])

                            if t == 0:
                                blocked_timeslots = 0
                                los_state = 0
                                blockage_flag = False
                                if state == 0:
                                    blocked_timeslots = 1
                                    blockage_flag = True
                                continue

                            for n, neighbour in enumerate(neighbourhood[m]):
                                if rawinstance['blockage'][neighbour][0][t] == 0:
                                    neighbour_state[n].append(0)
                                    blocked_basestations[t][-1] += 1


                            #else:
                            previous_state = uelist[t-1]

                            if state == 0 and previous_state == 1:
                                blockage_flag = True

                                for n, neighbour in enumerate(neighbourhood[m]):
                                    neighbour_state[n].append(0)

                            elif state == 1 and previous_state == 1:
                                los_state += 1

                            # Blockage period is over, then store the blockage 
                            # duration and reset the counter
                            #elif los_state > 20 and previous_state == 0:
                            #    los_state = 0
                            elif state == 1 and previous_state == 0:
                                #all_blockage_intervals.append(blocked_timeslots)
                                maxis = []
                                for n, neighbour in enumerate(neighbourhood[m]):
                                    maxis.append(np.max(neighbour_state[n]))
                                    #print(np.max(neighbour_state[n]))
                                    neighbour_state[n].clear()
                                #print('max',np.max(maxis))
                                all_blockage_intervals.append(np.max(maxis))
                                blocked_timeslots = 0
                                blockage_flag = False


                            if blockage_flag:
                                for n, neighbour in enumerate(neighbourhood[m]):
                                    if rawinstance['blockage'][neighbour][0][t] == 1:
                                        try:
                                            neighbour_state[n][-1] += 1
                                        except IndexError:
                                            neighbour_state[n].append(1)

                                blocked_timeslots += 1

        #results['serving'][Lambda] = {}
        results['all'][Lambda] = {}

        #results['serving'][Lambda]['raw'] = serving_blockage_intervals
        results['all'][Lambda]['raw'] = all_blockage_intervals
        #print(results['all'][Lambda]['raw'])

        for i, bsarray in enumerate(blocked_basestations):
            blocked_basestations[i] = np.mean(bsarray)

        results['all'][Lambda]['slot'] = blocked_basestations



            
        if args.distribution:
            #get the PDFs
            #serv_bins = np.unique(serving_blockage_intervals, return_counts=True)
            all_bins = np.unique(all_blockage_intervals, return_counts=True)

            #for plot_type in ['serving','all']:
            plot_type = 'all'

            bins = np.unique(results[plot_type][Lambda]['raw'], return_counts=True)

            # Data cleanning
            copy_bin = bins[0].tolist()
            copy_data = bins[1].tolist()

            for n, i in enumerate(bins[0]):
                if bins[1][n] < 3:# or i > 5000 or i < 100:
                    index = copy_bin.index(i)
                    copy_bin.pop(index)
                    copy_data.pop(index)

            results[plot_type][Lambda]['bins'] = copy_bin[1:]
            n_uniques = sum(copy_data[1:])

            results[plot_type][Lambda]['pdf'] = []

            temp = []
            for i in copy_data[1:]:
                #results[plot_type][Lambda]['pdf'].append(i/n_uniques)
                temp.append(i/n_uniques)

            results[plot_type][Lambda]['pdf'] = smooth(temp, min(25, len(temp))) 
            #print(results[plot_type][Lambda]['pdf'], results[plot_type][Lambda]['bins'])

            #results[plot_type][Lambda]['cdf'] = pdf2cdf(results[plot_type][Lambda]['pdf'])
            results[plot_type][Lambda]['cdf'] = np.cumsum(results[plot_type][Lambda]['pdf'])

            

            #results['serving'][Lambda]['data'], results['serving'][Lambda]['bins'] = np.histogram(serving_blockage_intervals, bins=serv_bins, density=True)
            #results['all'][Lambda]['data'], results['all'][Lambda]['bins'] = np.histogram(all_blockage_intervals, bins=all_bins, density=True)

        if args.histogram:
            #test_bins = [0, 100, 1000, 10000, 100000]#[i*100 for i in range(21)] + [4000 + i*2000 for i in range(9)]
            test_bins = [100 + i*100 for i in range(20)] + [4000 + i*2000 for i in range(9)]
            #width_array = [80, 800, 8000, 80000]#[80 for i in range(20)] + [800 for i in range(9)]
            width_array = [80 for i in range(19)] + [800 for i in range(9)]

            results['all'][Lambda]['hist'], bins = np.histogram(
                                    results['all'][Lambda]['raw'], 
                                    bins=test_bins)#, density=True)

            #print(results['all'][Lambda]['hist'])
            #print(bins)



    if args.distribution:
        #for plot_type in results.keys():
        plot_type = 'all'


        for Lambda in labels:
            plt.semilogx(results[plot_type][Lambda]['bins'],
                    results[plot_type][Lambda]['pdf'], 
                    label=Lambda)

        plt.xlim(0,30000)
        plt.ylabel('PDF $[f_{\lambda}(t)]$')
        plt.xlabel('blockage duration (t)')
        plt.grid(True, which="both", linestyle='-')
        plt.tight_layout()
        plt.legend()
        plt.show()

        for Lambda in labels:
            plt.semilogx(results[plot_type][Lambda]['bins'],
                    results[plot_type][Lambda]['cdf'], 
                    label=Lambda)
        plt.xlim(0, 30000)
        plt.ylabel('CDF $[F_{\lambda}(t)]$')
        plt.xlabel('blockage duration (t)')
        plt.grid(True, which="both", linestyle='-')
        plt.legend()
        plt.show()

    if args.histogram:
        for Lambda in labels:
            #plt.hist(results['all'][Lambda]['hist'], bins)
            plt.bar(bins[:-1], results['all'][Lambda]['hist'], width=width_array)
            plt.title(f'Blockage Density $\lambda$ = {Lambda}')
            plt.xlabel('Blockage Duration t')
            plt.xscale('log')
            plt.ylim(0,10000)
            plt.grid()
            plt.show()


    if args.average:
        for Lambda in labels:
            plt.plot(results['all'][Lambda]['slot'], label=Lambda)
        plt.grid()
        plt.legend()
        plt.xlabel('Timeslot t')
        plt.ylabel('Average Blocked BS')
        plt.show()

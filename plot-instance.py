from matplotlib import pyplot as plt
import argparse
import sys
import json
import numpy as np
from decompressor import decompressor

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
parser.add_argument('-n','--node', help='The node of interest', type=int, default=0, nargs="+")
parser.add_argument('-a','--all_bs',help='show all BS SNR', action='store_true')

args = parser.parse_args()


def todb(x : float) -> float:
    return 10*np.log10(x)


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

    return 10**((calc_recv(base, user, channel, los, t) - noise_power)/10)


def load_inputFile(inputFile, beginning = 0):
    network = []
    nodes = []
    print('Loading Instance...')

    with open(inputFile) as json_file:
        try:
            data = json.load(json_file)
        except:
            sys.exit()

        decompressor(data)

        scenario = data['scenario']
        channel = data['channel']
        LOS = data['blockage']
        #gamma = data['gamma']
        for p in data['baseStation']:
            network.append(p)
        for p in data['userEquipment']:
            nodes.append(p)
     # Resource blocks attribution
    R = []
    for bs in network:
        bw_per_rb = 12*bs['subcarrierSpacing'] #12 subcarriers per resouce block times 120kHz subcarrier spacing
        R.append(bw_per_rb*bs['resourceBlocks'])

    return scenario, channel, LOS, network, nodes, R


def snr_processing(scenario, network, nodes, channel, LOS, beginning=0):
    SNR = []            
                        
    print('Preprocessing...')
                        
    # SNR evaluation    
    print('SNR Evaluation...')
    for m, bs in enumerate(network):
        SNR.append([])  
        for n,ue in enumerate(nodes):
            SNR[m].append([])
            #Adding the time dependency
            for t in range(scenario['simTime']):
                if LOS[m][n][beginning+t] == 1:
                    los = True
                else:   
                    los = False
                SNR[m][n].append(todb(calc_snr(bs,ue,channel,los,t)))
                        
    return SNR

if __name__ == "__main__":
    scenario, channel, LOS, network, nodes, R = load_inputFile(args.inputFile)

    m_bs = len(network)
    n_ue = len(nodes)

    SNR = snr_processing(scenario, network, nodes, channel, LOS)

    if args.all_bs:
        for m in network:
            plt.plot(SNR[m['index']][0], label=m['index'])
    else:
        for m in args.node:
            plt.plot(SNR[m][0], label=m)

    plt.legend()
    plt.show()

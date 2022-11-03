import numpy as np
from matplotlib import pyplot as plt
import json

from decompressor import decompressor
import operator

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
from plot_utils import calc_delays


def todb(x : float) -> float:
    return 10*np.log10(x)

def tolin(x : float) -> float:
    return 10**(x/10)

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

def load_inputFile(inputFile, beginning = 0, span = 5000):
    network = []
    nodes = []

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

    scenario['simTime'] = min(span, scenario['simTime'])

    for ue in nodes:
        #ue['nPackets'] = int(scenario['simTime']/120 - 1)
        #ue['packets'] = ue['packets'][:ue['nPackets']]
        temp = []
        for arrival in ue['packets']:
            if arrival < scenario['simTime']:
                temp.append(arrival)
        ue['packets'] = temp
        ue['nPackets'] = len(temp)
        ue['capacity'] = 750e6
        ue['delay'] = 1



        ue['threshold'] = tolin(15) #10**(ue['threshold']/10)

        ue['position']['x'] += (ue['speed']['x']/3.6)*(beginning*1e-3)
        ue['position']['y'] += (ue['speed']['y']/3.6)*(beginning*1e-3)


    n_ue = len(nodes)
    m_bs = len(network)

    # Resource blocks attribution
    R = []
    for bs in network:
        bw_per_rb = 12*bs['subcarrierSpacing'] #12 subcarriers per resouce block times 120kHz subcarrier spacing
        R.append(bw_per_rb*bs['resourceBlocks'])

    return scenario, channel, LOS, network, nodes, R


def snr_processing(scenario, network, nodes, channel, LOS, beginning=0):
    SNR = []
    RSRP = []

    # SNR evaluation
    for m, bs in enumerate(network):
        SNR.append([])
        RSRP.append([])
        for n,ue in enumerate(nodes):
            SNR[m].append([])
            RSRP[m].append([])
            #Adding the time dependency
            for t in range(scenario['simTime']):
                if LOS[m][n][beginning+t] == 1:
                    los = True
                else:
                    los = False
                rsrp = calc_recv(bs, ue, channel, los, t)
                RSRP[m][n].append(rsrp)
                SNR[m][n].append(calc_snr2(rsrp, channel))

    return SNR, RSRP

def beta_processing(SNR, m_bs, n_ue, simTime, offset=3,hysteresis=0, tau=640):
    # Creating Beta array (handover flag)
    beta = []

    print('Generating Beta Array...')
    for n in range(n_ue):
        for p in range(m_bs):
            beta.append([])
            handover_points = {}
            for q in range(m_bs):
                beta[p].append([])
                counter = 0
                snr_accumulator = []
                if p != q:
                    beta[p][q].append([])
                    for t in range(simTime):
                        diff = todb(SNR[q][n][t]) - (todb(SNR[p][n][t]) +
                                 offset + 2*hysteresis)

                        beta[p][q][n].append(0)

                        if counter >= tau: # sum(temp[t-tau:t])>=tau:
                            counter = 0
                            #counter -= 1
                            try:
                                handover_points[t].append([q, np.mean(snr_accumulator)])
                            except KeyError as error:
                                handover_points[t] = [[q, np.mean(snr_accumulator)]]

                            beta[p][q][n][t] = 1

                        if diff >= 0:
                            counter += 1
                            snr_accumulator.append(todb(SNR[q][n][t]))

                        else:
                            counter = 0

            for t in handover_points.keys():
                #print(handover_points[t])
                best_bs = max(handover_points[t], key=operator.itemgetter(1))
                #'''
                #'''
                beta[p][best_bs[0]][n][t] = 1

    return beta

def bs_snr(bs, ue, channel, LOS, t):
    if LOS[bs['index']][0][t] == 1:
        los = True
    else:
        los = False

    rsrp = calc_recv(bs, ue, channel, los, t)
    snr = todb(calc_snr2(rsrp, channel))

    return snr


if __name__ == "__main__":

    seed = 0
    offset = 3
    hysteresis = 0
    out_dict = {}

    for var in [round(i*0.002 + 0.001,3) for i in range(5)]: 
        out_dict[var] = {}
        for vel, ttt, t in [(22,160,203647),(43,80,101823),(64,40,67882)]:
            out_dict[var][vel]={}
            result, _ = load_result('instances/no-interference/opt/{tau}/{vel}/{Lambda}/750e6/1/'.format(
                            vel=vel,
                            tau=ttt,
                            Lambda=var)).load()

            for seed in range(60):
                print('=x= {var} {vel} {seed} =x='.format(var=var,vel=vel,seed=seed))
                out_dict[var][vel][seed]={}
                instance = 'instances/full-scenario/{vel}/{Lambda}/{seed}'.format(
                                vel=vel,
                                Lambda=var,
                                seed=seed)
                scenario, channel, LOS, network, nodes, R = load_inputFile(instance, 0, t)
                
                m_bs = len(network)
                n_ue = len(nodes)

                n_conditional_op = [[0 for j in network] for i in network]
                n_conditional_ho = [[0 for j in network] for i in network]
                ho_probability = []

                SNR, RSRP = snr_processing(scenario, network, nodes, channel, LOS, 0)

                n_opportunities = 0
                n_handovers = 0 
                counter = 0
                last_bs = None
                beta = {}
                for assoc in result[str(seed)]['association']:
                    p = assoc[0]
                    sbs = network[p]
                    init = assoc[1]
                    end = min(assoc[2]+2,scenario['simTime'])

                    if last_bs == None:
                        ho_probability.append((0,0))
                    else:
                        n_conditional_ho[last_bs][p] += 1
                        ho_probability.append((init,n_handovers/n_opportunities))


                    opportunity = {}
                    for q,tbs in enumerate(network):
                        counter = 0
                        for t in range(init, end):
                            if p != q:
                                sbs_snr = todb(SNR[p][0][t])#bs_snr(sbs,nodes[0],channel,LOS,t)
                                tbs_snr = todb(SNR[q][0][t])#bs_snr(tbs,nodes[0],channel,LOS,t)

                                diff = tbs_snr - (sbs_snr + offset + 2*hysteresis)

                                if counter >= ttt:
                                    counter = 0
                                    #beta[p,q,t] = 1
                                    n_conditional_op[p][q] += 1 #beta[p,q,t]
                                    opportunity[t] = 1 #max(opportunity,beta[p,q,t])

                                if diff >= 0:
                                    counter += 1
                                else:
                                    counter = 0


                                #n_conditional_op[p][q] += beta[p,q,t]
                                #opportunity = max(opportunity,beta[p,q,t])

                    n_opportunities += sum(opportunity.values())

                    n_opportunities = max(n_opportunities,1)
                    n_handovers += 1

                    last_bs = p

                for p in range(m_bs):
                    for q in range(m_bs):
                        if p!=q and n_conditional_op[p][q] != 0:
                            n_conditional_ho[p][q] /= sum(n_conditional_op[p])
                    n_conditional_ho[p][p] = 1 - sum(n_conditional_ho[p])


                out_dict[var][vel][seed]['total'] = ho_probability
                out_dict[var][vel][seed]['cond'] = n_conditional_ho
    with open('prob.json','w') as out_file:
        json.dump(out_dict,out_file)

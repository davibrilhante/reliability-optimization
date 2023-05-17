#!/usr/bin/env python
# -*- coding : utf8 -*-


import json
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import argparse
import sys
from matplotlib import pyplot as plt
import time
import operator
from multiprocessing import Process, Manager, Queue, Pool
import subprocess
from guppy import hpy
import logging


from add_constraints import add_all_constraints
from decompressor import decompressor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inputFile', help='Instance json input file')
    parser.add_argument('-o','--outputFile', default='results', help='outputs json result file')
    parser.add_argument('-r','--report', default='rel', help='cpu and memory resources profiling report file')
    parser.add_argument('-p','--plot', action='store_true', help='Enables plot')
    parser.add_argument('-P','--Print', action='store_true', help='Enables printing variables')
    parser.add_argument('-s','--save', action='store_true', help='Save statistics')
    #parser.add_argument('-t','--threads', type=int, help='Number of threads', default=3)
    parser.add_argument('--ttt', type=int, default=640)
    parser.add_argument('-b','--beginning', type=int, default=0)
    parser.add_argument('-T','--simutime', type=int, default=5000)
    parser.add_argument('--uedelay', type=int)
    parser.add_argument('--uecapacity', type=float)


    args = parser.parse_args()
    return args

heap = hpy()
heap_status_0 = heap.heap()

heap.setref()

global mvars
mvars = {}

logging.basicConfig(filename='myfirstlog.log',
                    level=logging.DEBUG,
                    format='%(asctime)s : %(funcName)s : %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler('myfirstlog.log')
formatter = logging.Formatter('%(asctime)s : %(funcName)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def todb(x : float) -> float:
    return 10*np.log10(x)

def tolin(x : float) -> float:
    return 10**(x/10)

def getKpi(x, y, m_bs, n_ue, simTime, SNR, BW, packets, delayTolerance, RSRP):
    #create dict
    kpi = {}

    nPackets = len(packets)

    kpi['deliveryRate'] = 0
    kpi['partDelay'] = 0
    linearSnr = []
    snr = []
    rsrp = []
    cap = []
    #get average snr
    for m in range(m_bs):
        for t in range(simTime):
            try:
                if x[m,n_ue,t].getAttr('X') == 1:
                    #val = x[m][n_ue][t]*SNR[m][n_ue][t]
                    #val = x[m,n_ue,t].getAttr('X')*SNR[m][n_ue][t]
                    val = SNR[m][n_ue][t]
                    linearSnr.append(val)
                    snr.append(todb(val))
                    if RSRP is not None:
                        rsrp.append(RSRP[m][n_ue][t])

                    cap.append(BW[m]*np.log2(1+val))
            except KeyError:
                pass
            
            try:
                if y[m,n_ue,t].getAttr('X')==1:
                    kpi['deliveryRate']+=1
                    kpi['partDelay']+=1
            except KeyError:
                pass


    kpi['deliveryRate']/=nPackets
    kpi['partDelay']/=nPackets
    kpi['snr'] = np.mean(snr)
    if rsrp:
        kpi['rsrp'] = np.mean(rsrp)
    kpi['linearSNR'] = np.mean(linearSnr)
    kpi['capacity'] = np.mean(cap)
    

    associated = [[],[],[]]
    for t in range(simTime):
        for m in range(m_bs):
            try:
                if x[m,n_ue,t].getAttr('X') == 1:
                    if (len(associated[0]) > 0 and associated[0][-1] != m) or len(associated[0])==0:
                        if len(associated[0]) > 1: 
                            associated[2].append(todb(SNR[m][n_ue][t]) - 
                                    10*np.log10(SNR[associated[0][-1]][n_ue][t-args.ttt]))
                        else:
                            associated[2].append(0)

                        associated[0].append(m)
                        associated[1].append(t)
            except KeyError:
                pass

    num = 0
    for m in range(m_bs):
        if associated[0].count(m)>1:
            num+= associated[0].count(m)-1
    rate = num/len(associated[0])

    kpi['handover'] = len(associated[0]) - 1
    kpi['handoverRate'] = kpi['handover']/(simTime*1e-3)
    kpi['pingpong'] = rate 

    # Gaps or time out of sync!
    T311 = 1000
    kpi['gap'] = 0
    for t in range(simTime):
        for m in range(m_bs):
            if x[m,n_ue,t].getAttr('X') == 1 and SNR[m][n_ue][t] <= tolin(15) and t >= T311:
                if outofsync(SNR[m][n_ue][t-T311:t]):
                    kpi['gap'] += 1

            

    #Average delay
    delay = []
    for p in packets:
        for m in range(m_bs):
            for k in range(delayTolerance):
                if y[m, n_ue,p+k].getAttr('X') == 1:
                    delay.append(k)
                    break

    #print(np.mean(delay))
    kpi['delay'] = np.mean(delay)

    kpi['association'] = []
    for i in range(kpi['handover']+1):
        if i != (kpi['handover']):
            kpi['association'].append([associated[0][i], associated[1][i], associated[1][i+1]-1, associated[2][i]])
        else:
            kpi['association'].append([associated[0][i], associated[1][i], simTime, associated[2][i]])

    return kpi


def outofsync(snrslice, qout=-7.2, qin=-4.8, N310=1, N311=1, T310=1000):
    # Checks Synchronization with base station according 3gpp protocol
    outcounter = 0
    incounter = 0
    synctimer = 0

    for sample in snrslice:
        if outcounter >= N310:
            synctimer += 1

        if incounter == N311:
            synctimer = 0

        if sample <= tolin(qout):
            outcounter += 1
            incounter = 0

        if sample >= tolin(qin):
            incounter += 1
            outcounter = 0
    
    if synctimer >= T310:
        return True

    else:
        return False
    


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


def handover_detection(_vars):
    handovers = []
    for p,n1,t1 in _vars:
        for q,n2,t2 in _vars:
            if p!=q and n1==n2 and t2 > t1:
                n = n1
                handovers.append([p,q,n,t1,t2])

    return handovers


comp_resources = {
    'loading': [0, 0],
    'snrprocess': [0,0],
    'betaprocess': [0,0],
    'addvars': [0,0],
    'addconsts': [0,0],
    'addobj': [0,0],
    'optimize': [0,0],
    'log': [0,0],
    'plot': [0,0]
    }



#topdir = 'instances/full-scenario/'
### Create base data

def load_inputFile(inputFile, beginning = 0, span = 5000):
    network = []
    nodes = []
    print('Loading Instance...')
    start = time.time()
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


        if args.uecapacity:
            ue['capacity'] = args.uecapacity #Bits per second
        else:
            ue['capacity'] = 750e6

        if args.uedelay:
            ue['delay'] = args.uedelay

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

    comp_resources['loading'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['loading'][1] = heap_stat.size/(1024*1024)

    logger.info('Function processing time %d:'%comp_resources['loading'][0])
    logger.info('Memory stat: %d'%comp_resources['loading'][1])

    return scenario, channel, LOS, network, nodes, R

def snr_processing(scenario, network, nodes, channel, LOS, beginning=0):
    SNR = []
    RSRP = []

    print('Preprocessing...')

    start = time.time()

    # SNR evaluation
    print('SNR Evaluation...')
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

    comp_resources['snrprocess'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['snrprocess'][1] = heap_stat.size/(1024*1024)

    logger.info('Function processing time : %d'%comp_resources['snrprocess'][0])
    logger.info('Memory stat: %d'%comp_resources['snrprocess'][1])

    return SNR, RSRP

def beta_processing(SNR, m_bs, n_ue, simTime, offset=3,hysteresis=0, tau=640):
    # Creating Beta array (handover flag)
    start = time.time()
    beta = []

    print('Generating Beta Array...')
    for n in range(n_ue):
        for p in range(m_bs):
            logging.info('--------- %2d ----------'%p)
            logging.info('%3s |%7s | %8s'%('tgt', 'slot', 'snr diff'))
            logging.info('-----------------------')
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
                                #logger.debug('Key Error %d at BS %d to BS %d'%(t, p, q))

                            beta[p][q][n][t] = 1
                            logging.info('%4d,%8d,%8d'%(q, t, SNR[q][n][t] - SNR[p][n][t]))


                        if diff >= 0:
                            counter += 1
                            snr_accumulator.append(todb(SNR[q][n][t]))

                        else:
                            counter = 0
                            snr_accumulator = []
                        '''
                        else:
                            beta[p][q][n].append(0)

                        if sum(temp[t-tau:t])>=tau:
                            print('Is possible to handover from %i to %i at %i'%(p,q,t))
                        '''
            
            for t in handover_points.keys():
                #print(handover_points[t])
                best_bs = max(handover_points[t], key=operator.itemgetter(1))
                #'''
                #'''
                beta[p][best_bs[0]][n][t] = 1

    comp_resources['betaprocess'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['betaprocess'][1] = heap_stat.size/(1024*1024)

    
    logger.info('Function processing time : %d'%comp_resources['betaprocess'][0])
    logger.info('Memory stat: %d'%comp_resources['betaprocess'][1])
    return beta




def model_setting():
    ### Create environment and model
    optEnv = gb.Env('myEnv.log')
    #optEnv.setParam('OutputFlag', 0)
    model = gb.Model('newModel', optEnv)

    ### Quadratic constraints control
    model.presolve().setParam(GRB.Param.PreQLinearize,1)
    #model.setParam(GRB.Param.Threads, args.threads)
    
    return model

def add_variables(model, scenario, network, nodes, SNR, beta):
    ### Add variables to the model
    start = time.time()
    print('Adding model variables...')
    m_bs = len(network)
    n_ue = len(nodes)
    M = {i for i in range(m_bs)}

    #x = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='x')
    #y = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='y')
    #u = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='u')

    x = {}
    for m in M:
        for n, ue in enumerate(nodes):
            ue[m, 'available'] = set()
            for t in range(scenario['simTime']):
                #if SNR[m][n][t] >= ue['threshold']:
                x[m,n,t] = model.addVar(vtype=GRB.BINARY,
                            name='x[{bs},{ue},{tempo}]'.format(bs=m,ue=n,tempo=t))
                ue[m, 'available'].add(t)


    y = {}
    for n, ue in enumerate(nodes):
        for arrival in ue['packets']:
            for t in range(arrival,arrival + ue['delay']):
                for m in M:
                    y[m,n,t] = model.addVar(vtype=GRB.BINARY,
                                name='y[{bs},{ue},{tempo}]'.format(bs=m,ue=n,tempo=t))



    u = {}
    for p in M:
        for n in range(n_ue):
            for t in range(scenario['simTime']):
                for q in M - {p}:
                    if beta[p][q][n][t]==1:
                        u[p,n,t] = model.addVar(vtype=GRB.BINARY, 
                                name='u[{bs},{ue},{tempo}]'.format(bs=p,ue=n,tempo=t))
                        break



    #sumbeta = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='sumbeta')

    comp_resources['addvars'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['addvars'][1] = heap_stat.size/(1024*1024)

    logger.info('Function processing time : %d'%comp_resources['addvars'][0])
    logger.info('Memory stat: %d'%comp_resources['addvars'][1])

    return x, y, u

def model_optimize(model, x, y, scenario, network, nodes, SNR, beta, R, m_bs, n_ue):
    ### Set objective function
    #
    # 1 - Maximize the number of users which delay and capacity requirements were
    # fulfilled
    print('Setting model objective function...')
    intervals = {}
    for n, ue in enumerate(nodes):
        intervals[n] = []
        for arrival in ue['packets']:
            for t in range(arrival, arrival+ue['delay']):
                intervals[n].append(t)

    start = time.time()
    model.setObjective(
            gb.quicksum(
                gb.quicksum(
                    gb.quicksum(
                        #(1-gamma[m][n][t])*(x[m][n][t]+y[m][n][t]) 
                        #(x[m,n,t]+y[m,n,t]) 
                        (SNR[m][n][t]*x[m,n,t])
                            for t in nodes[n][m,'available'])
                        + gb.quicksum(y[m,n,t] 
                            for t in intervals[n])
                    for n in range(n_ue)) 
                for m in range(m_bs)), 
            GRB.MAXIMIZE
            )

    model._beta = gb.tuplelist(beta)

    #model.Params.lazyConstraints = 1

    model.write('myModel.lp')

    comp_resources['addobj'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['addobj'][1] = heap_stat.size/(1024*1024)


    logger.info('Function processing time : %d'%comp_resources['addobj'][0])
    logger.info('Memory stat: %d'%comp_resources['addobj'][1])

    ### Compute optimal Solution
    start = time.time()
    try:
        print('Begining Optimization')
        model.optimize()#handover_callback)
        end = time.time()
        print(end - start)


        mvars.update({
            'nvars': model.getAttr('numVars'),
            'nconst': model.getAttr('numConstrs'),
            'nqconst': model.getAttr('numQConstrs'),
            'ngconst': model.getAttr('numGenConstrs'),
            'status': model.Status == GRB.OPTIMAL,
            'runtime': model.getAttr('Runtime'),
            'node': model.getAttr('nodecount'),
            'obj': model.objVal,
        })


    except gb.GurobiError as error:
        logging.debug('Optimization  failed\n\n')
        logging.debug(error)
        end = time.time()
        print(end - start)
        sys.exit()




    comp_resources['optimize'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['optimize'][1] = heap_stat.size/(1024*1024)

    logger.info('Function processing time : %d'%comp_resources['optimize'][0])
    logger.info('Memory stat: %d'%comp_resources['optimize'][1])

    return mvars



##################### Collecting network results ##############################
def statistics(x, y, m_bs, SNR, R, scenario, save=True,_print=False, outputFile=None, RSRP=None):
    start = time.time()
    kpi = {}

    try:
        kpi = getKpi(x, y, m_bs, 0, scenario['simTime'], SNR, R, nodes[0]['packets'], nodes[0]['delay'], RSRP)
    except Exception as error:
        logging.debug(error)

    kpi['optimization'] = mvars


    finaldict = {}
    if _print:
        oldv = ''
        vardict = {}
        oldvalue = None
        for v in model.getVars():
            varname = v.varName.split('[')
            varindex = varname[1].split(',')[0]

            if varname[0]+varindex != oldv:
                try:
                    vardict[oldvalue][-1].append(scenario['simTime'])
                except KeyError:
                    pass

                if oldv != '':
                    finaldict[vardict['variable']] = vardict

                oldv = varname[0]+varindex
                vardict = {}
                vardict['variable'] = varname[0]+'_'+varindex
                oldvalue = None

            if varname[0] == 'y':
                if v.x != 0:
                    timeslot = int(varname[1].split(',')[2].split(']')[0])
                    try:
                        vardict['timeslot'].append(timeslot)
                    except KeyError:
                        vardict['timeslot'] = [timeslot]


            else:
                if v.x != oldvalue:
                    if oldvalue is not None:
                        fim = int(varname[1].split(',')[2].split(']')[0]) - 1
                        vardict[oldvalue][-1].append(fim)

                        try:
                            vardict[v.x].append([fim+1])

                        except KeyError:
                            vardict[v.x] = [[fim+1]]
                    else:
                        ini = int(varname[1].split(',')[2].split(']')[0])
                        vardict[v.x] = [[ini]]
                oldvalue = v.x
        print(finaldict)

    kpi['variables'] = finaldict

    if save:
        filename = outputFile
        with open(filename, 'w') as jsonfile:
            json.dump(kpi, jsonfile, indent=4)

    results = json.dumps(kpi, indent=4)
            
    comp_resources['log'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['log'][1] = heap_stat.size/(1024*1024)

    logger.info('Function processing time : %d'%comp_resources['log'][0])
    logger.info('Memory stat: %d'%comp_resources['log'][1])

############################## PLOT SECTION ###################################
def plot_stats(model, x, y, SNR, m_bs):
    start = time.time()
    print('Ploting SNR')
    plot = []

    for m in range(m_bs):
        plot.append([])
        timeslots = []
        for t in range(scenario['simTime']):
            if t in nodes[0][m,'available'] and x[m,0,t].getAttr('X') == 1:
                plot[-1].append(todb(SNR[m][0][t]))
                timeslots.append(t)
                #print(t, SNR[m][0][t])

        #plt.plot(plot)
        if plot[-1]:
            plt.scatter(timeslots,plot[-1], label=m)#, color=colors[m])

    plot = []
    for m in range(6):
        plot.append([])
        timeslots = []
        for t in range(scenario['simTime']):
            if t%50 == 0:
                plot[-1].append(todb(SNR[m][0][t]))
                timeslots.append(t)
                #print(t, SNR[m][0][t])

        #plt.plot(plot)
        plt.scatter(timeslots,plot[-1], marker = '+', label=m)#, color = colors[m])

    intervals = {}
    for n, ue in enumerate(nodes):
        intervals[n] = []
        for arrival in ue['packets']:
            for t in range(arrival, arrival+ue['delay']):
                intervals[n].append(t)

    plot = []
    for m in range(m_bs):
        plot.append([])
        timeslots = []
        for t in intervals[0]:
            if y[m,0,t].getAttr('X') == 1:
                plot[-1].append(0)
                timeslots.append(t)
                #print(t, SNR[m][0][t])

        #plt.plot(plot)
        if plot[-1]:
            plt.scatter(timeslots,plot[-1],marker='s')#, color=colors[m])

    plt.ylabel('SNR')
    plt.xlabel('Time (mS)')
    plt.legend()
    plt.savefig('snr.png')
    plt.show()

    comp_resources['plot'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['plot'][1] = heap_stat.size/(1024*1024)


    logger.info('Function processing time : %d'%comp_resources['plot'][0])
    logger.info('Memory stat: %d'%comp_resources['plot'][1])

def test_api(inputFile, beginning = 0, simutime = 50000, ttt = 640):
    scenario, channel, LOS, network, nodes, R = load_inputFile(inputFile, 0, simutime)

    m_bs = len(network)
    n_ue = len(nodes)
    scenario['ttt'] = ttt
    scenario['simTime'] = simutime

    SNR = snr_processing(scenario, network, nodes, channel, LOS)
    beta = beta_processing(SNR, m_bs, n_ue, simTime=scenario['simTime'], tau=scenario['ttt'])
    model = model_setting()
    x, y, u = add_variables(model, scenario, network, nodes, SNR, beta)

    print('Adding model Constraints...')
    start = time.time()

    Vars = [x, y, u]
    add_all_constraints(model, Vars, nodes, network, SNR, beta, R, scenario)

    comp_resources['addconsts'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['addconsts'][1] = heap_stat.size/(1024*1024)

    result = model_optimize(model, x, y, scenario, network, nodes, SNR, beta, R, m_bs, n_ue)

    intervals = {}
    for n, ue in enumerate(nodes):
        intervals[n] = []
        for arrival in ue['packets']:
            for t in range(arrival, arrival+ue['delay']):
                intervals[n].append(t)
    var_x = 0
    var_y = 0

    old_bs = None
    entry = 0
    assoc = []

    pre_objval = 0
    result['assoc'] = []
    for t in range(scenario['simTime']):
        for n in range(n_ue):
            for m in range(m_bs):
                if t in nodes[n][m,'available'] and x[m,n,t].x == 1:
                    var_x += 1
                    if m != old_bs:
                        old_bs = m
                        result['assoc'].append([m, t])

                    if t < beginning:
                        pre_objval += SNR[m][n][t]

                if t in intervals[n] and y[m,n,t].x == 1:
                    var_y += 1

                    if t < beginning:
                        pre_objval += 1

    
    for (bs, entry) in sorted(result['assoc'], key=lambda x: x[1], reverse=True):
        if entry < beginning or len(result['assoc']) == 1:
            result['last_bs'] = bs
            result['assoc_time'] = beginning - entry
            break

    result['pre_objval'] = pre_objval
    result['x'] = var_x
    result['y'] = var_y



    return result, SNR, network, nodes


if __name__ == '__main__':
    #proc = subprocess.Popen(['./monitor.sh &'], shell=True)

    args = get_args()

    scenario, channel, LOS, network, nodes, R = load_inputFile(args.inputFile, args.beginning, args.simutime)

    m_bs = len(network)
    n_ue = len(nodes)
    scenario['ttt'] = args.ttt

    SNR, RSRP = snr_processing(scenario, network, nodes, channel, LOS)


    beta = beta_processing(SNR, m_bs, n_ue, simTime=scenario['simTime'], tau=scenario['ttt'])
    model = model_setting()
    x, y, u = add_variables(model, scenario, network, nodes, SNR, beta)

    print('Adding model Constraints...')
    start = time.time()

    Vars = [x, y, u]
    add_all_constraints(model, Vars, nodes, network, SNR, beta, R, scenario)

    comp_resources['addconsts'][0] = time.time() - start
    heap_stat = heap.heap()
    comp_resources['addconsts'][1] = heap_stat.size/(1024*1024)

    model_optimize(model, x, y, scenario, network, nodes, SNR, beta, R, m_bs, n_ue)   
    statistics(x, y, m_bs, SNR, R, scenario, save=args.save, _print=args.Print, outputFile='instances/opt'+args.outputFile, RSRP=RSRP)
    
    interim_objval = 0
    for t in range(126663):
        for n in range(n_ue):
            for m in range(m_bs):
                if t in nodes[n][m,'available'] and x[m,n,t].x == 1:
                    interim_objval += SNR[m][n][t]


    if args.plot:
        plot_stats(model, x, y, SNR, m_bs)

        filename = 'instances/opt/plot_points_'+args.outputFile
        with open(filename, "w") as output_plot:
            for m in range(m_bs):
                for n in range(n_ue):
                    for t in range(scenario['simTime']):
                        if t in nodes[n][m,'available'] and x[m,n,t].x == 1:
                            output_plot.write(str(t)+','+str(m)+','+str(todb(SNR[m][n][t]))+'\n')

    print(mvars)
    print('obj: %g'% model.objVal)
    #print('X: %g'% x.sum('*','*','*').getValue())
    print('X: %g'% sum([i.x for i in x.values()]))
    #print('Y: %g'% y.sum('*','*','*').getValue())
    print('Y: %g'% sum([i.x for i in y.values()]))
    print(comp_resources)
    #print(proc.stdout)

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


from add_constraints import add_all_constraints
from decompressor import decompressor

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
parser.add_argument('-o','--outputFile', default='results', help='outputs json result file')
parser.add_argument('-p','--plot', action='store_true', help='Enables plot')
parser.add_argument('-s','--save', action='store_true', help='Save statistics')
parser.add_argument('-t','--threads', type=int, help='Number of threads', default=3)
parser.add_argument('--ttt', type=int, default=640)


args = parser.parse_args()

def todb(x : float) -> float:
    return 10*np.log10(x)

def getKpi(x, y, m_bs, n_ue, simTime, SNR, BW, nPackets):
    #create dict
    kpi = {}

    kpi['deliveryRate'] = 0
    kpi['partDelay'] = 0
    linearSnr = []
    snr = []
    cap = []
    #get average snr
    for m in range(m_bs):
        for t in range(simTime):
            if x[m,n_ue,t].getAttr('X') == 1:
                #val = x[m][n_ue][t]*SNR[m][n_ue][t]
                val = x[m,n_ue,t].getAttr('X')*SNR[m][n_ue][t]
                linearSnr.append(val)
                snr.append(todb(val))
                cap.append(BW[m]*np.log2(1+val))

            if y[m,n_ue,t].getAttr('X')==1:
                kpi['deliveryRate']+=1
                kpi['partDelay']+=1

    kpi['deliveryRate']/=nPackets
    kpi['partDelay']/=nPackets
    kpi['snr'] = np.mean(snr)
    kpi['linearSNR'] = np.mean(linearSnr)
    kpi['capacity'] = np.mean(cap)
    

    associated = [[],[],[]]
    for t in range(simTime):
        for m in range(m_bs):
            if x[m,n,t].getAttr('X') == 1:
                if (len(associated[0]) > 0 and associated[0][-1] != m) or len(associated[0])==0:
                    if len(associated[0]) > 1: 
                        associated[2].append(todb(SNR[m][n_ue][t]) - 
                                10*np.log10(SNR[associated[0][-1]][n_ue][t-args.ttt]))
                    else:
                        associated[2].append(0)

                    associated[0].append(m)
                    associated[1].append(t)

    num = 0
    for m in range(m_bs):
        if associated[0].count(m)>1:
            num+= associated[0].count(m)-1
    rate = num/len(associated[0])

    kpi['handover'] = len(associated[0]) - 1
    kpi['handoverRate'] = kpi['handover']/simTime
    kpi['pingpong'] = rate 
    kpi['association'] = []
    for i in range(kpi['handover']+1):
        if i != (kpi['handover']):
            kpi['association'].append([associated[0][i], associated[1][i], associated[1][i+1]-1, associated[2][i]])
        else:
            kpi['association'].append([associated[0][i], associated[1][i], simTime, associated[2][i]])

    '''
    #Average delay
    delay = []
    for m in range(m_bs):
        #for k in ue['packets']:
        for p in range(ue['nPackets']):
            k = ue['packets'][p]
            l = ue['packets'][p+1]
            #print(k)
            for t in range(k, l):
                if t < scenario['simTime'] and x[m][n][t] == 1:
                    delay.append(t - k)
                    break
    #print(np.mean(delay))
    kpi['delay'] = np.mean(delay)
    '''

    return kpi


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


def handover_callback(model, where):
    hit = 68
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)

        selected = gb.tuplelist((m, n, t) for m, n, t in model._vars.keys()
                                if vals[m,n,t] > 0.5)
        m_bs = len(vals)

        handovers = handover_detection(selected)

        for p, q, n, t1, t2 in handovers:
            #if model._beta[p,q,n,t2] == 0 and model._beta[q,p,n,t2] == 0:
            if t2 > t1+args.ttt:
                try:
                    '''
                    model.cbLazy(x[p,n,t1]*x[p,n,t2] +
                            beta[p][q][n][t2]*x[p,n,t1]*x[q,n,t2] +
                            beta[q][p][n][t2]*x[q,n,t1]*x[p,n,t2] +
                            x[q,n,t1]*x[q,n,t2] <= 1)
                    '''
                    model.cbLazy(sum(x[q,n,t] for t in range(t2,t2+hit))<=
                            1 - sum(model._beta[p,q,t1,t2]*x[p,n,t1]*x[q,n,t2] for p in range(m_bs) if p != q)
                    )
                except Exception as error:
                    print('Error adding lazy constraints to the model %i %i %i %i'%(p,q,t1,t2))
                    print(error)


def handover_detection(_vars):
    handovers = []
    for p,n1,t1 in _vars:
        for q,n2,t2 in _vars:
            if p!=q and n1==n2 and t2 > t1:
                n = n1
                handovers.append([p,q,n,t1,t2])

    return handovers





network = []
nodes = []

#topdir = 'instances/full-scenario/'
### Create base data
print('Loading Instance...')
start = time.time()
with open(args.inputFile) as json_file:
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

scenario['simTime'] = min(5000, scenario['simTime'])

for ue in nodes:
    ue['nPackets'] = int(scenario['simTime']/120 - 1)
    ue['packets'] = ue['packets'][:ue['nPackets']]
    ue['capacity'] = 750e6 #Bits per second
    ue['threshold'] = 10**(ue['threshold']/10)

n_ue = len(nodes)
m_bs = len(network)

end = time.time()
print(end - start)

SNR = []

print('Preprocessing...')

start = time.time()
### -------- Beginning of the Preprocessing phase ----------
#
# SNR evaluation
print('SNR Evaluation...')
for m, bs in enumerate(network):
    SNR.append([])
    for n,ue in enumerate(nodes):
        SNR[m].append([])
        #Adding the time dependency
        for t in range(scenario['simTime']):
            if LOS[m][n][t] == 1:
                los = True
            else:
                los = False
            SNR[m][n].append(calc_snr(bs,ue,channel,los,t))
            #if gamma[m][n][t] == float('inf') or gamma[m][n][t] == float('nan'):
            #    gamma[m][n][t] = 1.0

# Creating Beta array (handover flag)
tau = args.ttt
offset = 3 #dB 
hysteresis = 0 #db
beta = []
print('Generating Beta Array...')
for n in range(n_ue):
    for p in range(m_bs):
        #print('Base Station %i'%(p))
        beta.append([])
        handover_points = {}
        for q in range(m_bs):
            beta[p].append([])
            counter = 0
            snr_accumulator = []
            if p != q:
                beta[p][q].append([])
                for t in range(scenario['simTime']):
                    diff = todb(SNR[q][n][t]) - (todb(SNR[p][n][t]) + 
                             offset + 2*hysteresis)
                    beta[p][q][n].append(0)

                    if diff >= 0:
                        counter += 1
                        snr_accumulator.append(todb(SNR[q][n][t]))

                    else:
                        counter = 0
                        snr_accumulator = []

                    if counter >= tau: # sum(temp[t-tau:t])>=tau:
                       # print(p, q, t)
                        counter = 0
                        try:
                            handover_points[t].append([q, np.mean(snr_accumulator)])
                        except KeyError:
                            handover_points[t] = [[q, np.mean(snr_accumulator)]]
                    '''
                    else:
                        beta[p][q][n].append(0)

                    if sum(temp[t-tau:t])>=tau:
                        print('Is possible to handover from %i to %i at %i'%(p,q,t))
                    '''
        for t in handover_points.keys():
            #print(handover_points[t])
            best_bs = max(handover_points[t], key=operator.itemgetter(1))
            '''
            print(p, t, best_bs, 10*np.log10(SNR[p][n][t]), 
                    10*np.log10(SNR[best_bs[0]][n][t]))
            '''
            beta[p][best_bs[0]][n][t] = 1

'''
print('3', LOS[3][0][1350:1356], SNR[3][0][1350:1356])
for q in range(m_bs):
    if q != 3:
        print(q, beta[3][q][0][1350:1356], LOS[q][0][1350:1356], SNR[q][0][1350:1356]) 
'''


# Resource blocks attribution
R = []
for bs in network:
    bw_per_rb = 12*bs['subcarrierSpacing'] #12 subcarriers per resouce block times 120kHz subcarrier spacing
    R.append(bw_per_rb*bs['resourceBlocks'])

end = time.time()
print(end - start)
### ----------- End of preprocessing phase ---------------


### Create environment and model
optEnv = gb.Env('myEnv.log')
#optEnv.setParam('OutputFlag', 0)
model = gb.Model('newModel', optEnv)

### Quadratic constraints control
model.presolve().setParam(GRB.Param.PreQLinearize,1)
model.setParam(GRB.Param.Threads, args.threads)



### Add variables to the model
start = time.time()
print('Adding model variables...')
w = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='w')
x = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='x')
y = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='y')
u = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='u')
z = model.addVars(m_bs, m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='z')
b = model.addVars(m_bs, m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='b')

vars = x
end = time.time()
print(end - start)

start = time.time()
print('Adding model Constraints...')
start = time.time()

Vars = [x, y, z, b, w, u]
add_all_constraints(model, Vars, nodes, network, SNR, beta, R, scenario)

end = time.time()
print(end - start)
'''
# 9 - Delay and Capacity requirements coupling
for m in range(m_bs):
    for n in range(n_ue):
        for t in range(scenario['simTime']):
            model.addConstr(2*w[m][n][t] == x[m][n][t]+y[m][n][t])
#'''

### Set objective function
#
# 1 - Maximize the number of users which delay and capacity requirements were
# fulfilled
print('Setting model objective function...')
start = time.time()
model.setObjective(
        sum(
            sum(
                sum(
                    #(1-gamma[m][n][t])*(x[m][n][t]+y[m][n][t]) 
                    #(x[m,n,t]+y[m,n,t]) 
                    (SNR[m][n][t]*x[m,n,t])+y[m,n,t]
                    for t in range(scenario['simTime'])) 
                for n in range(n_ue)) 
            for m in range(m_bs)), 
        GRB.MAXIMIZE
        )

model._vars = vars
model._beta = gb.tuplelist(beta)

#model.Params.lazyConstraints = 1

model.write('myModel.lp')

end = time.time()
print(end - start)

### Compute optimal Solution
start = time.time()
try:
    print('Begining Optimization')
    model.optimize()#handover_callback)
    end = time.time()
    print(end - start)

except gb.GurobiError as error:
    print('Optimize  failed\n\n')
    print(error)
    end = time.time()
    print(end - start)
    sys.exit()



##################### Collecting network results ##############################
print('Generating results...')
print(beta[4][3][0][2210:2216])
print(beta[3][4][0][2210:2216])
print('x', x[3,0,1350].x, x[3,0,1351].x, x[3,0,1352].x)
print('u', u[3,0,1350].x, u[3,0,1351].x, u[3,0,1352].x)
print('b', b[3,4,0,1350].x, b[3,4,0,1351].x, b[3,4,0,1352].x)
print('z', z[3,4,0,1350].x, z[3,4,0,1351].x, z[3,4,0,1352].x)

kpi = getKpi(x, y, m_bs, 0, scenario['simTime'], SNR, R, nodes[0]['nPackets'])

if args.save:
    filename = 'instances/opt/'+args.outputFile
    with open(filename, 'w') as jsonfile:
        json.dump(kpi, jsonfile, indent=4)

results = json.dumps(kpi, indent=4)
#print(results)
#print(beta)

############################## PLOT SECTION ###################################
if args.plot:
    print('Ploting SNR')
    plot = []
    #colors = ['blue', 'orange', 'green', 'red']
    '''
    oldv = ''
    for v in model.getVars():
        varname = v.varName.split('[')
        if varname[0] != oldv:
            print('---',varname[0])
            oldv = varname[0]
        if v.x != 0:
            print('  %s %g' % (v.varName, v.x))
    '''

    for m in range(m_bs):
        plot.append([])
        time = []
        for t in range(scenario['simTime']):
            if x[m,0,t].getAttr('X') == 1:
                plot[-1].append(todb(SNR[m][0][t]))
                time.append(t)
                #print(t, SNR[m][0][t])

        #plt.plot(plot)
        if plot[-1]:
            plt.scatter(time,plot[-1], label=m)#, color=colors[m])

    plot = []
    for m in range(5):
        plot.append([])
        time = []
        for t in range(scenario['simTime']):
            if t%100 == 0:
                plot[-1].append(todb(SNR[m][0][t]))
                time.append(t)
                #print(t, SNR[m][0][t])

        #plt.plot(plot)
        plt.scatter(time,plot[-1], marker = '+', label=m)#, color = colors[m])


    plot = []
    for m in range(m_bs):
        plot.append([])
        time = []
        for t in range(scenario['simTime']):
            if y[m,0,t].getAttr('X') == 1:
                plot[-1].append(0)
                time.append(t)
                #print(t, SNR[m][0][t])

        #plt.plot(plot)
        if plot[-1]:
            plt.scatter(time,plot[-1],marker='s')#, color=colors[m])

    plt.ylabel('SNR')
    plt.xlabel('Time (mS)')
    plt.legend()
    plt.savefig('snr.png')
    plt.show()

filename = 'instances/opt/plot_points_'+args.outputFile
with open(filename, "w") as output_plot:
    for m in range(m_bs):
        for n in range(n_ue):
            for t in range(scenario['simTime']):
                if x[m,n,t].x == 1:
                    output_plot.write(str(t)+','+str(m)+','+str(todb(SNR[m][n][t]))+'\n')

print('obj: %g'% model.objVal)
print('X: %g'% x.sum('*','*','*').getValue())
print('Y: %g'% y.sum('*','*','*').getValue())

'''
if __name__ == '__main__':
    1
'''

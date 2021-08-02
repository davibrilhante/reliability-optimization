#!/usr/bin/env python3
# -*- coding : utf8 -*-


import json
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import argparse
import sys
from matplotlib import pyplot as plt
import time

from decompressor import decompressor

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
parser.add_argument('-o','--outputFile', default='results', help='outputs json result file')
parser.add_argument('-p','--plot', action='store_true', help='Enables plot')
parser.add_argument('-s','--save', action='store_true', help='Save statistics')
parser.add_argument('-t','--threads', type=int, help='Number of threads', default=3)
parser.add_argument('--ttt', type=int, default=640)


args = parser.parse_args()

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
                kpi['deliveryRate']+=1
                #val = x[m][n_ue][t]*SNR[m][n_ue][t]
                val = x[m,n_ue,t].getAttr('X')*SNR[m][n_ue][t]
                linearSnr.append(val)
                snr.append(10*np.log10(val))
                cap.append(BW[m]*np.log2(1+val))

            if y[m,n_ue,t].getAttr('X')==1:
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
                        associated[2].append(SNR[m][n_ue][t] - 
                                SNR[associated[0][-1]][n_ue][t-args.ttt])
                    else:
                        associated[2].append(0)

                    associated[0].append(m)
                    associated[1].append(t)

    num = 0
    for m in range(m_bs):
        if associated[0].count(m)>1:
            num+= associated[0].count(m)-1
    rate = num/len(associated[0])

    kpi['handover'] = len(associated[0])
    kpi['handoverRate'] = kpi['handover']/simTime
    kpi['pingpong'] = rate 
    kpi['association'] = []
    for i in range(kpi['handover']):
        if i != (kpi['handover']-1):
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

#scenario['simTime'] = min(12000, scenario['simTime'])
for ue in nodes:
    ue['nPackets'] = int(scenario['simTime']/500 - 1)
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
offset = 3 
beta = []
for p in range(m_bs):
    beta.append([])
    for q in range(m_bs):
        beta[p].append([])
        for n in range(n_ue):
            beta[p][q].append([])
            temp = []
            for t in range(scenario['simTime']):
                if 10*np.log10(SNR[q][n][t]) >= 10*np.log10(SNR[p][n][t]) + offset:
                    temp.append(1) 
                else:
                    temp.append(0)
                if t>tau and sum(temp[t-tau:t])==tau:
                    beta[p][q][n].append(1)
                else:
                    beta[p][q][n].append(0)


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
x = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='x')
y = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='y')
z = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='z')
vars = x
end = time.time()
print(end - start)


### Add constraints to the model
print('Adding model Constraints...')
start = time.time()
#
# 1 - Capacity requirement constraint
#   - Blockage constraint added
for m in range(m_bs):
    for n,ue in enumerate(nodes):
        for t in range(scenario['simTime']):
            try:
                model.addConstr(ue['capacity']*x[m,n,t] <= 
                        R[m]*np.log2(1+SNR[m][n][t]), name='capcity_constr')
            except Exception as error:
                print('Error at constraint #1 %i %i %i'%(m, n, t))
                print(error)
                exit()

print('Constraints #1 added...')


# 2 - Delay requirement constraints
for m, bs in enumerate(network):
    for n,ue in enumerate(nodes):
        #for p, arrival in enumerate(ue['packets']):
        for p in range(ue['nPackets']):
            arrival = ue['packets'][p]
            #model.addConstr(sum(sum(x[m][n][k] for k in range(arrival,arrival+ue['delay'])) for m in range(m_bs))
            #        == sum(sum(y[m][n][k] for k in range(arrival,arrival+ue['delay'])) for m in range(m_bs)))

            try:
                model.addConstr(
                        sum(x[m,n,k] for k in range(arrival,arrival+ue['delay']))
                        == sum(y[m,n,k] for k in range(arrival,arrival+ue['delay'])),
                        name='delay_constr')
            except Exception as error:
                print('Error at constraint #2 %i %i %i'%(m,n,p))
                print(error)
                exit()

print('Constraints #2 added...')


# 3 - If the Received power is under the threshold, then the transmission cannot occur through this BS
#for m in range(m_bs):
for n,ue in enumerate(nodes):
    for t in range(scenario['simTime']):
        try:
            model.addConstr(sum((SNR[m][n][t] - ue['threshold'])*x[m,n,t] for m in range(m_bs))>= 0, name='snr_constr')
        except Exception as error:
            print('Error adding constraint #3 to the model %i %i'%(n,t))
            print(error)
            exit()

print('Constraints #3 added...')

bs_pairs = []
for i in range(m_bs):
    for j in range(m_bs):
        if i!=j:
            bs_pairs.append([i,j])

# 4 - Handover definition constraint
for p,q in bs_pairs:
    for n,ue in enumerate(nodes):

        #for k in range(ue['nPackets']):
            #arrival = ue['packets'][n]
            #if k < ue['nPackets'] - 1:

        for t1 in range(scenario['simTime']-1):
            #for t2 in range(ue['delay']):
            #    arrival2 = ue['packets'][k+1]
            t2 = t1 + 1 #tau
            #if beta[p][q][n][t2] != 0:  and beta[q][p][n][t2] != 0:
            try:
                '''
                model.addConstr(x[p,n,t1]*x[p,n,t2] +
                        beta[p][q][n][t2]*x[p,n,t1]*x[q,n,t2] +
                        beta[q][p][n][t2]*x[q,n,t1]*x[p,n,t2] +
                        x[q,n,t1]*x[q,n,t2] <= 1)
                model.addConstr(beta[p][q][n][t2]*x[p,n,t1]*x[q,n,t2] <= 1)
                '''
                model.addConstr((x[p,n,t1]*x[q,n,t2]) <= beta[p][q][n][t2], name='handover_constr')

            except Exception as error:
                print('Error adding constraint #4 to the model %i %i %i %i'%(p,q,n,t1))
                print(error)
                exit()

            '''
            model.addConstr(y[p][n][arrival+t1]*y[p][n][arrival2+t2] +
                    beta[p][q][n][arrival+t2]*y[p][n][arrival+t1]*y[q][n][arrival2+t2] +
                    beta[q][p][n][arrival+t2]*y[q][n][arrival+t1]*y[p][n][arrival2+t2] +
                    y[q][n][arrival+t1]*y[q][n][arrival2+t2] <= 1)
            '''

print('Constraints #4 added...')

# 5 - 
for n,ue in enumerate(nodes):
    #model.addConstr(sum(sum(x[m,n,t] for t in range(scenario['simTime'])) for m in range(m_bs)) <= scenario['simTime']) #ue['nPackets'])
    try:
        model.addConstr(sum(sum(y[m,n,t] for t in range(scenario['simTime'])) for m in range(m_bs)) <= ue['nPackets'], 
                        name='delay_pkt_constr')
    except Exception as error:
        print('Error adding constraint #5b to the model')
        print(error)
        exit()

print('Constraints #5 added...')
        

# 6 - Y can be equal to 1 only inside the delay interval
for n,ue in enumerate(nodes):
    for p in range(ue['nPackets']):
        arrival = ue['packets'][p]
        try:
            model.addConstr(sum(sum(y[m,n,t] for t in range(arrival,arrival+ue['delay'])) for m in range(m_bs)) <= 1, name='constr_6')
        except Exception as error:
            print('Error adding constraint #6 to the model %i %i'%(n,p))
            print(error)
            exit()

print('Constraints #6 added...')


# 7 - There is no transmission to an UE before it arrives, also y cannot be 1 if after the delay interval
for n,ue in enumerate(nodes):
    for p in range(ue['nPackets']):
        arrival = ue['packets'][p]
        # Y must be equal to 0 until the first packet arrives
        if p == 0:
            model.addConstr(sum(sum(y[m,n,t] for t in range(arrival)) for m in range(m_bs)) == 0, name='constr_7a')

        # From the arrival of the last packet plus the tolerable delay until 
        # the end of simulation Y must be equal to 0
        if p == ue['nPackets']-1:
            model.addConstr(sum(sum(y[m,n,t] for t in range(arrival+ue['delay'],scenario['simTime'])) for m in range(m_bs)) == 0,
                            name='constr_7b')

        # Between packets, Y must be equal to zero
        else:
            model.addConstr(sum(sum(y[m,n,t] for t in range(arrival+ue['delay'],ue['packets'][p+1])) for m in range(m_bs)) == 0,
                            name='constr_7c')


# 8 - LOS condition and UE association limit constraint. Each UE can be associate with only one BS
for n in range(n_ue):
    for t in range(scenario['simTime']):
        try:
            model.addConstr(sum(x[m,n,t] for m in range(m_bs)) <=1, name='constr_8')
            model.addConstr(sum(y[m,n,t] for m in range(m_bs)) <=1, name='constr_8')
        except Exception as error:
            print('Error adding constraint #8 to the model %i %i'%(n,t))
            print(error)
            exit()

print('Constraint #8 added...')


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
                    (x[m,n,t]+y[m,n,t]) 
                    #SNR[m][n][t]*(x[m][n][t]+y[m][n][t]) 
                    for t in range(scenario['simTime'])) 
                for n in range(n_ue)) 
            for m in range(m_bs)), 
        GRB.MAXIMIZE
        )

model._vars = vars
model._beta = gb.tuplelist(beta)

model.Params.lazyConstraints = 1

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

kpi = getKpi(x, y, m_bs, 0, scenario['simTime'], SNR, R, nodes[0]['nPackets'])

if args.save:
    filename = 'instances/opt/'+args.outfile
    with open(filename, 'w') as jsonfile:
        json.dump(kpi, jsonfile, indent=4)

results = json.dumps(kpi, indent=4)
#print(results)

############################## PLOT SECTION ###################################
if args.plot:
    print('Ploting SNR')
    plot = []
    for m in range(m_bs):
        plot.append([])
        time = []
        for t in range(scenario['simTime']):
            if x[m,0,t].getAttr('X') == 1:
                plot[-1].append(10*np.log10(SNR[m][0][t]))
                time.append(t)
                #print(t, SNR[m][0][t])

        #plt.plot(plot)
        plt.scatter(time,plot[-1], label=m)

    plt.ylabel('SNR')
    plt.xlabel('Time (mS)')
    plt.legend()
    plt.savefig('snr.png')
    plt.show()


print('obj: %g'% model.objVal)
print('X: %g'% x.sum('*','*','*').getValue())
print('Y: %g'% y.sum('*','*','*').getValue())

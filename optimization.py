#!/usr/bin/env python3
#

import json
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import argparse
import sys


def calc_recv(base, user, channel, t=0):
    # Evaluating the new position according with the vehicle speed (Change for vectorial speed)
    new_position_x = user['position']['x'] + (user['speed']['x']/3.6)*(t*1e-3)
    new_position_y = user['position']['y'] + (user['speed']['y']/3.6)*(t*1e-3)
    distance = np.hypot(base['position']['x'] - new_position_x, base['position']['y'] - new_position_y)
    wavelength = 3e8/base['frequency']

    bs_antenna_gain = 10
    ue_antenna_gain = 10
    exponent = channel['lossExponent']

    pl_0 = 20*np.log10(4*np.pi/wavelength)
    path_loss = pl_0 + 10*exponent*np.log10(distance) #- np.random.normal(0,8.96)
    return base['txPower'] + bs_antenna_gain + ue_antenna_gain - path_loss


def calc_snr(base, user, channel, t=0):
    noise_power = channel['noisePower']

    return calc_recv(base, user, channel, t) - noise_power



parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
parser.add_argument('-p','--plot', action='store_true', help='Enables plot')
args = parser.parse_args()

network = []
nodes = []


### Create base data
with open(args.inputFile) as json_file:
    data = json.load(json_file)
    scenario = data['scenario']
    channel = data['channel']
    LOS = data['blockage']
    gamma = data['gamma']
    for p in data['baseStation']:
        network.append(p)
    for p in data['userEquipment']:
        nodes.append(p)

n_ue = len(nodes)
m_bs = len(network)


SNR = []

for m, bs in enumerate(network):
    SNR.append([])
    bw_per_rb = 12*bs['subcarrierSpacing'] #12 subcarriers per resouce block times 120kHz subcarrier spacing
    for n,ue in enumerate(nodes):
        SNR[m].append([])
        #Adding the time dependency
        for t in range(scenario['simTime']):
            #SNR[m][n].append(bw_per_rb*calc_snr(bs,ue,channel))
            SNR[m][n].append(calc_snr(bs,ue,channel,t))


tau = 100
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
                if SNR[q][n][t] >= SNR[p][n][t] + offset:
                    temp.append(1)#model.addVar(vtype=GRB.BINARY, 
                else:
                    temp.append(0)
                if t>tau and sum(temp[t-tau:t])==tau:
                    beta[p][q][n].append(1)
                else:
                    beta[p][q][n].append(0)


R = []
for bs in network:
    R.append(bs['resourceBlocks'])


### Create environment and model
optEnv = gb.Env('myEnv.log')
model = gb.Model('newModel', optEnv)

### Quadratic constraints control
model.presolve().setParam(GRB.Param.PreQLinearize,1)



### Add variables to the model
w = []
x = []
y = []
for m in range(m_bs):
    w.append([])
    x.append([])
    y.append([])
    for n in range(n_ue):
        w[m].append([])
        x[m].append([])
        y[m].append([])
        for t in range(scenario['simTime']):
            w[m][n].append(model.addVar(vtype=GRB.BINARY, 
                        name='w'+str(m)+str(n)+str(t)))

            x[m][n].append(model.addVar(vtype=GRB.BINARY, 
                        name='x'+str(m)+str(n)+str(t)))

            y[m][n].append(model.addVar(vtype=GRB.BINARY, 
                        name='y'+str(m)+str(n)+str(t)))



### Add constraints to the model
#
# 1 - Capacity requirement constraint
#   - Blockage constraint added
for m in range(m_bs):
    for n,ue in enumerate(nodes):
        for t in range(scenario['simTime']):
            #model.addConstr(r[m][n][t]*LOS[m][n][t] 
            #        >= ue['capacity']*x[m][n][t])
            model.addConstr(R[m]*LOS[m][n][t]
                    >= ue['capacity']*x[m][n][t])



# 3 - LOS condition and UE association limit constraint. Each UE can be associate with only one BS
for n in range(n_ue):
    for t in range(scenario['simTime']):
        for m in range(m_bs):
            if LOS[m][n][t] == 1: 
                k=1
                break
            else: 
                k=0
        model.addConstr(sum(x[m][n][t] for m in range(m_bs)) <= k)
        model.addConstr(sum(y[m][n][t] for m in range(m_bs)) <= k)


# 4 - Delay requirement constraints
for n,ue in enumerate(nodes):
    for p, arrival in enumerate(ue['packets']):
        model.addConstr(sum(sum(x[m][n][k] for k in range(arrival,arrival+ue['delay'])) for m in range(m_bs))
                == sum(sum(y[m][n][k] for k in range(arrival,arrival+ue['delay'])) for m in range(m_bs)))


# 5 - Delay and Capacity requirements coupling
for m in range(m_bs):
    for n in range(n_ue):
        for t in range(scenario['simTime']):
            model.addConstr(2*w[m][n][t] == x[m][n][t]+y[m][n][t])

# 6 - There is no transmission to an UE before it arrives, also y cannot be 1 if after the delay interval
for n,ue in enumerate(nodes):
#    model.addConstr(sum(sum(y[m][n][t] for t in range(ue['arrival']+ue['delay'],scenario['simTime'])) for m in range(m_bs)) == 0)
    for p, arrival in enumerate(ue['packets']):
        model.addConstr(sum(sum(y[m][n][t] for t in range(arrival,arrival+ue['delay'])) for m in range(m_bs)) <= 1)

        if p == 0:
            model.addConstr(sum(sum(x[m][n][t] for t in range(arrival)) for m in range(m_bs)) == 0)
        else:
            model.addConstr(sum(sum(x[m][n][t] for t in range(ue['packets'][p-1]+ue['delay']+1,arrival)) for m in range(m_bs)) == 0)

        if p == ue['nPackets']-1:
            model.addConstr(sum(sum(y[m][n][t] for t in range(arrival+ue['delay'],scenario['simTime'])) for m in range(m_bs)) == 0)
        else:
            model.addConstr(sum(sum(y[m][n][t] for t in range(arrival+ue['delay'],ue['packets'][p+1])) for m in range(m_bs)) == 0)

# 7 - If the Received power is under the threshold, them the transmission cannot occur through this BS
for n,ue in enumerate(nodes):
    for t in range(scenario['simTime']):
        model.addConstr(sum((SNR[m][n][t] - ue['threshold'])*x[m][n][t]*LOS[m][n][t] for m in range(m_bs)) >= 0)

bs_pairs = []
for i in range(m_bs):
    for j in range(m_bs):
        if i!=j:
            bs_pairs.append([i,j])
# 8 - 
for p,q in bs_pairs:
    for n,ue in enumerate(nodes):
        for k,arrival  in enumerate(ue['packets']):
            if k < len(ue['packets'])- 1:
                for t1 in range(ue['delay']):
                    for t2 in range(ue['delay']):
                        arrival2 = ue['packets'][k+1]
                        model.addConstr(x[p][n][arrival+t1]*x[p][n][arrival2+t2] +
                                beta[p][q][n][arrival+t2]*x[p][n][arrival+t1]*x[q][n][arrival2+t2] +
                                beta[q][p][n][arrival+t2]*x[q][n][arrival+t1]*x[p][n][arrival2+t2] +
                                x[q][n][arrival+t1]*x[q][n][arrival2+t2] <= 1)


### Set objective function
#
# 1 - Maximize the number of users which delay and capacity requirements were
# fulfilled
model.setObjective(sum(sum(sum((1-gamma[m][n][t])*(x[m][n][t]+y[m][n][t]) for t in range(scenario['simTime'])) for n in range(n_ue)) for m in range(m_bs)), 
        GRB.MAXIMIZE)


#model.write('myModel.lp')

### Compute optimal Solution
try:
    model.optimize()
except gb.GurobiError:
    print('Optimize  failed')


### Print Info
print()
v = model.getVars()
n_var = 3
x = [[] for i in range(m_bs)]
y = [[] for i in range(m_bs)]
for t in range(scenario['simTime']):
    for n in range(n_ue):
        for m in range(m_bs):
            #print('Capacity: %d Delay: %d'%(nodes[n]['capacity'],nodes[n]['delay']))
            counter =  m*n_var*scenario['simTime'] + t*n_var
            #print('UE %d: %g %g %g %g %d'%(n, v[counter].x, v[counter+1].x, v[counter+2].x,v[counter+3].x, LOS[m][n][t]))
            #if t % 100 == 0:
            #    print('BS %d: %g %g %g %d'%(m, v[counter].x, v[counter+1].x, v[counter+2].x, SNR[m][n][t]))
            if v[counter+2].x == 1:
                x[m].append(20)#SNR[m][n][t])
                y[m].append(1)#SNR[m][n][t])
            else:
                x[m].append(-1)
                y[m].append(0)
            

print('obj: %g'% model.objVal)


##################### Collecting network results ##############################
print('\n\n\n@@@@@')

#Calculating number of handovers
associated = []
for t in range(scenario['simTime']):
    for m in range(m_bs):
        if y[m][t] == 1:
            if (len(associated) > 0 and associated[-1] != m) or len(associated)==0:
                associated.append(m)
print(len(associated))
            
#Calculating Ping Pong Rate
num = 0
for m in range(m_bs):
    if associated.count(m)>0:
        num+= 1 - associated.count(m)
rate = num/len(associated)
print(rate)

#Calculating Capacity achived/Throughput
throughput = []
for t in range(scenario['simTime']):
    for m in range(m_bs):
        if y[m][t] == 1:
            throughput.append(12*network[m]['subcarrierSpacing']*R[m]*np.log2(1+SNR[m][0][t]))
print(np.mean(throughput))


#Calculating Packets Lost
packetsSent = []
for n,ue in enumerate(nodes):
    temp = []
    for m in range(m_bs):
        temp.append(y[m].count(1))
    packetsSent.append(1 - sum(temp)/ue['nPackets'])
print(np.mean(packetsSent))

############################## PLOT SECTION ###################################
if args.plot:
    from matplotlib import pyplot as plt
    plos = []
    for m in range(m_bs):
        plos.append([])
        for i in LOS[m][0]:
            if i == 1:
                plos[m].append(m+1)
            else:
                plos[m].append(-1)

    colors = ['b','g','orange','r']
    for m in range(m_bs):
        plt.plot(SNR[m][0], label='BS '+str(m), color=colors[m])
        plt.scatter(range(scenario['simTime']),x[m], color=colors[m], marker='o', s=8)
        plt.scatter(range(scenario['simTime']),plos[m], color=colors[m], marker='o', s=8)


    plt.ylabel('SNR')
    plt.xlabel('Time (mS)')
    plt.legend()
    plt.ylim(0,25)
    plt.show()

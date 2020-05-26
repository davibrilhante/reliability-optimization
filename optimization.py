#!/usr/bin/env python3

import json
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import argparse
import sys


def calc_snr(base, user, channel):
    distance = np.hypot(base['position']['x'] - user['position']['x'], base['position']['y'] - user['position']['y'])
    wavelength = 3e8/base['frequency']

    exponent = channel['lossExponent']
    noise_power = channel['noisePower']

    pl_0 = 20*np.log10(4*np.pi/wavelength)
    path_loss = pl_0 + 10*exponent*np.log10(distance) #- np.random.normal(0,8.96)
    power_linear = 10*((base['txPower'] - path_loss)/10)
    noise_power = 10*(noise_power/10)
    return np.log2(1 + (power_linear/noise_power))


parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
args = parser.parse_args()

network = []
nodes = []


### Create base data
with open(args.inputFile) as json_file:
    data = json.load(json_file)
    scenario = data['scenario']
    channel = data['channel']
    LOS = data['blockage']
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
            SNR[m][n].append(bw_per_rb*calc_snr(bs,ue,channel))

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
r = []
for m in range(m_bs):
    w.append([])
    x.append([])
    y.append([])
    r.append([])
    for n in range(n_ue):
        w[m].append([])
        x[m].append([])
        y[m].append([])
        r[m].append([])
        for t in range(scenario['simTime']):
            w[m][n].append(model.addVar(vtype=GRB.BINARY, 
                        name='w'+str(m)+str(n)+str(t)))

            x[m][n].append(model.addVar(vtype=GRB.BINARY, 
                        name='x'+str(m)+str(n)+str(t)))

            y[m][n].append(model.addVar(vtype=GRB.BINARY, 
                        name='y'+str(m)+str(n)+str(t)))

            r[m][n].append(model.addVar(vtype=GRB.INTEGER, 
                        name='r'+str(m)+str(n)+str(t)))


### Add constraints to the model
#
# 1 - Capacity requirement constraint
#   - Blockage constraint added
for m in range(m_bs):
    for n,ue in enumerate(nodes):
        for t in range(scenario['simTime']):
            #model.addConstr(r[m][n][t]*SNR[m][n][t]*LOS[m][n][t] 
            #        >= ue['capacity']*x[m][n][t])
            model.addConstr(r[m][n][t]*LOS[m][n][t] 
                    >= ue['capacity']*x[m][n][t])

# 2 - Resource Blocks boudaries constraints
for m in range(m_bs):
    for t in range(scenario['simTime']):
        #model.addConstr(sum(r[m][n][t]*x[m][n][t] for n in range(n_ue)) <= R[m])
        model.addConstr(sum(r[m][n][t] for n in range(n_ue)) <= R[m])

#2.1 - A strongest contraint is needed to force r[m][n][t] = 0, when x[m][n][t] = 0 

#2.2 - Forcing a "worst case" round robin
for n, ue in enumerate(nodes):
    model.addConstr(sum(sum(x[m][n][t] for t in range(ue['arrival'],scenario['simTime'])) for m in range(m_bs)) <= 1)

# 3 - UE association limit constraint. Each UE can be associate with only one BS
for n in range(n_ue):
    for t in range(scenario['simTime']):
        model.addConstr(sum(x[m][n][t] for m in range(m_bs)) <= 1)
        model.addConstr(sum(y[m][n][t] for m in range(m_bs)) <= 1)

# 4 - Delay requirement constraints
for n,ue in enumerate(nodes):
    model.addConstr(sum(sum(x[m][n][k]*LOS[m][n][k] for k in range(ue['arrival'],ue['arrival']+ue['delay'])) for m in range(m_bs))
            == sum(sum(y[m][n][k] for k in range(ue['arrival'],ue['arrival']+ue['delay'])) for m in range(m_bs)))

# 5 - Delay and Capacity requirements coupling
for m in range(m_bs):
    for n in range(n_ue):
        for t in range(scenario['simTime']):
            model.addConstr(2*w[m][n][t] == x[m][n][t]+y[m][n][t])

# 6 - There is no transmission to an UE before it arrives, also y cannot be 1 if after the delay interval
for n,ue in enumerate(nodes):
    model.addConstr(sum(sum(x[m][n][t] for t in range(ue['arrival'])) for m in range(m_bs)) == 0)
    model.addConstr(sum(sum(y[m][n][t] for t in range(ue['arrival']+ue['delay'],scenario['simTime'])) for m in range(m_bs)) == 0)




### Set objective function
#
# 1 - Maximize the number of users which delay and capacity requirements were
# fulfilled
model.setObjective(sum(sum(sum(x[m][n][t]+y[m][n][t] for t in range(scenario['simTime'])) for n in range(n_ue)) for m in range(m_bs)), 
        GRB.MAXIMIZE)


### Compute optimal Solution
try:
    model.optimize()
except gb.GurobiError:
    print('Optimize  failed')


### Print Info
print()
v = model.getVars()
'''
for n in range(n_ue):
    print('UE%d C %d D %d' % (n, nodes[n]['capacity'], nodes[n]['delay']))
    for m in range(m_bs):
        print('BS%d' % m, end=' ')
        counter = m*640 + n*40
        for t in range(scenario['simTime']):
            #print('%s %g %s %g %s %g %s %g B%d%d%d %d D%d %d'%(v[counter].varName, v[counter].x, v[counter+1].varName, v[counter+1].x, 
            #    v[counter+2].varName, v[counter+2].x, v[counter+3].varName, v[counter+3].x, m, n, t, LOS[m][n][t], n, nodes[n]['delay']))
            print('%g %g %g %g %d'%(v[counter].x, v[counter+1].x, v[counter+2].x,v[counter+3].x, LOS[m][n][t]), end=' ')
            counter+=4
            if t == scenario['simTime']-1:
                print()
'''
for t in range(scenario['simTime']):
    print()
    print('====================================')
    print()
    for m in range(m_bs):
        for n in range(n_ue):
            #print('Capacity: %d Delay: %d'%(nodes[n]['capacity'],nodes[n]['delay']))
            counter = m*400 + n*40 + t*4 
            print('UE %d: %g %g %g %g %d'%(n, v[counter].x, v[counter+1].x, v[counter+2].x,v[counter+3].x, LOS[m][n][t]))


model.write('myModel.lp')
print('obj: %g'% model.objVal)

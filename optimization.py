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


network = []
nodes = []


### Create base data
with open('instances/instance-1-4.json') as json_file:
    data = json.load(json_file)
    scenario = data['scenario']
    channel = data['channel']
    B = data['blockage']
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

for i in range(m_bs):
    print(B[i])

### Create environment and model
optEnv = gb.Env('myEnv.log')
model = gb.Model('newModel', optEnv)


### Add variables to the model
x = []
r = []
for m in range(m_bs):
    x.append([])
    r.append([])
    for n in range(n_ue):
        x[m].append([])
        r[m].append([])
        for t in range(scenario['simTime']):
            x[m][n].append(model.addVar(vtype=GRB.BINARY, 
                        name='x'+str(m)+str(n)+str(t)))

            r[m][n].append(model.addVar(vtype=GRB.INTEGER, 
                        name='r'+str(m)+str(n)+str(t)))


### Add constraints to the model
#
# 1 - Capacity requirement constraint
#   - Blockage constraint added
for n,ue in enumerate(nodes):
    for t in range(scenario['simTime']):
        model.addConstr(sum(r[m][n][t]*SNR[m][n][t]*(1-B[m][n][t]) for m in range(m_bs)) >= ue['capacity'][t]*x[m][n][t])

# 2 - Resource Blocks boudaries constraints
for m, bs in enumerate(network):
    for t in range(scenario['simTime']):
        model.addConstr(sum(r[m][n][t]*x[m][n][t] for n in range(n_ue)) <= R[m])

# 3 - UE association limit constraint. Each UE can be associate with only one BS
for n, ue in enumerate(nodes):
    for t in range(scenario['simTime']):
        model.addConstr(sum(x[m][n][t] for m in range(m_bs)) <= 1)

# 4 - Blockage constraint: it prevents that a link from being schedule when there is blockage
#for m in range(m_bs):
#   for n in range(n_ue):
#       for t in range(scenario['simTime']):
#           model.addConstr(x[m][n][t] <= (1 - B[m][n][t]))


### Set objective function
model.setObjective(sum(sum(sum(x[m][n][t] for t in range(scenario['simTime'])) for n in range(n_ue)) for m in range(m_bs)), GRB.MAXIMIZE)


### Compute optimal Solution
try:
    model.optimize()
except gb.GurobiError:
    print('Optimize  failed')


### Print Info
print()
counter = 0
v = model.getVars()
for m in range(m_bs):
    for n in range(n_ue):
        for t in range(scenario['simTime']):
            print('%s %g %s %g B%d%d%d %d'%(v[counter].varName,v[counter].x,v[counter+1].varName,v[counter+1].x, m,n,t,B[m][n][t]))
            counter+=2

print('obj: %g'% model.objVal)

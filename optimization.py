#!/usr/bin/env python3

import json
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import argparse
import sys


def calc_snr(base, user, scenario):
    distance = np.hypot(base['position']['x'] - user['position']['x'], base['position']['y'] - user['position']['y'])
    wavelength = 3e8/base['frequency']

    exponent = scenario['lossExponent']
    noise_power = scenario['noisePower']

    pl_0 = 20*np.log10(4*np.pi/wavelength)
    path_loss = pl_0 + 10*exponent*np.log10(distance) #- np.random.normal(0,8.96)
    power_linear = 10*((base['txPower'] - path_loss)/10)
    noise_power = 10*(noise_power/10)
    return np.log2(1 + (power_linear/noise_power))


network = []
nodes = []

parser = argparse.ArgumentParser()



### Create base data
with open('instances/test.json') as json_file:
    data = json.load(json_file)
    channel = data['channel']
    for p in data['baseStation']:
        network.append(p)
    for p in data['userEquipment']:
        nodes.append(p)



SNR = []
for m, bs in enumerate(network):
    SNR.append([])
    for ue in nodes:
        bw_per_rb = 12*bs['subcarrierSpacing'] #12 subcarriers per resouce block times 120kHz subcarrier spacing
        SNR[m].append(bw_per_rb*calc_snr(bs,ue,channel))

R = []
for bs in network:
    R.append(bs['resourceBlocks'])


### Create environment and model
optEnv = gb.Env('myEnv.log')
model = gb.Model('newModel', optEnv)


### Add variables to the model
x = []
r = []
for m, bs in enumerate(network):
    x.append([])
    r.append([])
    print(x, m)
    for n, ue in enumerate(nodes):
        print(n)
        x[m].append(model.addVar(vtype=GRB.BINARY, 
                    name='x'+str(m)+str(n)))

        r[m].append(model.addVar(vtype=GRB.INTEGER, 
                    name='r'+str(m)+str(n)))


### Add constraints to the model
#
# 1 - Capacity requirement constraint
for n,ue in enumerate(nodes):
    model.addConstr(sum(r[m][n]*SNR[m][n]*x[m][n] for m, bs in enumerate(network)) >= ue['capacity'])

# 2 - Resource Blocks boudaries constraints
for m, bs in enumerate(network):
    model.addConstr(sum(r[m][n]*x[m][n] for n, ue in enumerate(nodes)) <= R[m])

# 3 - UE association limit constraint
for n, ue in enumerate(nodes):
    model.addConstr(sum(x[m][n] for m, bs in enumerate(network)) <= 1)


### Set objective function
model.setObjective(sum(sum(x[m][n] for n, ue in enumerate(nodes)) for m, bs in enumerate(network)), GRB.MAXIMIZE)


### Compute optimal Solution
try:
    model.optimize()
except gb.GurobiError:
    print('Optimize  failed')


### Print Info
print()
for v in model.getVars():
    print('%s %g'%(v.varName,v.x))
print('obj: %g'% model.objVal)

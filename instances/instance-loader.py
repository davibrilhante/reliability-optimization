#!/usr/bin/env python3

import json
import uuid
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-s","--seed", type=int, required=False, default=1)
parser.add_argument("--vx", type=int, required=False, default=0)
parser.add_argument("--vy", type=int, required=False, default=90)
parser.add_argument("-b","--blockageDensity", type=float, required=False, default=0.001)
parser.add_argument("-t","--simTime", type=int, required=False, default=12000)
parser.add_argument("--blockage2", required=False, action='store_true')


args = parser.parse_args()
seed = args.seed

sqrt2 = np.sqrt(2)
np.random.seed(seed)

f = open('inst-4-1-24')
lines = f.readlines()
for i,j in enumerate(lines):
    lines[i] = lines[i].strip()


simTime = int(300*3.6*1e3/args.vy)
#print(simTime)

data = {}
data['scenario'] = {
    'simTime': simTime, #int(lines[0]),
    'boundaries': {
        'xmin': int(lines[1]),
        'ymin': int(lines[2]),
        'xmax': int(lines[3]),
        'ymax': int(lines[4])
        }
    }

#data['channel'] = []
data['channel'] = {
    'lossExponent': float(lines[5]),
    'noisePower': float(lines[6]),
    }

n_BS = 4
n_UE = 1
n_PKT = 24

data['baseStation'] = []
for i in range(n_BS):
    counter = i*6 + 8
    data['baseStation'].append({
        'uuid': str(uuid.uuid4()),
        'resourceBlocks': int(lines[counter]),
        'frequency': float(lines[counter+1]),
        'subcarrierSpacing': (2**int(lines[counter+2]))*15e3,
        'txPower': 30, #dBm
        'position': {
            'x': int(lines[counter+3]),
            'y': int(lines[counter+4])
            }
        })

data['userEquipment'] = []
for i in range(n_UE):
    counter = i*7 + n_BS*6 + 8
    data['userEquipment'].append({
    'uuid': str(uuid.uuid4()),
    'capacity': int(lines[counter]),
    'delay': int(lines[counter+1]), #classOfService[userType][1],
    'speed': {
        'x': args.vx,
        'y': args.vy #int(lines[counter+2])
        },
    'position': {
        'x': int(lines[counter+3]),
        'y': int(lines[counter+4])
        },
    'threshold': 20,
    'nPackets': int(lines[counter+5]), 
    'packets': [int(lines[counter+7+j])  for j in range(int(lines[counter+5]))]
    })

from blockage import blockage, blockage2
density = args.blockageDensity

if args.blockage2:
    data['blockage'], data['gamma'] = blockage2(density, data['scenario'], data['baseStation'], data['userEquipment'])
else:
    data['blockage'], data['gamma'] = blockage(density, data['scenario'], data['baseStation'], data['userEquipment'])


'''
data['blockage'] = []
for m in range(n_BS):
    data['blockage'].append([])
    for n in range(n_UE):
        data['blockage'][m].append([])
        for t in range(data['scenario']['simTime']):
            if m==0 and t>=1200 and t<= 2000:
                data['blockage'][m][n].append(0)
            else:
                data['blockage'][m][n].append(1)

'''

f.close()

outname = 'mobility.json'
with open(outname, 'w') as outfile:
    json.dump(data, outfile, indent=4)

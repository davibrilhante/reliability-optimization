#!/usr/bin/env python3
# -*- coding : utf8 -*-

import json
import uuid
import argparse
import sys
import numpy as np

from blockage import blockage, blockage2

parser = argparse.ArgumentParser()
parser.add_argument("-s","--seed", type=int, required=False, default=1)
parser.add_argument("--vx", type=int, required=False, default=0)
parser.add_argument("--vy", type=int, required=False, default=90)
parser.add_argument("-b","--blockageDensity", help='obstacles density (obstacle/m2)',type=float, required=False, default=0.001)
parser.add_argument("-t","--simTime", type=int, required=False, default=12000)
parser.add_argument("-u","--ue", type=int, required=False, default=1)
parser.add_argument("-c","--bs", type=float, help='BS density (base station/km2)', required=False, default=0.05)
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

density = args.blockageDensity

data = {}
data['scenario'] = {
    'simTime': simTime, #int(lines[0]),
    'boundaries': {
        'xmin': int(lines[1]),
        'ymin': int(lines[2]),
        'xmax': int(lines[3]),
        'ymax': int(lines[4])
        },
    'blockageDensity': density,
    'seed':seed
    }

#data['channel'] = []
data['channel'] = {
    'lossExponent': float(lines[5]),
    'noisePower': float(lines[6]),
    }

area = (int(lines[3])-int(lines[1]))*(int(lines[4])-int(lines[2]))/1e3 #in Km2
bs_row = int(np.sqrt(args.bs*area))
n_BS = bs_row**2
n_UE = args.ue
n_PKT = 24

counter = 8
data['baseStation'] = []
for i in range(bs_row):
    x = int(lines[1])+(i+0.5)*((int(lines[3])-int(lines[1]))/bs_row)
    for j in range(bs_row):
        y = int(lines[2])+(j+0.5)*((int(lines[4])-int(lines[2]))/bs_row)
        #counter = (i+j*2)*6 + 8
        data['baseStation'].append({
            'uuid': str(uuid.uuid4()),
            'resourceBlocks': int(lines[counter]),
            'frequency': float(lines[counter+1]),
            'subcarrierSpacing': (2**int(lines[counter+2]))*15e3,
            'txPower': 30, #dBm
            'position': {
                'x': x, #int(lines[counter+3]),
                'y': y #int(lines[counter+4])
                }
            })

counter += 4*6
occupation = [0,0]
data['userEquipment'] = []
for i in range(n_UE):
    #counter = i*7 + n_BS*6 + 8
    arrival = np.random.poisson(500,int(lines[counter+5])).tolist()
    lane = np.random.choice([0,1]) # A lane is 5 meter apart from the centre
    occupation[lane] += 1

    data['userEquipment'].append({
    'uuid': str(uuid.uuid4()),
    'capacity': int(lines[counter]),
    'delay': int(lines[counter+1]), #classOfService[userType][1],
    'speed': {
        'x': args.vx,#
        'y': args.vy #
        },
    'position': {
        # A lane is 5 meter apart from the centre
        'x': int(lane*10 - 5),  #int(lines[counter+3])
        #2 seconds is a safe distance between vehicles
        'y': int(lines[counter+4]) + i*np.random.poisson(2*args.vy/3.6)
        },
    'threshold': 20,
    'nPackets': int(lines[counter+5]), 
    #'packets': [int(lines[counter+7+j])  for j in range(int(lines[counter+5]))]
    'packets': [sum(arrival[:i]) for i in range(int(lines[counter+5]))]
    })


if args.blockage2:
    data['blockage'], data['gamma'] = blockage2(density, data['scenario'], data['baseStation'], data['userEquipment'])
else:
    data['blockage'], data['gamma'] = blockage(density, data['scenario'], data['baseStation'], data['userEquipment'])



f.close()

outname = str(n_BS)+'-'+str(n_UE)+'-'+str(density)+'-'+str(args.vx)+'-'+str(args.vy)+'-'+str(args.seed) #'mobility.json'
with open(outname, 'w') as outfile:
    json.dump(data, outfile, indent=4)
#y = json.dumps(data, indent=4)
#print(y)

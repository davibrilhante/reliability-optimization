#!/usr/bin/env python3
# -*- coding : utf8 -*-

import json
import uuid
import argparse
import sys
import numpy as np

from blockage import blockage, blockage2
from bsplacement import hexagonalbsplacement
from shapely.geometry import LineString

parser = argparse.ArgumentParser()
parser.add_argument("-s","--seed", type=int, required=False, default=1)
parser.add_argument("--vx", type=int, required=False, default=0)
parser.add_argument("--vy", type=int, required=False, default=90)
parser.add_argument("-b","--blockageDensity", help='obstacles density (obstacle/m2)',type=float, required=False, default=0.001)
parser.add_argument("-t","--simTime", type=int, required=False, default=12000)
parser.add_argument("-u","--ue", type=int, required=False, default=1)
parser.add_argument("-r","--bsradius", type=int, required=False, default=150)
parser.add_argument("-d","--uedelay", type=int, required=False, default=2)
parser.add_argument("-c","--uecapacity", type=float, required=False, default=750e6)
parser.add_argument("--blockage2", required=False, action='store_true')



args = parser.parse_args()
seed = args.seed

sqrt2 = np.sqrt(2)
np.random.seed(seed)

f = open('inst-4-1-24')
#f = open('toy2')
lines = f.readlines()
for i,j in enumerate(lines):
    lines[i] = lines[i].strip()


simTime = args.simTime #min(args.simTime, int(300*3.6*1e3/args.vy))

density = args.blockageDensity

data = {}
data['scenario'] = {
    'simTime': simTime,#int(lines[0]),
    'boundaries': {
        'xmin': int(lines[1]),
        'ymin': int(lines[2]),
        'xmax': int(lines[3]),
        'ymax': int(lines[4])
        },
    'blockageDensity': density,
    'seed':seed
    }

data['channel'] = {
    'lossExponent': float(lines[5]),
    'noisePower': float(lines[6]),
    }


## for an hexagonal grid
width = data['scenario']['boundaries']['xmax'] - data['scenario']['boundaries']['xmin']
height = data['scenario']['boundaries']['ymax'] - data['scenario']['boundaries']['ymin']
bs_radius = args.bsradius

route = LineString([(0,0),(1200,1200)])
bs_coords = hexagonalbsplacement(width, height, bs_radius, route)

n_BS = len(bs_coords)
n_UE = args.ue

counter = 8
data['baseStation'] = []
for m, coord in enumerate(bs_coords):
    counter = 8
    data['baseStation'].append({
        'index': m,
        'uuid': str(uuid.uuid4()),
        'resourceBlocks': int(lines[counter]),
        'frequency': float(lines[counter+1]),
        'subcarrierSpacing': (2**int(lines[counter+2]))*15e3,
        'txPower': 30, #dBm
        'position': {
            'x': coord[0], #int(lines[counter+3]), #x,
            'y': coord[1]  #int(lines[counter+4])#y
            }
        })

occupation = [0,0]
data['userEquipment'] = []
for i in range(n_UE):
    Lambda = 120
    nPackets = int((data['scenario']['simTime']/Lambda) - 1)
    arrival = np.random.poisson(Lambda,nPackets).tolist() #int(lines[counter+5])).tolist()
    lane = np.random.choice([0,1]) # A lane is 5 meter apart from the centre
    occupation[lane] += 1

    data['userEquipment'].append({
    'uuid': str(uuid.uuid4()),
    'capacity': args.uecapacity, #int(lines[counter]),
    'delay': args.uedelay, #int(lines[counter+1]), #classOfService[userType][1],
    'speed': {
        'x': args.vx,#
        'y': args.vy #
        },
    'position': {
        # A lane is 5 meter apart from the centre
        'x': 0, #int(lane*10 - 5),  #int(lines[counter+3])
        #2 seconds is a safe distance between vehicles
        'y': 0 #int(lines[counter+4]) + i*np.random.poisson(2*args.vy/3.6)
        },
    'threshold': 20,
    'nPackets': nPackets, #int(lines[counter+5]), 
    #'packets': [int(lines[counter+7+j])  for j in range(int(lines[counter+5]))]
    'packets': [sum(arrival[:i+1]) for i in range(nPackets)]#int(lines[counter+5]))]
    })

'''
outname = 'scenario-'+str(n_BS)+'-'+str(n_UE)+'-'+str(density)+'-'+str(args.vx)+'-'+str(args.vy)+'-'+str(args.seed) #'mobility.json'
with open(outname, 'w') as outfile:
    json.dump(data, outfile, indent=4)

outfile.close()

'''
data2 = {}
if args.blockage2:
    data['blockage'], data['gamma'] = blockage2(density, data['scenario'], data['baseStation'], data['userEquipment'])

elif density > 0:
    #data['blockage'], data['gamma'] = blockage(density, data['scenario'], data['baseStation'], data['userEquipment'])
    data['blockers'], data['blockage'] = blockage(density, data['scenario'], 
                                data['baseStation'], data['userEquipment'], tolerance = 2*bs_radius)

else:
    data['blockage'] = []
    for j in range(n_BS):
        data['blockage'].append([])
        data['blockage'][j].append([])
        for i in range(data['scenario']['simTime']):
                data['blockage'][j][0].append(1)
            #if ((i < data['scenario']['simTime']/4)
            #        or (i > data['scenario']['simTime']/2 and i <=  3*data['scenario']['simTime']/4)):
            #else:
            #    data['blockage'][j][0][i] = 1

#f.close()

#outname = str(n_BS)+'-'+str(n_UE)+'-'+str(density)+'-'+str(args.vx)+'-'+str(args.vy)+'-'+str(args.seed) #'mobility.json'
#with open(outname, 'w') as outfile:
#    json.dump(data, outfile, indent=4)
y = json.dumps(data, indent=4)
print(y)

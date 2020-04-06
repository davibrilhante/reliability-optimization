#!/usr/bin/env python3

import json
import uuid
import argparse
import sys
import numpy as np

sqrt2 = np.sqrt(2)

parser = argparse.ArgumentParser()

parser.add_argument('-b','--baseStations', help='Number of Base Stations', default=1)
parser.add_argument('-u','--userEquipments', help='Number of User Equipments', default=4)

parser.add_argument('-e','--lossExponent', help='Path loss exponent', default=3.41, type=float, required=False)
parser.add_argument('-s','--subcarrierSpacing', help='5G millimeter waves subcarrier spacing numerology',
                    choices=[2, 3, 4], type=int, required=False)
parser.add_argument('-n','--noise-power', help='Channel noise power', type=float, required=False)
parser.add_argument('--xmin', help='Scenario x min', type=float, required=False, default=-100.0)
parser.add_argument('--ymin', help='Scenario y min', type=float, required=False, default=-100.0)
parser.add_argument('--xmax', help='Scenario x max', type=float, required=False, default=100.0)
parser.add_argument('--ymax', help='Scenario y max', type=float, required=False, default=100.0)

parser.add_argument('-R','--resource-blocks', default=275, type=int, required=False)
parser.add_argument('-f','--frequency', default=30e9, type=float, required=False)
parser.add_argument('-O','--outputFile', default='instance.json', required=False)
parser.add_argument('-I','--inputFile', required=False)
parser.add_argument('-r','--notRandomUe', default=False, type=bool, required=False, action='store_true')
#The input file must contain at least the base station positions

args = parser.parse_args()

data = {}
data['channel'] = []
data['channel'] = {
    'lossExponent': args.loss-exponent,
    'noisePower': args.noise-power,
    'boundaries': {
        'xmin': args.xmin,
        'ymin': args.ymin,
        'xmax': args.xmax,
        'ymax': args.ymax
        }
    }

data['baseStation'] = []
if args.baseStations == 1 and args.inputFile == None:
    data['baseStation'].append({
        'uuid': str(uuid.uuid4()),
        'resourceBlocks': args.resource-blocks,
        'frequency': args.frequency,
        'subcarrierSpacing': args.subcarrier-spacing,
        'txPower': 30, #dBm
        'position': {
            'x': args.xmin + (args.xmax-args.xmin)/2,
            'y': args.ymin + (args.ymax-args.ymin)/2
            }
        })

elif args.baseStations > 1 and args.inputFile == None:
    print('Base Station position files not specified')
    sys.exit()

elif args.baseStations > 1 and args.inputFile != None:
    1
    

data['userEquipment'] = []
pos = []

if not args.notRandomUe: 
    for i in range(args.userEquipments):
        x = np.random.uniform(args.xmin,args.xmax)
        y = np.random.uniform(args.ymin,args.ymax)
        while pos.count([x,y]) != 0:
            x = np.random.uniform(args.xmin,args.xmax)
            y = np.random.uniform(args.ymin,args.ymax)

        cap = np.random.choice([1e5,1e6,1e7,1e8])*np.random.normal(5,2.5)

        data['userEquipment'].append({
            'uuid': str(uuid.uuid4()),
            'capacity': cap,
            'position': {
                'x': x,
                'y': y
                }
            })
elif args.inputFile != None and args.notRandomUe:
    1

elif args.inputFile == None and args.notRandomUe:
    print('No user equipment data specified. Set -r/--notRandomUe or provide an input file with position and capacity')
    sys.exit()

with open(args.outputFile, 'w') as outfile:
    json.dump(data, outfile, indent=4)

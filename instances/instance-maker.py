#!/usr/bin/env python3

import json
import uuid
import argparse
import sys
import numpy as np

sqrt2 = np.sqrt(2)

parser = argparse.ArgumentParser()

parser.add_argument('-b','--baseStations', help='Number of Base Stations', type=int, default=1)
parser.add_argument('-u','--userEquipments', help='Number of User Equipments', type=int, default=4)
parser.add_argument('-e','--lossExponent', help='Path loss exponent', default=3.41, type=float, required=False)
parser.add_argument('-s','--subcarrierSpacing', help='5G millimeter waves subcarrier spacing numerology',
                    choices=[2, 3, 4], type=int, required=False, default=3)
parser.add_argument('-n','--noisePower', help='Channel noise power', type=float, required=False, default=-80)
parser.add_argument('-t','--simTime', help='Simulation time in number of time slots',
                    type=int, required=False, default=10)
parser.add_argument('--xmin', help='Scenario x min', type=float, required=False, default=-100.0)
parser.add_argument('--ymin', help='Scenario y min', type=float, required=False, default=-100.0)
parser.add_argument('--xmax', help='Scenario x max', type=float, required=False, default=100.0)
parser.add_argument('--ymax', help='Scenario y max', type=float, required=False, default=100.0)
parser.add_argument('--seed', help='Seed for the random number generator', required=False, default=1, type=int)
parser.add_argument('-B','--blockage', help='Activate random blockage between the BS-UE links', default=False, 
        required=False, action='store_true')
parser.add_argument('-R','--resourceBlocks', default=10, type=int, required=False)
parser.add_argument('-f','--frequency', default=30e9, type=float, required=False)
parser.add_argument('-O','--outputFile', required=False)
#The input file must contain at least the base station positions
parser.add_argument('-I','--inputFile', help='Input file with at least base station positions (x and y, one per line)',
                    required=False)
parser.add_argument('--notRandomUe', default=False, required=False, action='store_true')
parser.add_argument('--staticCapacity', default=False, required=False, action='store_true')

args = parser.parse_args()
np.random.seed(args.seed)


data = {}
data['scenario'] = {
    'simTime': args.simTime,
    'boundaries': {
        'xmin': args.xmin,
        'ymin': args.ymin,
        'xmax': args.xmax,
        'ymax': args.ymax
        }
    }

data['channel'] = []
data['channel'] = {
    'lossExponent': args.lossExponent,
    'noisePower': args.noisePower,
    }

data['baseStation'] = []
if args.baseStations == 1 and args.inputFile == None:
    data['baseStation'].append({
        'uuid': str(uuid.uuid4()),
        'resourceBlocks': args.resourceBlocks,
        'frequency': args.frequency,
        'subcarrierSpacing': (2**args.subcarrierSpacing)*15e3,
        'txPower': 30, #dBm
        'position': {
            'x': args.xmin + (args.xmax-args.xmin)/4,
            'y': args.ymin + (args.ymax-args.ymin)/2
            }
        })
    data['baseStation'].append({
        'uuid': str(uuid.uuid4()),
        'resourceBlocks': args.resourceBlocks,
        'frequency': args.frequency,
        'subcarrierSpacing': (2**args.subcarrierSpacing)*15e3,
        'txPower': 30, #dBm
        'position': {
            'x': args.xmax - (args.xmax-args.xmin)/4,
            'y': args.ymin + (args.ymax-args.ymin)/2
            }
        })

elif args.baseStations > 1 and args.inputFile == None:
    print('Base Station position files not specified')
    sys.exit()

elif args.baseStations > 1 and args.inputFile != None:
    with open(args.inputFile) as inputFile:
        for bs in range(args.baseStations): 
            line = inputFile.readline()
            pos = line.strip('\n').split(' ')
            data['baseStation'].append({
                'uuid': str(uuid.uuid4()),
                'resourceBlocks': args.resourceBlocks,
                'frequency': args.frequency,
                'subcarrierSpacing': (2**args.subcarrierSpacing)*15e3,
                'txPower': 30, #dBm
                'position': {
                    'x': float(pos[0]),
                    'y': float(pos[1])
                    }
                })

    

data['userEquipment'] = []
pos = []

if not args.notRandomUe: 
    for i in range(args.userEquipments):
        x = np.random.uniform(args.xmin,args.xmax)
        y = np.random.uniform(args.ymin,args.ymax)
        while pos.count([x,y]) != 0:
            x = np.random.uniform(args.xmin,args.xmax)
            y = np.random.uniform(args.ymin,args.ymax)

        '''
        if args.staticCapacity:
            #The capacity requirement does not varies with the time
            cap = [np.random.choice([0,1e5,5e5,1e6,5e6,1e7])*abs(np.random.normal(5,2.5))]*args.simTime

        else:
            cap = [np.random.choice([0,1e5,5e5,1e6,5e6,1e7])*abs(np.random.normal(5,2.5)) for i in range(args.simTime)]

        delay = np.random.randint(1,5)
        '''

        
        classOfService = [[500e3, 1000],[500e3, 300], [1e6, 100], [5e6, 100],[1e7, 10]]
        userType = np.random.choice(range(5), p=[0.4, 0.3, 0.1, 0.1, 0.1])

        active = np.random.randint(10)
        cap = np.zeros(10).tolist()
        delay = np.zeros(10).tolist()
        cap[active] = classOfService[userType][0]
        delay[active] = classOfService[userType][1]
        
        if int(i/2)%2 == 0:
            cap = 10
        else:
            cap = 5
        
        data['userEquipment'].append({
            'uuid': str(uuid.uuid4()),
            'capacity': cap,
            'delay': 1, #classOfService[userType][1],
            'arrival': int(i/2), # tarrival,
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


if args.blockage:
    data['blockage'] = []
    data['blockage'].append([])
    data['blockage'].append([])
    for m in range(len(data['baseStation'])):
        #data['blockage'].append([])
        for n in range(args.userEquipments):
            #data['blockage'][m].append([])
            if m == 0:
                data['blockage'][0].append([])
                data['blockage'][1].append([])
            for t in range(args.simTime):
                if data['userEquipment'][n]['capacity'] == 10:
                    data['blockage'][m][n].append(1)
                elif m == 0:
                    if n%2==0:
                        data['blockage'][0][n].append(0)#np.random.randint(2))
                        data['blockage'][1][n].append(1)#np.random.randint(2))
                    else:
                        data['blockage'][0][n].append(1)#np.random.randint(2))
                        data['blockage'][1][n].append(0)#np.random.randint(2))
                elif m==1:
                    continue

if args.outputFile == None:
    outname = 'instance-'+str(args.baseStations)+'-'+str(args.userEquipments)+\
        '-'+str(args.simTime)+'-'+str(args.seed)+'.json'
else:
    outname = args.outputFile
with open(outname, 'w') as outfile:
    json.dump(data, outfile, indent=4)

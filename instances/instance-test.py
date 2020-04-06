#!/usr/bin/env python3

import json
import uuid
import argparse


data = {}
data['channel'] = []
data['channel'] = {
    'lossExponent': 3.41,
    'noisePower': -80
    }

data['baseStation'] = []
data['baseStation'].append({
    'uuid': str(uuid.uuid4()),
    'resourceBlocks': 275,
    'frequency': 30e9,
    'subcarrierSpacing': 120e3,
    'txPower': 30, #dBm
    'position': {
        'x': 0,
        'y': 0
        }
    })

data['userEquipment'] = []
data['userEquipment'].append({
    'uuid': str(uuid.uuid4()),
    'capacity': 1e8,
    'position': {
        'x': 35,
        'y': 35
        }
    })

data['userEquipment'].append({
    'uuid': str(uuid.uuid4()),
    'capacity': 1e8,
    'position': {
        'x': 35,
        'y': -35
        }
    })

data['userEquipment'].append({
    'uuid': str(uuid.uuid4()),
    'capacity': 1e8,
    'position': {
        'x': -35,
        'y': -35
        }
    })

data['userEquipment'].append({
    'uuid': str(uuid.uuid4()),
    'capacity': 1e8,
    'position': {
        'x': -35,
        'y': 35
        }
    })


with open('test.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)


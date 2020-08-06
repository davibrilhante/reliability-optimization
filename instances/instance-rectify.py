#!/usr/bin/env python3
# -*- coding : utf8 -*-

import json
import numpy as np
import os

topdir = 'out/'
files = [f for f in os.listdir(topdir) if os.path.isfile(os.path.join(topdir, f))]
#data = [[] for i in files]
data = {}

print('Loading files ', end='')

for f in files:
    print(f, end='')
    with open(topdir+f, 'r') as infile:
        try:
            data = json.load(infile)
        except:
            continue
    
    for n,_ in enumerate(data['userEquipment']):
        Lambda = 500  #milliseconds
        newPackets = int(data['scenario']['simTime']/Lambda - 1)
        arrival = np.random.poisson(Lambda,newPackets).tolist()
        data['userEquipment'][n]['nPackets'] = newPackets
        data['userEquipment'][n]['packets'] = [sum(arrival[:i+1]) for i in range(newPackets)]

    infile.close()

    with open(topdir+f, 'w') as outfile:
        json.dump(data,outfile,indent=4)
    outfile.close()

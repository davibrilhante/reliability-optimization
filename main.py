#! /usr/bin/env python3
# -*- coding = utf-8 -*-
 
import numpy as np
import argparse
import json
import operator
import sys

from collections import OrderedDict
 
from matplotlib import pyplot as plt
 
#### 5G NR ENV
import simutime
import components
import definitions as defs
import simpy as sp

from decompressor import decompressor
from fivegmodules.devices import MobileUser
from fivegmodules.devices import KPI
from fivegmodules.devices import NetworkParameters
from fivegmodules.devices import Numerology
from fivegmodules.devices import BaseStation
from fivegmodules.miscellaneous import Scenario
from fivegmodules.miscellaneous import AWGNChannel
from fivegmodules.miscellaneous import DummyAntenna
from fivegmodules.miscellaneous import ConePlusCircle
from fivegmodules.mobility import StraightRoute
from fivegmodules.handover import A3Handover


parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
parser.add_argument('-p','--plot', action='store_true', help='xxx')
parser.add_argument('--ttt', type=int, default=640)
parser.add_argument('--untoggleBlockage', action='store_true')
parser.add_argument('--untoggleRayleigh', action='store_false')
parser.add_argument('--untoggleShadowing', action='store_false')
parser.add_argument('--untoggleInterference', action='store_false')
parser.add_argument('--uedelay', type=int)
parser.add_argument('--uecapacity', type=float)
args = parser.parse_args()


if __name__ == '__main__':
    network = []
    nodes = []

    ### Create base data
    try:
        with open(args.inputFile) as json_file:
            data = json.load(json_file)
    except Exception as e:
        print(e)
        print(10)
        sys.exit()

    decompressor(data)

    #scenario = data['scenario']
    #channel = data['channel']
    for p in data['baseStation']:
        p['sensibility'] = -120 #dB m
        network.append(p)
    for p in data['userEquipment']:
        p['txPower'] = 20 #dBm
        p['sensibility'] = -110 #dB m
        nodes.append(p)

    #scenario['simTime'] = min(12000, scenario['simTime'])
    for ue in nodes:
        #ue['nPackets'] = int(scenario['simTime']/500) - 1
        ue['capacity'] = 750e6 #Bits per second

    if args.untoggleBlockage:
        LOS = np.ones((len(network), len(nodes), data['scenario']['simTime']), dtype = np.int8)

    else:
        LOS = data['blockage']

    channel = AWGNChannel()
    channel.noisePower = data['channel']['noisePower']
    scenario = Scenario(data['scenario']['simTime'])
    scenario.frequency = 28e9 #Hz
    scenario.wavelength = 2e8/scenario.frequency
    networkParams = NetworkParameters()
    networkParams.timeToTrigger = args.ttt

    np.random.seed(data['scenario']['seed'])

    numerology = Numerology(120, networkParams)

    #antenna = DummyAntenna(10) #gain
    antenna = ConePlusCircle(5,np.deg2rad(120)) #gain

    baseStations = {}
    ### Creating list of Base Stations
    for i in network:
        baseStations[i['uuid']] = BaseStation(scenario, i)
        baseStations[i['uuid']].channel = channel
        baseStations[i['uuid']].networkParameters = networkParams
        baseStations[i['uuid']].numerology = numerology
        baseStations[i['uuid']].antenna = antenna
        baseStations[i['uuid']].initializeServices()

    scenario.addBaseStations(baseStations)

    mobiles = {}
    ### Creating a list of nodes
    for n, i in enumerate(nodes):
        mobiles[i['uuid']] = MobileUser(
                                        scenario, i,
                                        i['speed']['x'],
                                        i['speed']['y']
                                        )
        mobiles[i['uuid']].mobilityModel = StraightRoute()
        mobiles[i['uuid']].channel = {}

        if args.uedelay:
            mobiles[i['uuid']].delay = args.uedelay
        else:
            mobiles[i['uuid']].delay = i['delay']

        if args.uecapacity:
            mobiles[i['uuid']].capacity = args.uecapacity
        else:
            mobiles[i['uuid']].capacity = i['capacity']

        for j in network:                
            mobiles[i['uuid']].channel[j['uuid']] = AWGNChannel()
            mobiles[i['uuid']].channel[j['uuid']].noisePower = data['channel']['noisePower']
            mobiles[i['uuid']].channel[j['uuid']].switchShadowing = args.untoggleShadowing
            mobiles[i['uuid']].channel[j['uuid']].switchFading = args.untoggleRayleigh

            if mobiles[i['uuid']].channel[j['uuid']].switchFading:
                try: 
                    with open(j['uuid']) as filehandle:
                        mobiles[i['uuid']].channel[j['uuid']].fadingSamples = json.load(filehandle)

                except FileNotFoundError:
                    doppler = np.hypot(mobiles[i['uuid']].Vx, mobiles[i['uuid']].Vy)/scenario.wavelength
                    mobiles[i['uuid']].channel[j['uuid']].generateRayleighFading(doppler, scenario.simTime)

                    with open(j['uuid'], 'w') as filehandle:
                        json.dump(mobiles[i['uuid']].channel[j['uuid']].fadingSamples.tolist(), filehandle)

        mobiles[i['uuid']].switchInterference = args.untoggleInterference
        mobiles[i['uuid']].ignoreFirstAssociation = True
                

        mobiles[i['uuid']].initializeServices()
        mobiles[i['uuid']].networkParameters = networkParams
        mobiles[i['uuid']].packetArrivals = i['packets']
        mobiles[i['uuid']].antenna = antenna
        mobiles[i['uuid']].handover = A3Handover()

        mobiles[i['uuid']].addLosInfo(LOS, n)

    scenario.addUserEquipments(mobiles)

    scenario.run(until=scenario.simTime)


    if args.plot:
        scenario.plot()
    for n, i in enumerate(nodes):
        mobiles[i['uuid']].kpi.printAsDict()
        if args.plot:
            mobiles[i['uuid']].plotRSRP.plot()
            mobiles[i['uuid']].plotSINR.plot()

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

    #scenario = data['scenario']
    #channel = data['channel']
    LOS = data['blockage']
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


    channel = AWGNChannel()
    channel.noisePower = data['channel']['noisePower']
    scenario = Scenario(data['scenario']['simTime'], channel)
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
        mobiles[i['uuid']].channel = channel
        mobiles[i['uuid']].initializeServices()
        mobiles[i['uuid']].networkParameters = networkParams
        mobiles[i['uuid']].packetArrivals = i['packets']
        mobiles[i['uuid']].antenna = antenna
        mobiles[i['uuid']].handover = A3Handover()
        mobiles[i['uuid']].addLosInfo(LOS, n)

    scenario.addUserEquipments(mobiles)

    scenario.run(until=scenario.simTime)

    for n, i in enumerate(nodes):
        mobiles[i['uuid']].kpi._print()

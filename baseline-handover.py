#! /usr/bin/env python3
# -*- coding = utf-8 -*-

import numpy as np
import argparse
import json
import operator

#### 5G NR ENV
import simutime
import components
import definitions
import simpy as sp


class BaseStation(object):
    def __init__(self, bsDict, env):
        self.x = bsDict['position']['x']
        self.y = bsDict['position']['y']
        self.frequency = bsDict['frequency']
        self.env = env

        self.txPower = bsDict['txPower']
        self.antennaGain = 10 #### STUB!!!

        self.bandwidth = 12*bsDict['resourceBlocks']*bsDict['subcarrierSpacing']

        self.associatedUsers = []
        self.inRangeUsers = []
        self.frameIndex = 0
        self.ssbIndex = 0

        self.numerology = components.numerology(bsDict['subcarrierSpacing']/1e3)


    def burstSet(self, burstDuration, burstPeriod, rachPeriod):
        '''
        Schedules a Burst Set event with burstDuration (in seconds)
        each burstPeriod (in seconds), except when a defs.RACH Opportunity
        is scheduled.

        The counter verifies if its time of a burst set or a defs.RACH Opportunity

        burstDuration = 5 milliseconds
        burstPeriod = 20 milliseconds
        rachPeriod = 40 milliseconds
        '''
        while True:
            self.availableSlots = self.numerology['ssblocks']
            if (self.frameIndex % (rachPeriod/defs.FRAME_DURATION) != 0) and (self.frameIndex != 1):
                print('A new burst set is starting at %d and it is the %d ss burst in %d frame' % 
                        (self.env.now, self.ssbIndex, self.frameIndex))
                yield self.env.timeout(burstDuration)
                print('The burst set has finished at %d' % self.env.now)
                #self.calcNetworkCapacity()
                yield self.env.timeout(burstPeriod - burstDuration)
            else:
                yield self.env.timeout(burstPeriod)

    def updateFrame(self):
        #self.frameIndex+=1
        while True:
            print('Frame:',self.frameIndex,'in',self.env.now)
            self.frameIndex+=1
            yield self.env.timeout(defs.FRAME_DURATION)
            self.calcNetworkCapacity()
            if self.frameIndex % (defs.BURST_PERIOD/defs.FRAME_DURATION) == 0:
                self.ssbIndex+=1

    def rachOpportunity(self, rachDuration, rachPeriod):
        '''
        Schedules a defs.RACH Opportunity event with rachDuration (in seconds)
        each rachPeriod (in seconds)
        
        rachDuration = 5 milliseconds
        rachOpportunity = 40 milliseconds
        '''
        while True:
            ### Grants a SS Burst at the first frame but avoids a defs.RACH at the first frame
            if self.frameIndex==1:
                yield self.env.timeout(rachPeriod)
            else:
                print('A new rach opportunity is starting at %d and it is the %d ss burst in %d frame' 
                        % (self.env.now, self.ssbIndex, self.frameIndex))
                yield self.env.timeout(rachDuration)
                print('The rach opportunity  has finished at %d' % self.env.now)
                '''
                #GAMBIARRA
                temp = self.ALG
                self.ALG = '0'
                self.calcNetworkCapacity()
                self.ALG = temp
                '''
                yield self.env.timeout(rachPeriod - rachDuration)

    def initializeServices(self):
        self.env.process(self.updateFrame())
        self.env.process(self.burstSet(defs.BURST_DURATION, defs.BURST_PERIOD, defs.RACH_PERIOD))
        self.env.process(self.rachOpportunity(defs.BURST_DURATION, defs.RACH_PERIOD))

    def calcUserDist(self, user):
        '''
        Returns the distance from user to base station
        '''
        return np.hypot(user.x, user.y)

    def calcUserAngle(self, user):
        '''
        Returns the the angle between user and cartesian plane
        defined with base station at the center
        '''
        return np.rad2deg(np.arctan2(user.y, user.x))



class MobileUser(object):
    def __init__(self, ueDict, scenario):
        self.x = ueDict['position']['x']
        self.y = ueDict['position']['y']
        self.Vx = ueDict['speed']['x']
        self.Vy = ueDict['speed']['y']
        self.channel = scenario.channel
        self.env = scenario.env
        self.packetArrivals = ueDict['packets']
        self.nPackets = ueDict['nPackets']

        self.snrThreshold = ueDict['threshold']
        self.uuid = ueDict['uuid']
        self.servingBS = None
        self.lastBS = None

        # This is a dictionary of all BS in the scenario
        self.listBS = scenario.baseStations

        # This is a dictionary of BS and their respective RSRP
        # which is updated at each time to measure
        self.listedRSRP = { }

        self.sensibility = -120 #dBm
        self.antennaGain = 10 #dB (whatever the value that came to my mind)
        self.timeToTrigger = 640 #milliseconds
        self.timeToMeasure = 20 #millisecond

        self.qualityOut =  -110 #dBm
        self.qualityIn = -90 #dBm
        self.qualityInCounter = 0
        self.qualityOutCounter = 0
        self.n310 = 10 #default is 1?
        self.t310 = 100 #milliseconds #default is 1000?
        self.n311 = 10 #default is 1?
        self.sync = False

        self.HOHysteresis = 0 #dB from 0 to 30
        self.HOOffset = 3 #dB
        self.HOThreshold = -90 #dBm

        self.triggerTime = 0
        self.measOccurring = False
        #self.handoverEvent = 'A3_EVENT'

        self.kpi = {
                'handover' : 0,
                'handoverFail' : 0,
                'pingpong' : 0,
                'throughput' : [],
                'deliveryRate' : 0
                }

        self.blockage = {}

    # Launches the initial service (process)
    def initializeServices(self):
        self.env.process(self.measurementEvent())
        self.env.process(self.sendingPackets())

    # Configs the TTT, seting a time different from default value
    def configTTT(self,time):
        self.timeToTrigger = time

    '''
    # We dont need it anymore
    def configHandoverEvent(self,event):
        self.handoverEvent = event
    '''


    ### This method updates the list of BS and their RSRP
    def updateListBS(self):
        timePast = self.env.now
        xnow = self.x + (self.Vx/3.6)*timePast*1e-3 # time is represented in milliseconds
        ynow = self.y + (self.Vy/3.6)*timePast*1e-3 

        for uuid, bs in self.listBS.items():
            ##### NEED TO INCLUDE BLOCKAGE!!!
            distance = np.hypot(xnow - bs.x, ynow - bs.y)
            wavelength = 3e8/bs.frequency
            
            #NLOS condition
            if self.blockage[uuid][self.env.now] == 0:
                pathloss = 61.4 + 20*np.log10(distance)
            #LOS condition
            else:
                pathloss = 72 + 29.2*np.log10(distance)

            #pl_0 = 20*np.log10(4*np.pi/wavelength)
            #pathloss = pl_0 + 10*self.channel['lossExponent']*np.log10(distance)
            RSRP = bs.txPower + bs.antennaGain + self.antennaGain - pathloss

            if RSRP > self.sensibility:
                self.listedRSRP[uuid] = RSRP
            else:
                self.listedRSRP[uuid] = None 


    # At each time to measure, the UE updates the list of BS and check
    # if the handover event is happening
    def measurementEvent(self):
        while True:
            self.updateListBS()
            self.measurementCheck()
            yield self.env.timeout(self.timeToMeasure)


    # Checks the handover event cnfition
    def measurementCheck(self):
        maxRSRP = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]

        ### FIRST TIME USER ASSOCIATON
        if self.servingBS == None:
            self.servingBS = maxRSRP
            self.sync = True

        else:
            #Check if the UE is sync with the BS
            self.env.process(self.signalQualityCheck())

            snr = self.listedRSRP[self.servingBS] - self.channel['noisePower']
            print(self.env.now, maxRSRP, self.listedRSRP[maxRSRP], self.servingBS, self.listedRSRP[self.servingBS])
            rate = self.listBS[self.servingBS].bandwidth*np.log2(1 + snr)
            self.kpi['throughput'].append(rate)

            # This is the condition of an A3 event, triggering a RSRP measurement
            if self.listedRSRP[maxRSRP] - self.HOHysteresis > self.listedRSRP[self.servingBS] + self.HOOffset:

                targetBS = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]
                self.measOccurring = True

                if self.triggerTime == 0:
                    #First time A3 condition is satisfied
                    self.triggerTime = self.env.now

                else:
                    # It is not the first time A3 codition is satified by maxRSRP BS 
                    if self.sync:
                        #Handover to maxRSRP BS
                        self.sendMeasurementReport(targetBS) 

                    else:
                        #Handover failure
                        self.kpi['handoverFail'] += 1

            # The A3 event condition was not maintained, so the measurement should stop 
            elif self.listedRSRP[maxRSRP] - self.HOHysteresis < self.listedRSRP[self.servingBS] + self.HOOffset:
                self.measOccurring = False
                if self.triggerTime != 0:
                    self.triggerTime = 0



    # Testing whether the handover did not fail due to be late
    def signalQualityCheck(self):
        if self.listedRSRP[self.servingBS] < self.qualityOut:
            downcounter = self.t310
            while downcounter > 0:
                self.qualityOutCounter += 1

                if self.qualityOutCounter >= self.n310:
                    #Start out of sync counter
                    downcounter -= 1
                    yield self.env.timeout(1)

                if self.listedRSRP[self.servingBS] >= self.qualityIn:
                    self.qualityInCounter += 1

                    if self.qualityInCounter >= self.n311:
                        #Stop out of sync counter 
                        self.qualityOutCounter = 0
                        downcounter = self.t310
                        break

                    else:
                        #Signal strength is better but the sync still
                        #unconfirmed by the N311 counter, so T310 continues
                        downcounter -= 1
                        yield self.env.timeout(1)

            if downcounter == 0:
                #out of sync!
                self.sync = False

                #Need to reassociate with the network
                self.sevingBS = None

            self.qualityOutCounter = 0
            self.qualityInCounter = 0


    #  
    def sendMeasurementReport(self, targetBS):
        if (self.env.now >= self.triggerTime + self.timeToTrigger) and self.measOccurring:
            #Check if it is not a reassociation
            if self.listedRSRP[self.servingBS] != None:
                # Check if it is a pingpong, just for kpi assessment
                if self.lastBS == targetBS:
                    self.kpi['pingpong'] += 1

                # Does the handover, simple as that!
                self.lastBS = self.servingBS
                self.servingBS = targetBS
                self.kpi['handover'] += 1

    def sendingPackets(self):
        for t in self.packetArrivals:
            yield self.env.timeout(t - self.env.now)
            if self.sync and self.servingBS != None:
                self.kpi['deliveryRate'] += 1


    def addLosInfo(self, los : list, n : int) -> list:
        for m,bs in enumerate(self.listBS):
            self.blockage[bs] = los[m][n]

    def printKPI(self):
        print(self.uuid)
        print(self.kpi['handover'])
        print(self.kpi['pingpong'])
        print(self.kpi['handoverFail'])
        print(np.mean(self.kpi['throughput']))
        print(self.kpi['deliveryRate']/self.nPackets)



class Simulator(object):
    def __init__(self, env, data, channel):
        self.simTime = data['simTime']
        self.env = env
        self.channel = channel

        self.userEquipments = None
        self.baseStations = None

    def addBaseStations(self, baseStations : dict):
        self.baseStations = baseStations

    def addUserEquipments(self, userEquipments : dict):
        self.userEquipments = userEquipments

        

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')
args = parser.parse_args()



if __name__ == '__main__':
    network = []
    nodes = []

    ### Create base data
    with open(args.inputFile) as json_file:
        data = json.load(json_file)
        scenario = data['scenario']
        channel = data['channel']
        LOS = data['blockage']
        for p in data['baseStation']:
            network.append(p)
        for p in data['userEquipment']:
            nodes.append(p)


    env = sp.Environment()

    sim = Simulator(env, scenario, channel)

    baseStations = {}
    ### Creating list of Base Stations
    for i in network:
        baseStations[i['uuid']] = BaseStation(i, sim)

    sim.addBaseStations(baseStations)

    mobiles = {}
    ### Creating a list of nodes
    for n, i in enumerate(nodes):
        mobiles[i['uuid']] = MobileUser(i, sim)
        mobiles[i['uuid']].initializeServices()
        mobiles[i['uuid']].addLosInfo(LOS, n)


    env.run(until=scenario['simTime'])

    for _,ue in mobiles.items():
        ue.printKPI()

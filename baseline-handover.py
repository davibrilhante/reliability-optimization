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
        self.delay = ueDict['delay']
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

        self.sensibility = -110 #dBm
        self.antennaGain = 10 #dB (whatever the value that came to my mind)
        self.timeToTrigger = 640 #milliseconds
        self.timeToMeasure = 1 #20 #millisecond

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

        self.plotAssociation = []
        self.plotSNR = []
        self.plotCapacity = []
        self.plotLOS = []
        self.plotMaxRSRP = []
        self.plotRecvRSRP = [[] for i in self.listBS.keys()]

        self.reassociationFlag = False

        self.kpi = {
                'partDelay' : 0,
                'handover' : 0,
                'handoverFail' : 0,
                'pingpong' : 0,
                'reassociation' : 0,
                'throughput' : [],
                'deliveryRate' : 0,
                'delay' : []
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
            if self.blockage[uuid][self.env.now] == 1:
                pathloss = 61.4 + 20.0*np.log10(distance)
            #LOS condition
            else:
                pathloss = 72.0 + 29.2*np.log10(distance)

            #pl_0 = 20*np.log10(4*np.pi/wavelength)
            #pathloss = pl_0 + 10*self.channel['lossExponent']*np.log10(distance)
            RSRP = bs.txPower + bs.antennaGain + self.antennaGain - pathloss

            if RSRP > self.sensibility:
                self.listedRSRP[uuid] = RSRP
            else:
                self.listedRSRP[uuid] = float('-inf') #None


    # At each time to measure, the UE updates the list of BS and check
    # if the handover event is happening
    def measurementEvent(self):
        while True:
            self.updateListBS()
            self.measurementCheck()
            yield self.env.timeout(self.timeToMeasure)


    # Checks the handover event condition
    def measurementCheck(self):
        maxRSRP = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]
        for n,i in enumerate(self.listedRSRP.keys()):
            self.plotRecvRSRP[n].append(self.listedRSRP[i])

        ### FIRST TIME USER ASSOCIATON
        if self.servingBS == None and self.listedRSRP[maxRSRP] > self.sensibility:
            self.servingBS = maxRSRP
            self.sync = True

        elif self.servingBS == None and self.lastBS != None and self.listedRSRP[self.lastBS] == float('-inf'):
            if not self.reassociationFlag:
                self.kpi['reassociation'] += 1
                self.reassociationFlag = True


        elif self.servingBS != None:
            #Check if the UE is sync with the BS
            self.env.process(self.signalQualityCheck())

            snr = self.listedRSRP[self.servingBS] - self.channel['noisePower']
            #print(self.env.now, maxRSRP, self.listedRSRP[maxRSRP], self.servingBS, self.listedRSRP[self.servingBS], snr)
            #rate = self.listBS[self.servingBS].bandwidth*np.log2(1 + snr)
            #self.kpi['throughput'].append(rate)

            # This is the condition of an A3 event, triggering a RSRP measurement
            if self.listedRSRP[maxRSRP] - self.HOHysteresis > self.listedRSRP[self.servingBS] + self.HOOffset:

                targetBS = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]
                self.measOccurring = True

                if self.triggerTime == 0:
                    #First time A3 condition is satisfied
                    self.triggerTime = self.env.now
                    #self.sendMeasurementReport(targetBS) 

                    #print('A3 CONDITION SATISFIED', targetBS, self.listedRSRP[targetBS])

                else:
                    # It is not the first time A3 codition is satified by maxRSRP BS 
                    if self.sync:
                        #Handover to maxRSRP BS
                        self.sendMeasurementReport(targetBS) 


                    ### It is a mess and needs repair!!!
                    if not self.sync or self.listedRSRP[self.servingBS] < self.qualityOut:
                        #Handover failure
                        #print('HANDOVER FAILURE')
                        self.lastBS = self.servingBS
                        self.servingBS = None
                        self.reassociationFlag = False
                        if self.measOccurring:
                            self.kpi['handoverFail'] += 1
                            self.kpi['handover'] += 1
                            self.measOcurring = False

            # The A3 event condition was not maintained, so the measurement should stop 
            elif self.listedRSRP[maxRSRP] - self.HOHysteresis < self.listedRSRP[self.servingBS] + self.HOOffset:
                self.measOccurring = False
                if self.triggerTime != 0:
                    self.triggerTime = 0

        '''
        if self.listedRSRP[self.servingBS] < self.qualityOut:
            print('CONNECTION WITH SERVING BS LOST')
            self.servingBS = None
        '''
        self.plotAssociation.append(self.servingBS)
        if self.servingBS == None:
            self.plotMaxRSRP.append(float('-inf'))
        else:
            self.plotMaxRSRP.append(self.listedRSRP[self.servingBS])#self.listedRSRP[maxRSRP])



    # Testing whether the handover did not fail due to be late
    def signalQualityCheck(self):
        if self.servingBS != None:
            #print(self.env.now, self.servingBS,self.listedRSRP[self.servingBS])
            if self.listedRSRP[self.servingBS] < self.qualityOut and self.qualityOutCounter == 0:
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

                    self.lastBS = self.servingBS
                    self.reassociationFlag = False
                    #Need to reassociate with the network
                    self.servingBS = None
                    #print('USER OUT OF SYNC')
                    
                    #yield self.env.timeout(1)

                    #print('TRIGGERING NEW ASSOCIATION')
                    #self.measurementCheck()

                self.qualityOutCounter = 0
                self.qualityInCounter = 0

        else:
            self.sync = False


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
                #snr = self.listedRSRP[self.servingBS] - self.channel['noisePower']
                snr = max(0, self.listedRSRP[self.servingBS] - self.channel['noisePower'])
                temp = self.snrThreshold
                timer = 0 
                while timer < 10:
                    if snr > self.snrThreshold or snr > temp:
                        self.kpi['deliveryRate'] += 1
                        rate = self.listBS[self.servingBS].bandwidth*np.log2(1 + snr)
                        self.kpi['throughput'].append(rate)
                        self.kpi['delay'].append(self.env.now - t)
                        if self.kpi['delay'][-1] < self.delay:
                            self.kpi['partDelay'] += 1
                            
                        break
                    else:
                        temp *= 0.9
                        yield self.env.timeout(1)
                        timer += 1


    def addLosInfo(self, los : list, n : int) -> list:
        for m,bs in enumerate(self.listBS):
            self.blockage[bs] = los[m][n]

    def plot(self, plotType, subplot=True):
        plt.figure()

        ### plot association
        baseIndex = list(self.listBS.keys())
        association = []
        
        plt.xlabel('Measurement Index')
        handover = []
        log = { str(i) : [] for i in range(len(baseIndex)+1)}
        colors = ['slateblue', 'springgreen', 'tomato','silver','orange']

        #print(self.plotAssociation)

        for n,bs in enumerate(self.plotAssociation):
            if bs == None:
                association.append(0)
            else:
                association.append(baseIndex.index(bs)+1)

            if len(association)>1:
                #if baseIndex.index(bs)+1 != association[-1]:
                if bs != self.plotAssociation[n-1]:
                    handover.append(n)

        if self.kpi['handover']>=1:
            for i in range(len(handover)+1):
                if i == 0:
                    #log[str(association[0])] = [0,handover[0]]
                    log[str(association[0])].append([0,handover[0]])
                elif i == len(handover):
                    #log[str(association[-2])] = [handover[-1],len(association)]
                    log[str(association[-2])].append([handover[-1],len(association)])
                else:
                    #print(association[handover[i-1]])
                    #log[str(association[handover[i-1]])] = [handover[i-1],handover[i]]
                    log[str(association[handover[i-1]])].append([handover[i-1],handover[i]])
                
        #print(log)
        #print(handover)

        if plotType == 'MAX_RSRP' or subplot:
            if subplot:
                if self.kpi['handover'] == 0:
                    plt.subplot(311)
                else:
                    plt.subplot(221)
                plt.grid()
                plt.ylabel('Max RSRP [dBm]')
            else:
                plt.ylabel('Maximum Reference Signal Receive Power [dBm]')

            if self.kpi['handover']>0:
                for n,v in zip(log.keys(), log.values()):
                    for t in v:
                        if n == '0':
                            plt.plot(range(t[0], t[1]),self.plotMaxRSRP[t[0]:t[1]], 
                                    color=colors[int(n)], 
                                    label='No BS')
                        else:
                            plt.plot(range(t[0], t[1]),self.plotMaxRSRP[t[0]:t[1]], 
                                    color=colors[int(n)], 
                                    label='BS '+n)
            else:
                plt.plot(self.plotMaxRSRP)


        if plotType == 'RECV_RSRP' or subplot:
            if len(self.listBS.keys()) > 1:
                if subplot:
                    plt.subplot(223)
                    plt.grid()
                    plt.ylabel('RSRP [dBm]')
                else:
                    plt.ylabel('Received Reference Signal Receive Power [dBm]')

                for n, p in enumerate(self.plotRecvRSRP):
                    plt.plot(p,color=colors[n+1],label='BS '+str(n))



        if plotType == 'SNR' or subplot:
            plotter = []
            if subplot:
                if self.kpi['handover'] == 0:
                    plt.subplot(312)
                else:
                    plt.subplot(222)
                plt.grid()
                plt.ylabel('SNR')
            else:
                plt.ylabel('Signal to Noise Ratio')

            for r in self.plotMaxRSRP:
                plotter.append(max(0, r - self.channel['noisePower']))

            if self.kpi['handover']>0:
                for n,v in zip(log.keys(), log.values()):
                    for t in v:
                        if n == '0':
                            plt.plot(range(t[0], t[1]),plotter[t[0]:t[1]], 
                                    color=colors[int(n)], 
                                    label='No BS')
                        else:
                            plt.plot(range(t[0], t[1]),plotter[t[0]:t[1]], 
                                    color=colors[int(n)], 
                                    label='BS '+n)
            else:
                plt.plot(plotter)


        if plotType == 'Capacity' or subplot:
            plotter = []
            if subplot:
                if self.kpi['handover'] == 0:
                    plt.subplot(313)
                else:
                    plt.subplot(224)
                plt.grid()
                plt.ylabel('Data Rate [Mbps]')
            else:
                plt.ylabel('Achieved Data Rate [Mbps]')

            for r in self.plotMaxRSRP:
                snr = r - self.channel['noisePower']
                plotter.append(self.listBS[self.servingBS].bandwidth*np.log2(1 + max(0, snr))/1e6)

            if self.kpi['handover']>0:
                for n,v in zip(log.keys(), log.values()):
                    for t in v:
                        if n == '0':
                            plt.plot(range(t[0], t[1]),plotter[t[0]:t[1]], 
                                    color=colors[int(n)], 
                                    label='No BS')
                        else:
                            plt.plot(range(t[0], t[1]),plotter[t[0]:t[1]], 
                                    color=colors[int(n)], 
                                    label='BS '+n)
            else:
                plt.plot(plotter)


        if subplot:
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.15, right=0.95, hspace=0.25,
                    wspace=0.35)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()


                        

    def printKPI(self):
        self.kpi['uuid'] = self.uuid
        self.kpi['partDelay'] /= self.nPackets
        self.kpi['throughput'] = np.mean(self.kpi['throughput'])
        self.kpi['deliveryRate'] /= self.nPackets
        self.kpi['delay'] = np.mean(self.kpi['delay'])

        print(json.dumps(self.kpi, indent=4))        
        '''
        print(self.uuid)
        print(self.kpi['partDelay']/self.nPackets)
        print(self.kpi['handover'])
        print(self.kpi['pingpong'])
        print(self.kpi['handoverFail'])
        print(np.mean(self.kpi['throughput']))
        print(self.kpi['deliveryRate']/self.nPackets)
        print(np.mean(self.kpi['delay']))
        '''



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
parser.add_argument('-p','--plot', action='store_true', help='xxx')
args = parser.parse_args()



if __name__ == '__main__':
    network = []
    nodes = []

    ### Create base data
    with open(args.inputFile) as json_file:
        try:
            data = json.load(json_file)
        except:
            sys.exit()

        scenario = data['scenario']
        channel = data['channel']
        LOS = data['blockage']
        for p in data['baseStation']:
            network.append(p)
        for p in data['userEquipment']:
            nodes.append(p)

    #scenario['simTime'] = min(12000, scenario['simTime'])
    for ue in nodes:
        #ue['nPackets'] = int(scenario['simTime']/500) - 1
        ue['capacity'] = 750e6 #Bits per second


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

        if args.plot:
            #ue.plot('Association')
            ue.plot('MAX_RSRP')
            #ue.plot('RECV_RSRP', False)
            #ue.plot('SNR', False)
            #ue.plot('Capacity', False)

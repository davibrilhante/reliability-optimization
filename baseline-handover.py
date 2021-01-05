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


class BaseStation(object):
    def __init__(self, bsDict, scenario):
        self.x = bsDict['position']['x']
        self.y = bsDict['position']['y']
        self.frequency = bsDict['frequency']
        self.env = scenario.env

        self.txPower = bsDict['txPower']
        self.antennaGain = 10 #### STUB!!!

        self.bandwidth = 12*bsDict['resourceBlocks']*bsDict['subcarrierSpacing']
        self.resourceBlocks = bsDict['resourceBlocks']

        self.associatedUsers = []
        self.inRangeUsers = []
        self.frameIndex = 0
        self.ssbIndex = 0

        self.onSSB = False
        self.nextSSB = 0
        self.onRach = False
        self.nextRach = 0

        self.numerology = components.numerology(bsDict['subcarrierSpacing']/1e3)
        self.slotsPerSubframe = bsDict['subcarrierSpacing']/(15*1e3)


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
                #print('A new burst set is starting at %d and it is the %d ss burst in %d frame' % 
                #        (self.env.now, self.ssbIndex, self.frameIndex))

                ### Flagging that there is an SSB happening
                self.onSSB = True
                yield self.env.timeout(burstDuration)
                #print('The burst set has finished at %d' % self.env.now)
                #self.calcNetworkCapacity()

                ### Flagging that the SSB is finished
                self.onSSB = False

                if ((self.frameIndex + (burstPeriod/defs.FRAME_DURATION))/
                        (rachPeriod/defs.FRAME_DURATION) != 0):
                    self.nextSSB = self.env.now + (burstPeriod - burstDuration)
                else:
                    self.nextSSB = self.env.now + (rachPeriod - burstDuration)

                yield self.env.timeout(burstPeriod - burstDuration)

            else:
                yield self.env.timeout(burstPeriod)

    def updateFrame(self):
        #self.frameIndex+=1
        while True:
            #print('Frame:',self.frameIndex,'in',self.env.now)
            self.frameIndex+=1
            yield self.env.timeout(defs.FRAME_DURATION)
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
                self.nextRach = self.env.now + rachPeriod
                yield self.env.timeout(rachPeriod)
            else:
                #print('A new rach opportunity is starting at %d and it is the %d ss burst in %d frame' 
                #        % (self.env.now, self.ssbIndex, self.frameIndex))

                ## Flagging a new RACH opportunity
                self.onRach = True
                yield self.env.timeout(rachDuration)
                #print('The rach opportunity  has finished at %d' % self.env.now)

                ### Flagging the end of a RACH opportunity
                self.onRach = False
                self.nextRach = self.env.now + (rachPeriod - rachDuration)

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
        return np.hypot(user.x - self.x, user.y - self.y)

    def calcUserAngle(self, user):
        '''
        Returns the the angle between user and cartesian plane
        defined with base station at the center
        '''
        return np.rad2deg(np.arctan2(user.y - self.y, user.x - self.x))



class MobileUser(object):
    def __init__(self, ueDict, scenario):
        self.x = ueDict['position']['x']
        self.y = ueDict['position']['y']
        self.Vx = ueDict['speed']['x']
        self.Vy = ueDict['speed']['y']
        self.delay = ueDict['delay']
        self.capacity = ueDict['capacity']
        self.channel = scenario.channel
        self.env = scenario.env
        self.packetArrivals = ueDict['packets']
        self.nPackets = ueDict['nPackets']

        self.snrThreshold = ueDict['threshold']
        self.uuid = ueDict['uuid']
        self.servingBS = None
        self.lastBS = []

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
        self.n310 = 10 #default is 160 (one per millisecond)?
        self.t310 = 100 #milliseconds #default is 1000?
        self.n311 = 10 #default is 40 (one per millisecond)? 
        self.sync = False # Flag if the user is out of sync

        self.HOHysteresis = 1 #dB from 0 to 30
        self.HOOffset = 3 #dB
        self.HOThreshold = -90 #dBm

        self.triggerTime = 0
        self.measOccurring = False
        self.lastMeasurement = 0
        #self.handoverEvent = 'A3_EVENT'

        self.plotAssociation = []
        self.plotSNR = []
        self.plotCapacity = []
        self.plotLOS = []
        self.plotMaxRSRP = []
        self.plotRecvRSRP = [[] for i in self.listBS.keys()]

        self.reassociationFlag = False
        self.mobilityModel = None

        self.kpi = {
                'partDelay' : 0,
                'handover' : 0,
                'handoverFail' : 0,
                'pingpong' : 0,
                'reassociation' : 0,
                'throughput' : [],
                'capacity' : [],
                'deliveryRate' : 0,
                'delay' : [],
                'association' : []
                }

        self.blockage = {}

    # Launches the initial service (process)
    def initializeServices(self, mobilityModel = None):
        self.env.process(self.measurementEvent())
        self.env.process(self.sendingPackets())

        if callable(mobilityModel):
            self.setMobilityModel(mobilityModel)

    # Configs the TTT, seting a time different from default value
    def configTTT(self, time : int):
        self.timeToTrigger = time

    # Sets a mobility model to the UE
    def setMobilityModel(self, newModel):
        try:
            callable(newModel)
            self.mobilityModel = newModel
            self.env.process(self.userMobility())
        except Exception as e:
            print(e)
            print('Not a callable function past')

    # Once the mobility model is set, at every time slot the UE position will
    # be updated by this method
    def userMobility(self):
        while True:
            yield self.env.timeout(1) #updates at the minimal time cell
            newx, newy = self.mobilityModel(self)
            self.x += newx
            self.y += newy



    ### This method updates the list of BS and their RSRP
    def updateListBS(self):
        #timePast = self.env.now
        #xnow = self.x + (self.Vx/3.6)*timePast*1e-3 # time is represented in milliseconds
        #ynow = self.y + (self.Vy/3.6)*timePast*1e-3 

        for uuid, bs in self.listBS.items():
            ##### NEED TO INCLUDE BLOCKAGE!!!
            #distance = np.hypot(xnow - bs.x, ynow - bs.y)
            distance = np.hypot(self.x - bs.x, self.y - bs.y)
            #distance = bs.calcUserDist(self)

            wavelength = 3e8/bs.frequency
            ### There are 2 reference signal at each OFDM symbol, and 4 OFDM
            ### symbols in a slot carrying reference signals
            referencesignals = bs.slotsPerSubframe*(bs.resourceBlocks*2)*4
            
            #NLOS condition
            if self.blockage[uuid][self.env.now] == 1:
                pathloss = 61.4 + 20.0*np.log10(distance)
                    #+ np.mean(np.random.normal(0,5.4,referencesignals)))
            #LOS condition
            else:
                pathloss = 72.0 + 29.2*np.log10(distance)
                    #+ np.mean(np.random.normal(0,8.7,referencesignals)))

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
            # Check if the UE is in sync with the BS
            self.env.process(self.signalQualityCheck())
            self.measurementCheck()
            yield self.env.timeout(self.timeToMeasure)


    def firstAssociation(self, baseStation):
        ### FIRST TIME USER ASSOCIATON
        #if self.servingBS == None and self.listedRSRP[maxRSRP] > self.sensibility:
        self.servingBS = baseStation
        self.kpi['association'].append([list(self.listBS.keys()).index(self.servingBS),
                                        self.env.now])
        
        # yields untill the downlink and uplink sync is completed
        yield self.env.timeout(
                self.listBS[self.servingBS].nextRach + defs.BURST_DURATION  - self.env.now
                )

        # Now, the UE is in sync with the serving BS
        self.sync = True

    def reassociation(self, baseStation):
        self.kpi['reassociation'] += 1
        self.reassociationFlag = True
        self.env.process(self.firstAssociation(baseStation))


    # Checks the handover event condition
    def measurementCheck(self):
        maxRSRP = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]
        for n,i in enumerate(self.listedRSRP.keys()):
            self.plotRecvRSRP[n].append(self.listedRSRP[i])

        ### FIRST TIME USER ASSOCIATON
        if self.servingBS == None and self.listedRSRP[maxRSRP] > self.sensibility:
            if self.lastBS:
                self.reassociation(maxRSRP)
            else:
                self.env.process(self.firstAssociation(maxRSRP))
        

        elif self.servingBS != None:
            # Check if the UE is in sync with the BS
            #self.env.process(self.signalQualityCheck())

            snr = self.listedRSRP[self.servingBS] - self.channel['noisePower']
            #print(self.env.now, maxRSRP, self.listedRSRP[maxRSRP], self.servingBS, self.listedRSRP[self.servingBS], snr)
            rate = self.listBS[self.servingBS].bandwidth*np.log2(1 + max(snr,0))
            self.kpi['capacity'].append(rate)

            # This is the condition trigger a handover
            if maxRSRP != self.servingBS:
                self.env.process(self.handover(maxRSRP))

        '''# User became out of sync or a handover failed due to loss of sync
        elif self.servingBS == None and len(self.lastBS)>0 and not self.sync:
            print('We are here now %d'%(self.env.now))
            self.reassociation(maxRSRP)
        #'''

        self.plotAssociation.append(self.servingBS)

        if self.servingBS == None:
            self.plotMaxRSRP.append(float('-inf'))
        else:
            self.plotMaxRSRP.append(self.listedRSRP[self.servingBS])#self.listedRSRP[maxRSRP])



    # Testing whether the handover did not fail due to be late
    # The UE sync with the serving BS will be tested
    def signalQualityCheck(self):
        if self.servingBS != None:
            #print(self.env.now, self.servingBS,self.listedRSRP[self.servingBS])
            if self.listedRSRP[self.servingBS] < self.qualityOut and self.qualityOutCounter == 0:
                downcounter = self.t310
                while downcounter > 0:
                    #print(self.env.now, downcounter)

                    if self.qualityOutCounter >= self.n310:
                        # Start out of sync counter
                        downcounter -= 1
                        yield self.env.timeout(1)

                    elif self.listedRSRP[self.servingBS] < self.qualityOut:
                        # Saves the number of time slots with low received RSRP                        
                        self.qualityOutCounter += 1                        
                        yield self.env.timeout(1)


                    # The channel is better now and the signal to serving BS
                    # got greater or equal the "quality in" param
                    if self.listedRSRP[self.servingBS] >= self.qualityIn:
                        self.qualityInCounter += 1

                        # The signal power is above quality in for more than
                        # n311 samples and t310 has not already expired
                        if self.qualityInCounter >= self.n311:
                            #Stop out of sync counter 
                            self.qualityOutCounter = 0
                            self.qualityInCounter = 0
                            downcounter = self.t310
                            break

                        else:
                            #Signal strength is better but the sync still
                            #unconfirmed by the N311 counter, so T310 continues
                            downcounter -= 1
                            yield self.env.timeout(1)

                # T310 expired, the UE is out of sync with the serving BS
                if downcounter == 0:
                    # out of sync!
                    self.sync = False

                    # Need to reassociate with the network
                    self.lastBS.append(self.servingBS)
                    self.reassociationFlag = False
                    self.servingBS = None

                self.qualityOutCounter = 0
                self.qualityInCounter = 0

        # The user is already out of sync!
        else:
            self.sync = False



    def handover(self, targetBS):
        counterTTT = 0

        if not self.sync:
            self.handoverFailure()

        # First, check if another measurement is not in progress
        elif not self.measOccurring:
            #targetBS = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]

            # If it is not, check whether it is an A3 event or not
            if self.listedRSRP[targetBS] - self.HOHysteresis > self.listedRSRP[self.servingBS] + self.HOOffset:
                # Given that it is an A3 event, triggers the measurement
                self.measOccurring = True
                self.triggerTime = self.env.now


                while counterTTT < self.timeToTrigger:
                    yield self.env.timeout(1)
                    counterTTT += 1

                    if self.sync:
                        # The A3 condition still valid? If not, stop the timer
                        if self.listedRSRP[targetBS] - self.HOHysteresis <= self.listedRSRP[self.servingBS] + self.HOOffset:
                            break
                    else:
                        self.handoverFailure()
                        break


                if counterTTT == self.timeToTrigger:
                    self.kpi['handover']+=1

                    if self.sync:
                        self.sendMeasurementReport(targetBS)

                    else:
                        self.handoverFailure()

                    self.measOccurring = False
                    self.triggerTime = 0


    # Send the measurement report to the BS
    # Actually this method is doing all the handover procedure at once
    # To be more realistic it would need to send messages to the serving and
    # target BS, but it is not yet implemented
    def sendMeasurementReport(self, targetBS):
        if (self.env.now >= self.triggerTime + self.timeToTrigger) and self.measOccurring:

            #Check if it is not a reassociation
            if self.listedRSRP[self.servingBS] != None:

                # Check if it is a pingpong, just for kpi assessment
                if self.lastBS.count(targetBS)>0:
                    self.kpi['pingpong'] += 1

                # Switch to the new BS
                self.switchToNewBS(targetBS)

    def handoverFailure(self):
        self.lastBS.append(self.servingBS)
        self.servingBS = None
        self.reassociationFlag = False
        self.kpi['handoverFail'] += 1


    def switchToNewBS(self, targetBS):
        self.lastBS.append(self.servingBS)
        self.servingBS = targetBS
        self.kpi['association'].append([list(self.listBS.keys()).index(self.servingBS), self.env.now])
        #Needs to implement the gap between the user is in total sync with new BS


    # This method schedules the packets transmission
    def sendingPackets(self):
        for t in self.packetArrivals:
            yield self.env.timeout(t - self.env.now)

            # The packet will be sent if and only if the UE is in sync with the
            # serving BS and, of course, if it is associated whatsoever
            if self.sync and self.servingBS != None:
                #snr = self.listedRSRP[self.servingBS] - self.channel['noisePower']
                snr = max(0, self.listedRSRP[self.servingBS] - self.channel['noisePower'])
                temp = self.snrThreshold
                timer = 0 

                # There is a 10 milliseconds tolerance to send a packet
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


    # This method processes the LOS info from the instance
    def addLosInfo(self, los : list, n : int) -> list:
        for m,bs in enumerate(self.listBS):
            self.blockage[bs] = los[m][n]


    # This method plots the UE KPIs
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


                        

    # This method prints the kpi dict 
    def printKPI(self):
        self.kpi['uuid'] = self.uuid
        self.kpi['partDelay'] /= self.nPackets
        self.kpi['throughput'] = np.mean(self.kpi['throughput'])
        self.kpi['deliveryRate'] /= self.nPackets
        self.kpi['delay'] = np.mean(self.kpi['delay'])
        self.kpi['capacity'] = np.mean(self.kpi['capacity'])

        if self.kpi['handover'] > 0:
            self.kpi['pingpong'] /= self.kpi['handover']
            self.kpi['handoverFail'] /= self.kpi['handover']
        else:
            self.kpi['pingpong'] = 0
            self.kpi['handoverFail'] = 0

        print(json.dumps(self.kpi, indent=4))        



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
parser.add_argument('--ttt', type=int, default=640)
args = parser.parse_args()


def straightRoute(device : MobileUser):
    ### assumes the speed in km/h and the time in milliseconds
    x = (device.Vx/3.6)*1e-3
    y = (device.Vy/3.6)*1e-3
    return x, y



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
        baseStations[i['uuid']].initializeServices()

    sim.addBaseStations(baseStations)

    mobiles = {}
    ### Creating a list of nodes
    for n, i in enumerate(nodes):
        mobiles[i['uuid']] = MobileUser(i, sim)
        mobiles[i['uuid']].initializeServices(straightRoute)
        mobiles[i['uuid']].configTTT(args.ttt)
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

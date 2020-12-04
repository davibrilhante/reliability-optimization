#! /usr/bin/env python3
# -*- coding = utf-8 -*-

import numpy as np
import argparse
import json
import operator
import sys
from collections import OrderedDict
from itertools import product

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
        self.id = bsDict['uuid']
        self.uuid = bsDict['uuid']
        self.frequency = bsDict['frequency']
        self.env = scenario.env
        self.channel = scenario.channel
        self.simTime = scenario.simTime
        self.range = 150

        self.txPower = bsDict['txPower']
        self.antennaGain = 10 #### STUB!!!

        self.bandwidth = 12*bsDict['resourceBlocks']*bsDict['subcarrierSpacing']
        self.resourceBlocks = bsDict['resourceBlocks']

        self.associatedUsers = []
        self.inRangeUsers = []
        self.frameIndex = 0
        self.ssbIndex = 1#0

        self.onSSB = False
        self.nextSSB = 0
        self.onRach = False
        self.nextRach = 0

        self.numerology = components.numerology(bsDict['subcarrierSpacing']/1e3)
        self.slotsPerSubframe = bsDict['subcarrierSpacing']/(15*1e3)

        self.associationDelay = 0


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
                '''print('A new burst set is starting at %d and it is the %d ss burst in %d frame' % 
                        (self.env.now, self.ssbIndex, self.frameIndex))'''

                ### Flagging that there is an SSB happening
                self.onSSB = True

                ### Verifies whether the next SSB is a RACH
                if ((self.frameIndex + (burstPeriod/defs.FRAME_DURATION))%(rachPeriod/defs.FRAME_DURATION) != 0):
                    ### No, the next is not a RACH
                    self.nextSSB = self.env.now + (burstPeriod - burstDuration)
                else:
                    ### Yes, the next is a RACH
                    self.nextSSB = self.env.now + (2*burstPeriod - burstDuration)
                    #print('Next one is a RACH, next SSB in %d #%d' % (self.nextSSB, self.ssbIndex))

                yield self.env.timeout(burstDuration)
                #print('The burst set has finished at %d' % self.env.now)
                #self.calcNetworkCapacity()

                ### Flagging that the SSB is finished
                self.onSSB = False

                yield self.env.timeout(burstPeriod - burstDuration)

            else:
                ### The previous was a RACH
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
                '''print('A new rach opportunity is starting at %d and it is the %d ss burst in %d frame' 
                        % (self.env.now, self.ssbIndex, self.frameIndex))'''

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

    def setAssociationDelay(self, delay):
        self.associationDelay = delay

    def setRange(self, _range):
        self.range = _range


    # This method processes the LOS info from the instance
    def addLosInfo(self, los : list, m : int) -> list:
        self.blockage = los[m]

    def setPrediction(self, delta = 40, method='slot', **params):
        if method not in ('slot', 'image'):
            print('Chosen method not available')
            exit()
        else:
            self.predictionWindow = delta
            self.method = method

            self.criterea = params.get('criterea','AvgDelay')
            self.handoverInterruptionTime = params.get('interruptionTime',17)
            self.windowLen = params.get('windowLen',100)
            self.alpha = params.get('alpha',0.1)

            '''
            Variables that define the accuracy drop on the prediction

            Precision : The time when the accuracy begins to drop
            Decay: The type of the decay, that can be linear or exponential
            Factor: The decay factor, a quotient in the case of linear and
                    a exponent in the case of exponential decay
            '''
            self.precision = params.get('precision',self.predictionWindow)
            self.decay = params.get('decay', 'linear')
            self.factor = params.get('factor', 0)

            # Exponent of the prediction low pass 
            self.exponent = params.get('exponent',0)

            if method == 'slot':
                self.positive = params.get('positive',1)
                self.negative = params.get('negative',1)
                if (self.positive or self.negative) > 1:
                    print('Probability values greater than 1')
                    exit()
                elif (self.positive or self.negative) < 0:
                    print('Probability values lower than 0')
                    exit()

            elif method == 'image':
                self.occurrence = params.get('occurrence')
                self.duration = params.get('duration')
        

    '''def predictBlockageSlot(self, user, positive = self.positive, 
            negative = self.negative, time = self.env.now):'''
    def predictBlockageSlot(self, user, time = None):

        #print('predicting blockage for %s'%(self.uuid))

        if time == None:
            time = self.env.now
        predicted = []
        falsePositive = 1 - self.negative
        falseNegative = 1 - self.positive

        groundTruth = self.blockage[user.blkId][time:time+self.predictionWindow]

        for t in groundTruth:
            if t == 1:
                predicted.append(np.random.choice([0,1], p=[falsePositive, self.negative]))
            elif t == 0:
                predicted.append(np.random.choice([0,1], p=[self.positive, falseNegative]))
        #print(groundTruth)
        #print(predicted)

        return predicted

    '''def predictBlockageImage(self, user, stdevOccurrence = self.occurrence, 
            stdevDuration= self.duration, time = self.env.now):'''
    def predictBlockageImage(self, user, time = None):
        if time == None:
            time = self.env.now

        predicted = []

        groundTruth = self.blockage[user.blkId][time:time+self.predictionWindow]
        blockage = [n for n,t in enumerate(groundTruth) if t == 0]

        start = blockage[0]
        last = 0
        for n,t in enumerate(blockage[1:]):
            if blockage[n+1] - t > 1:
                trueDuration = t - start
                
                beginError = start + int(np.random.normal(0, self.occurrence))
                durationError =  start + trueDuration + int(np.random.normal(0, self.duration))

                predicted += [1 for i in range(last, beginError)] + [0 for i in range(beginError,durationError+1)]
                start = blockage[n+1]
                last = durationError + 1
            
        return predicted

    def addNeighbourhood(self, scenario):
        self.neighbours = {}
        self.metric = {}
        self.metric['delayMovAvg'] = {}
        self.countPredictions = {}
        for uuid, bs in scenario.baseStations.items():
            self.metric['delayMovAvg'][uuid] = []
            self.countPredictions[uuid] = 0
            if uuid != self.id:
                self.neighbours[uuid] = bs


    def identifyClosestNeighbours(self, ue, nOptions):

        #print('%s identifying neighbours'%(self.uuid))

        candidates = self.neighbours.items()
        selected = []

        #ueAngle = np.arctan2(ue.Vy, ue.Vx)
        tan = ue.Vy/ue.Vx

        #limitX = ue.x + (ue.y/tan)
        limitX = ue.x #+ ((ue.y**2)/ue.x)
        #limitY = ue.y + (ue.x*tan)
        limitY = ue.y #+ ((ue.x**2)/ue.y)
        #print(limitX, limitY, ue.x, ue.y)

        distance = []
        direction = []
        for bs in self.neighbours.values():

            #if bs.x < limitX and bs.y < limitY:
            if ((bs.x + np.sign(ue.Vx)*self.range < limitX)
                    and (bs.y + np.sign(ue.Vy)*self.range < limitY)):
                dist = bs.calcUserDist(ue)
                continue
            else:
                #dist = self.neighbours[bs].calcUserDist(ue)
                dist = bs.calcUserDist(ue)
                #print(bs.uuid, dist)

                if distance and max(distance) > dist:
                    if len(selected) < nOptions:
                        selected.append(bs)
                        #distance.append(bs)
                        distance.append(dist)
                    else:
                        index = np.argmax(distance) #distance.index(max(distance))
                        distance.pop(index)
                        selected.pop(index)

                        selected.append(bs)
                        distance.append(dist)

                elif not distance:
                    selected.append(bs)
                    #distance.append(bs)
                    distance.append(dist)

        return selected
        

    def requestBlockagePrediction(self, ue, nOptions=1, method='slot'):

        #print('%s requesting blockage prediction from neighbours'%(self.uuid))

        if method not in ('slot', 'image'):
            print('Chosen method not valid')
            exit()

        candidates = self.identifyClosestNeighbours(ue, nOptions) 

        predictions = {}

        for bs in candidates:
            # Need to solve parametes passed to predictBlockageSlot
            predictions[bs.uuid] = []
            self.countPredictions[bs.uuid] += 1
            if method == 'slot':
                #predicted = self.neighbours[bs].predictBlockageSlot(ue)
                predicted = bs.predictBlockageSlot(ue)
            elif method == 'image':
                #predicted = self.neighbours[bs].predictBlockageImage(ue)
                predicted = bs.predictBlockageImage(ue)

            for k, t in enumerate(predicted):
                ueX = ue.x + ue.Vx*3.6*t*1e-3
                ueY = ue.y + ue.Vx*3.6*t*1e-3
                '''distance = np.hypot(self.neighbours[bs].x - ueX, 
                            self.neighbours[bs].y -ueY)'''
                distance = np.hypot(bs.x - ueX, bs.y -ueY)
                if t == 1:
                    pathloss = 61.4 + 20.0*np.log10(distance)
                elif t == 0:
                    pathloss = 72.0 + 29.2*np.log10(distance)
                '''predictions[bs].append(self.neighbours[bs].txPower + ue.antennaGain +
                                    self.neighbours[bs].antennaGain - pathloss)'''

                #predictions[bs.uuid].append(bs.txPower + ue.antennaGain + bs.antennaGain - pathloss)
                receivedPower = bs.txPower + ue.antennaGain + bs.antennaGain - pathloss

                if k == 0: 
                    predictions[bs.uuid].append(receivedPower)
                else:
                    filtered = predictions[bs.uuid][-1]*(1 - np.exp(-1*self.exponent))
                    predictions[bs.uuid].append(receivedPower*np.exp(-1*self.exponent) + filtered)


        return predictions


    def enumerateHandoverConfigurations(self, ue, nOptions, method = 'slot'):
        #print('%s enumerating handover configurations'%(self.uuid))

        ### Requesting the predictions to the neighbour BS
        predictions = self.requestBlockagePrediction(ue, nOptions, method)

        ### Creating the prediction to the serving BS
        # Need to solve parametes passed to predictBlockageSlot
        if method not in ('slot', 'image'):
            print('Chosen method not valid')
            exit()

        if method == 'slot':
            selfPrediction = self.predictBlockageSlot(ue)
        elif method == 'image':
            selfPrediction = self.predictBlockageImage(ue)

        predictions[self.uuid] = []
        self.countPredictions[self.uuid] += 1
        for t in selfPrediction:
            ueX = ue.x + ue.Vx*3.6*t*1e-3
            ueY = ue.y + ue.Vx*3.6*t*1e-3
            distance = np.hypot(self.x - ueX, self.y -ueY)
            if t == 1:
                pathloss = 61.4 + 20.0*np.log10(distance)
            elif t == 0:
                pathloss = 72.0 + 29.2*np.log10(distance)
            predictions[self.uuid].append(self.txPower + ue.antennaGain + self.antennaGain - pathloss)

        candidates = list(predictions.keys())
        #values = list(zip(list(predictions.values())[0], list(predictions.values())[1]))
        values = []
        for t in range(len(list(predictions.values())[0])):
            temp = []
            for i in candidates:
                temp.append(predictions[i][t])
            values.append(tuple(temp))

        ### Evaluating the possible handover opportunities
        opportunities = []
        end = min(self.predictionWindow, self.simTime - self.env.now)
        for t in range(1, end):#self.predictionWindow):
            for n, bs in enumerate(candidates):
                ueX = ue.x + ue.Vx*3.6*t*1e-3
                ueY = ue.y + ue.Vx*3.6*t*1e-3
                if bs != self.uuid:
                    distance = np.hypot(self.neighbours[bs].x - ueX, 
                                self.neighbours[bs].y -ueY)
                else:
                    distance = np.hypot(self.x - ueX, self.y - ueY)

                ### Each time a BS is blocked, there is an opportunity to make handover
                #print(t, abs(values[t][n] - values[t-1][n]), (10.6 + 9.2*np.log10(distance)))
                if abs(values[t][n] - values[t-1][n]) >= 0.9*(10.6 + 9.2*np.log10(distance)):
                    #there is a blockage here
                    if t not in opportunities:
                        opportunities.append(t)

        if not opportunities:
            opportunities.append(end)


        enumerated = {}

        ### Enumerates all the possible combinations with repetitions among the
        ### BS through all the HO opportunities identified
        configs = product(candidates, repeat=len(opportunities))

        for n, p in enumerate(configs):
            receivedPower = []
            for bs in p:
                begin = 0
                for end in opportunities:
                    for t in range(begin, end):
                        ### The received power per slot of all the HO configurations
                        receivedPower.append(values[t][candidates.index(bs)])
            if opportunities:
                #for t in range(opportunities[-1], self.predictionWindow):
                for t in range(opportunities[-1], min(self.predictionWindow, self.simTime - self.env.now)):
                    receivedPower.append(values[t][candidates.index(p[-1])])


            ### This dict is the struct that stores the HO configs and its
            ### respective received power per slot
            enumerated[n] = [list(p), receivedPower]

        return enumerated, opportunities



    def chooseHandoverConfiguration(self, ue, nOptions, **params):
        #method= 'slot', criterea='power', handoverInterruptionTime = 17):

        #print('%s starting blockage prediction'%(self.uuid))

        #method = params.get('method','slot')

        if self.criterea not in ('power', 'cost', 'offperiod', 
                'coverage', 'features', 'AvgDelay', 'all'):
            print('Chosen criterea not available')
            exit()

        configurations, opportunities = self.enumerateHandoverConfigurations(ue, nOptions, self.method)
        #print(opportunities)
        
        selected = {}
        evaluation = {}
        #Analyze according wth given critereas
        if self.criterea == 'power' or self.criterea == 'all':
            evaluation['power'] = []
            for (cfg, recvPower) in configurations.values():
                #print(recvPower)
                evaluation['power'].append(np.mean(recvPower))
            #print(evaluation)
            selected['power'] = zip(configurations[np.argmax(evaluation['power'])][0], opportunities)
            #selected = [(x, y) for x, y in zip(configurations[np.argmin(evaluation)][0], opportunities)]


        if self.criterea == 'offperiod' or self.criterea == 'AvgDelay' or self.criterea == 'all':
            evaluation['offperiod'] = []
            evaluation['AvgDelay'] = []

            candidates = {}
            candidates[self.uuid]=[]
            for bs in self.neighbours.keys():
                candidates[bs]=[]

            for cfg, recvPower in configurations.values():
                offperiod = 0
                counter = 0
                actual = self.uuid
                begin = 0
                handover = False

                temp = 0
                cfgDelay = {}
                for bs in cfg:
                    cfgDelay[bs] = 0

                for m, (bs, end) in enumerate(zip(cfg, opportunities)):
                    delay = 0

                    #print(m, bs, begin, end)
                    for pw in recvPower[begin:end]:
                        if bs != actual and not handover:
                            offperiod += self.handoverInterruptionTime
                            delay += self.handoverInterruptionTime

                            counter = self.handoverInterruptionTime - 1
                            handover = True
                            actual = bs
                            continue

                        elif handover and counter != 0:
                            counter -= 1
                            continue

                        elif counter == 0:
                            handover = False

                        snr = max(0, pw - self.channel['noisePower'])
                        #print(snr, pw)

                        if bs != self.uuid:
                            capacity = self.neighbours[bs].bandwidth*np.log2(1 + snr)
                        else:
                            capacity = self.bandwidth*np.log2(1 + snr)

                        if capacity < ue.capacity:
                            offperiod += 1
                            delay += 1
                    #END for pw in recvPower[begin:end]

                    '''
                    self.metric['delayMovAvg'][bs] = (
                            (delay + (self.env.now+begin)*self.metric['delayMovAvg'][bs])/(self.env.now+end)
                            )

                    self.metric['delayMovAvg'][bs] = (
                            (delay + (self.countPredictions[bs]-1)*self.metric['delayMovAvg'][bs])/
                            self.countPredictions[bs]
                            )
                    #'''


                    cfgDelay[bs] += delay

                    end = begin

                # for m, (bs, end) in enumerate(zip(cfg, opportunities)):

                evaluation['AvgDelay'].append(0)
                for bs in cfg:
                    candidates[bs].append(cfgDelay[bs])

                    if len(self.metric['delayMovAvg'][bs]) > self.windowLen:
                        self.metric['delayMovAvg'][bs].pop(0)

                    samples = self.metric['delayMovAvg'][bs].copy()
                    samples.append(cfgDelay[bs])

                    '''
                    Moving average calc

                    EWMA_n = S_n + (1-a)S_n-1 + (1-a)^2 S_n-2 + ... + (1-a)^w S_n-w
                             ------------------------------------------------------
                                    1 + (1-a) + (1-a)^2 + ... + (1-a)^w

                    Temp stores the sum of the moving average of each BS in this configuration
                    '''

                    num = 0
                    den = 0
                    for w, val in enumerate(samples):
                        num += ((1-self.alpha)**(self.windowLen - w))*val
                        den += ((1-self.alpha)**(self.windowLen - w))

                    evaluation['AvgDelay'][-1] += num/den
                    
                    #temp +=  self.metric['delayMovAvg'][bs]#*(end - begin)/(self.predctionWindow)
                evaluation['offperiod'].append(offperiod)



            #print(evaluation)

            #selected = [configurations[np.argmin(evaluation)][0], opportunities]
            selected['offperiod'] = zip(configurations[np.argmin(evaluation['offperiod'])][0], opportunities)
            selected['AvgDelay'] = zip(configurations[np.argmin(evaluation['AvgDelay'])][0], opportunities)

            for bs in list(self.neighbours.keys())+[self.uuid]:
                if candidates[bs]:
                    self.metric['delayMovAvg'][bs].append(max(candidates[bs]))



        if self.criterea == 'coverage' or self.criterea=='all':
            1

        if self.criterea == 'features' or self.criterea == 'all':
            1

        if self.criterea == 'all':
            score = []
            weight = [1/4, 1/4, 1/2]
            maxPwr = max(evaluation['power'])
            maxDly = max(evaluation['AvgDelay'])
            for n, (cfg,_) in enumerate(configurations.values()):
                score.append(
                        weight[0]*(evaluation['power'][n]/maxPwr) +
                        weight[1]*(1 - evaluation['offperiod'][n]/self.predictionWindow) +
                        weight[2]*(1 - evaluation['AvgDelay'][n]/maxDly)
                        )

            selected['all'] = zip(configurations[np.argmax(score)][0], opportunities)
                        

        #print(self.metric['delayMovAvg'])

        return selected[self.criterea]

              


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
                #'hoAttempts': []
                }

        self.blockage = {}

        self.prediction = False

    # Launches the initial service (process)
    def initializeServices(self, mobilityModel = None):
        self.env.process(self.measurementEvent())
        self.env.process(self.sendingPackets())
        self.env.process(self.requestPrediction())

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
        #self.kpi['association'].append([list(self.listBS.keys()).index(self.servingBS),0])
        self.kpi['association'].append([list(self.listBS.keys()).index(self.servingBS),
            self.env.now])
        
        # yields untill the downlink sync is completed
        yield self.env.timeout(
                self.listBS[self.servingBS].nextSSB + defs.BURST_DURATION  - self.env.now
                )
        
        # yields untill the uplink sync is completed
        yield self.env.timeout(
                self.listBS[self.servingBS].nextRach + defs.BURST_DURATION  - self.env.now
                )

        # yields for the BS association delay
        yield self.env.timeout(self.listBS[self.servingBS].associationDelay)

        # Now, the UE is in sync with the serving BS
        self.sync = True

    def reassociation(self, baseStation):
        self.kpi['reassociation'] += 1
        self.reassociationFlag = True
        self.env.process(firstAssocation(baseStation))


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
            if maxRSRP != self.servingBS and not self.prediction:
                self.env.process(self.handover(maxRSRP))


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
                self.env.process(self.switchToNewBS(targetBS))

    def handoverFailure(self):
        self.lastBS.append(self.servingBS)
        self.servingBS = None
        self.reassociationFlag = False
        self.kpi['handoverFail'] += 1

        # It seems there is a sync=False here
        self.sync = False


    def switchToNewBS(self, targetBS):
        #print('Switching from %s to %s'%(self.servingBS, targetBS))
        self.lastBS.append(self.servingBS)
        self.servingBS = targetBS
        self.kpi['association'].append([list(self.listBS.keys()).index(self.servingBS), self.env.now])

        # The time gap between the disassociation from the Serving BS to the
        # target BS is known as handover interruption time (HIT) and it is
        # the for the UE to get synced with the target BS. There is no data
        # connection during this time interval, so the UE remains unsynced
        self.sync = False 

        '''
        print(self.listBS[self.servingBS].nextSSB, self.listBS[self.servingBS].frameIndex, self.env.now)
        print(defs.BURST_DURATION)
        print(self.listBS[self.servingBS].nextSSB + defs.BURST_DURATION  - self.env.now)
        '''

        # yields untill the downlink sync is completed
        yield self.env.timeout(
                self.listBS[self.servingBS].nextSSB + defs.BURST_DURATION  - self.env.now
                )

        #print(self.listBS[self.servingBS].nextRach, self.env.now)

        # yields untill the uplink sync is completed
        yield self.env.timeout(
                self.listBS[self.servingBS].nextRach + defs.BURST_DURATION  - self.env.now
                )

        # yields for the BS association delay
        yield self.env.timeout(self.listBS[self.servingBS].associationDelay)

        # Now the UE is up and downlink synced
        self.sync = True


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
            self.blkId = n

    def setPredictions(self, period = 40, nOptions=1, **params): #criterea='power'):
        self.prediction = True
        self.predictionPeriod = period
        self.neighbourOptions = nOptions
        #self.criterea = criterea

    def requestPrediction(self):
        while True:
            yield self.env.timeout(self.predictionPeriod)
            #config = self.listBS[self.servingBS].chooseHandoverConfiguration(self, self.neighbourOptions, criterea=self.criterea)
            config = self.listBS[self.servingBS].chooseHandoverConfiguration(self, self.neighbourOptions)
            ### Processing configuration
            actual = self.listBS[self.servingBS].uuid
            for bs, t in config:
                #print(self.env.now, actual, bs, t)
                ### There is a handover to be made
                if bs != actual:
                    self.kpi['handover'] += 1
                    if bs in self.lastBS:
                        self.kpi['pingpong'] += 1

                    actual = bs
                    if t > 0:
                        yield self.env.timeout(t)


                    ### Check whether the UE still has connection with the serving BS
                    if self.sync:
                        self.env.process(self.switchToNewBS(bs))
                    else:
                        #print('Xiiii', self.env.now, self.listBS[self.servingBS].uuid, bs)
                        self.handoverFailure()

                else:
                    continue



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
                    if n == 0 or n == 1:
                        continue
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
        #self.kpi['throughput'] = np.mean(self.kpi['throughput'])
        self.kpi['deliveryRate'] /= self.nPackets
        #self.kpi['delay'] = np.mean(self.kpi['delay'])

        if self.kpi['handover'] > 0:
            self.kpi['pingpong'] /= self.kpi['handover']
            self.kpi['handoverFail'] /= self.kpi['handover']

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
parser.add_argument('-p','--plot', action='store_true', help='plots something')
parser.add_argument('--ttt', type=int, default=640, help='Time to trigger value')
parser.add_argument('--delay', type=int, default=0, help='Base Stations association delay')
parser.add_argument('--predWindow', type=int, default=50, help='Defines the blockage prediction window')
parser.add_argument('--reqPeriod', type=int, default=50, help='Blockage prediction request period')
parser.add_argument('--windowLen', type=int, default=10, help='moving average of given prediction critereon')
parser.add_argument('--alpha', type=float, default=0.1, help='moving average of given prediction critereon')


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
    for n, i in enumerate(network):
        baseStations[i['uuid']] = BaseStation(i, sim)
        baseStations[i['uuid']].initializeServices()
        baseStations[i['uuid']].setAssociationDelay(args.delay)
        baseStations[i['uuid']].addLosInfo(LOS, n)
        baseStations[i['uuid']].setPrediction(
                delta = args.predWindow, positive=1.0, negative=1.0,
                windowLen=args.windowLen, alpha=args.alpha)


    sim.addBaseStations(baseStations)
    for n, i in enumerate(network):
        baseStations[i['uuid']].addNeighbourhood(sim)


    mobiles = {}
    ### Creating a list of nodes
    for n, i in enumerate(nodes):
        mobiles[i['uuid']] = MobileUser(i, sim)
        mobiles[i['uuid']].initializeServices(straightRoute)
        mobiles[i['uuid']].configTTT(args.ttt)
        mobiles[i['uuid']].addLosInfo(LOS, n)
        mobiles[i['uuid']].setPredictions(period=args.reqPeriod, nOptions=1)


    env.run(until=scenario['simTime'])

    for _,ue in mobiles.items():
        ue.printKPI()

        if args.plot:
            #ue.plot('Association')
            ue.plot('MAX_RSRP')
            #ue.plot('RECV_RSRP', False)
            #ue.plot('SNR', False)
            #ue.plot('Capacity', False)

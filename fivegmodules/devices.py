from numpy import hypot
from numpy import log10
from numpy import log2
from numpy import mean
from numpy.random import uniform
from numpy import pi
import operator

from json import dumps

from fivegmodules.handover import Handover
from fivegmodules.core import Scenario
from fivegmodules.core import Channel
from fivegmodules.core import WirelessDevice
from fivegmodules.mobility import MobilityModel
from fivegmodules.mobility import StraightRoute


__all__ = ['WirelessDevice','BaseStation', 'MeasurementDevice', 'MobileUser', 
            'PredictionBaseStation', 'SignalAssessmentPolicy']


class BaseStation(WirelessDevice):
    def __init__ (self, scenario, bsDict):
        super(BaseStation, self).__init__(scenario, bsDict)

        self.resourceBlocks = bsDict['resourceBlocks']
        self.slotsPerSubframe = bsDict['subcarrierSpacing']/(15*1e3)
        self.bandwidth = 12*bsDict['resourceBlocks']*bsDict['subcarrierSpacing']

        self.associatedUsers = []
        self.inRangeUsers = []
        self.frameIndex = 0
        self.ssbIndex = 1

        self.onSSB = False
        self.nextSSB = 0
        self.onRach = False
        self.nextRach = 0

        self.networkParameters = None
        self.numerology = None
        self.associationDelay = 10

    def initializeServices(self, **params):
        self.env.process(self.updateFrame())
        self.env.process(
                        self.burstSet(
                                    self.networkParameters.SSBurstDuration, 
                                    self.networkParameters.SSBurstPeriod, 
                                    self.networkParameters.RACHPeriod
                                    )
                        )
        self.env.process(
                        self.rachOpportunity(
                                            self.networkParameters.SSBurstDuration, 
                                            self.networkParameters.RACHPeriod
                                            )
                        )


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
            self.availableSlots = self.numerology.SSBlocks
            if (self.frameIndex % (rachPeriod/self.networkParameters.frameDuration) != 0) and (self.frameIndex != 1):
                '''print('A new burst set is starting at %d and it is the %d ss burst in %d frame' %
                        (self.env.now, self.ssbIndex, self.frameIndex))'''

                ### Flagging that there is an SSB happening
                self.onSSB = True

                ### Verifies whether the next SSB is a RACH
                if (
                        (
                            self.frameIndex + (burstPeriod/self.networkParameters.frameDuration)
                        )
                        %(rachPeriod/self.networkParameters.frameDuration) != 0
                    ):
                    ### No, the next is not a RACH
                    self.nextSSB = self.env.now + (burstPeriod - burstDuration)
                else:
                    ### Yes, the next is a RACH
                    self.nextSSB = self.env.now + (2*burstPeriod - burstDuration)
                    #print('Next one is a RACH, next SSB in %d #%d' % (self.nextSSB, self.ssbIndex))

                yield self.env.timeout(burstDuration)

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
            yield self.env.timeout(self.networkParameters.frameDuration)
            if self.frameIndex % (
                        self.networkParameters.SSBurstPeriod/self.networkParameters.frameDuration) == 0:
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

                yield self.env.timeout(rachPeriod - rachDuration)



class SignalAssessmentPolicy:
    def signalAssessment(self, device):
        raise NotImplementedError

class SynchronizationAssessment(SignalAssessmentPolicy):
    def __init__ (self):
        self.qualityOutCounter = 0    
        self.qualityInCounter = 0

    # Testing whether the handover did not fail due to be late
    # The UE sync with the serving BS will be tested
    def signalAssessment(self, device):
        if device.servingBS != None:
            if device.listedRSRP[device.servingBS] < device.networkParameters.qualityOut and self.qualityOutCounter == 0:
                downcounter = device.networkParameters.T310
                while downcounter > 0:
                    #print(device.env.now, device.listedRSRP[device.servingBS], downcounter)

                    if device.servingBS == None:
                        break

                    else:
                        if self.qualityOutCounter >= device.networkParameters.N310:
                            # Start out of sync counter
                            downcounter -= 1
                            #print('downcounter',downcounter)
                            device.T310running = True
                            yield device.env.timeout(1)
     
                        elif device.listedRSRP[device.servingBS] < device.networkParameters.qualityOut:
                            # Saves the number of time slots with low received RSRP                        
                            self.qualityOutCounter += 1
                            #print(device.env.now, 'out counter', self.qualityOutCounter, downcounter)
                            yield device.env.timeout(1)
     
     
                        # The channel is better now and the signal to serving BS
                        # got greater or equal the "quality in" param
                        elif device.listedRSRP[device.servingBS] >= device.networkParameters.qualityIn:
                            self.qualityInCounter += 1
                            #print(device.env.now, 'in counter', self.qualityInCounter, downcounter)

                            # If downcounter is not triggered, then there is no need 
                            # to carry on the synchronization assessment
                            if self.qualityOutCounter < device.networkParameters.N310:
                                downcounter -= 1
                                #print('downcounter',downcounter)
                                yield device.env.timeout(1)
     
                            # The signal power is above quality in for more than
                            # n311 samples and t310 has not already expired
                            if self.qualityInCounter >= device.networkParameters.N311:
                                #Stop out of sync counter 
                                self.qualityOutCounter = 0
                                self.qualityInCounter = 0
                                device.T310running = False
                                downcounter = device.networkParameters.T310
                                break

                            else:
                                #Signal strength is better but the sync still
                                #unconfirmed by the N311 counter, so T310 continues
                                downcounter -= 1
                                yield device.env.timeout(1)

                        # RSRP between quality in and quality out
                        else:
                            downcounter -= 1
                            #print('downcounter',downcounter)
                            yield device.env.timeout(1)

 
                # T310 expired, the UE is out of sync with the serving BS
                if downcounter == 0:
                    # out of sync!
                    device.sync = False
                    device.kpi.outofsync += 1
 
                    # Need to reassociate with the network
                    device.lastBS.append(device.servingBS)
                    #device.reassociationFlag = False
                    device.servingBS = None
                    device.T310running = False
                    #print(device.env.now)
 
                self.qualityOutCounter = 0
                self.qualityInCounter = 0
 
        # The user is already out of sync!
        else:
            device.sync = False
            device.kpi.outofsync += 1
            #print(device.env.now)

class HandoverAssessment(SignalAssessmentPolicy):
    def signalAssessment(self, device):
        maxRSRP = max(device.listedRSRP.items(), key=operator.itemgetter(1))[0]
        #for n,i in enumerate(device.listedRSRP.keys()):
        #    device.plotRecvRSRP[n].append(device.listedRSRP[i])
 
        ### FIRST TIME USER ASSOCIATON
        if device.servingBS == None and device.listedRSRP[maxRSRP] > device.sensibility:
            if device.lastBS:
                device.reassociation(maxRSRP)
            #else:
            #    device.env.process(device.firstAssociation(maxRSRP))
 
 
        elif device.servingBS != None and device.sync:
            snr = device.listedRSRP[device.servingBS] - device.channel.noisePower
            #print(self.env.now, maxRSRP, self.listedRSRP[maxRSRP], self.servingBS, self.listedRSRP[self.servingBS], snr)
            rate = device.scenarioBasestations[device.servingBS].bandwidth*log2(1 + max(snr,0))
            device.kpi.capacity.append(rate)
 
            # This is the condition trigger a handover
            if maxRSRP != device.servingBS:
                device.env.process(device.handover.triggerMeasurement(device, maxRSRP))



class MeasurementDevice(WirelessDevice):
    def __init__(self, scenario, inDict):
        super(MeasurementDevice, self).__init__(scenario, inDict)

        self.networkParameters = None
        self.scenarioBasestations = self.env.baseStations
        self.listedRSRP = { }
        self.servingBS = None
        self.lastBS = []

        self.sync = False
        self.handover = None
        self.syncAssessment = SynchronizationAssessment()
        self.bsAssessment = HandoverAssessment()
        self.kpi = KPI() 

        self.measOccurring = False
        self.T310running = False
        self.triggerTime = 0


    # At each time to measure, the UE updates the list of BS and check
    # if the handover event is happening
    def measurementEvent(self):
        while True:
            self.updateBSList()
            self.bsAssessment.signalAssessment(self)
            self.env.process(self.syncAssessment.signalAssessment(self))
            yield self.env.timeout(self.networkParameters.timeToMeasure)

    # This method processes the LOS info from the instance
    def addLosInfo(self, los : list, n : int):
        for m,bs in enumerate(self.scenarioBasestations):
            self.lineofsight[bs] = los[m][n]

    def firstAssociation(self, baseStation = None):
        ### FIRST TIME USER ASSOCIATON
        if baseStation == None:
            self.updateBSList()
            baseStation = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]

        #print(self.env.now, self.scenarioBasestations[baseStation].nextSSB)

        yield self.env.timeout(             
                    self.scenarioBasestations[baseStation].nextSSB + 
                    self.networkParameters.SSBurstDuration  - self.env.now
                    )

        if self.listedRSRP[baseStation] > self.networkParameters.qualityOut: 
            # yields untill the downlink and uplink sync is completed
            yield self.env.timeout(
                    self.scenarioBasestations[baseStation].nextRach 
                    + self.networkParameters.SSBurstDuration  - self.env.now
                    )
            
            if self.listedRSRP[baseStation] > self.networkParameters.qualityOut: 
                # Now, the UE is in sync with the serving BS
                self.sync = True
                self.servingBS = baseStation
                self.kpi.association.append([list(self.scenarioBasestations.keys()).index(self.servingBS),
                                                self.env.now])
                #print('Association complete!', self.env.now)

    def reassociation(self, baseStation):
        self.kpi.reassociation += 1
        self.reassociationFlag = True
        self.env.process(self.firstAssociation(baseStation))

    def servingBSSINR(self):
        interference = 0

        if self.servingBS == None:
            return 0

        else:
            for uuid, bs in self.scenarioBasestations.items():
                if uuid == self.servingBS:
                    continue

                distance = self.calcDist(bs) 

                if self.lineofsight[uuid][self.env.now] == 1:
                    reference = 61.4
                    exponent = 2.0
                    stdev = 5.8

                #LOS condition
                else:
                    reference = 72.0
                    exponent = 2.92
                    stdev = 8.7

                directionBSUE = uniform(0, 2*pi)
                angleBSUE = bs.calcAngle(self)
                directionUEBS = self.calcAngle(self.scenarioBasestations[self.servingBS])
                angleUEBS = self.calcAngle(bs)

                interference += 10**((bs.txPower + bs.antenna.gain(angle=angleBSUE, direction= directionBSUE) 
                        + self.antenna.gain(angle=angleUEBS, direction=directionUEBS) 
                        - self.channel.pathLossCalc(reference, exponent, 
                        distance, shadowingStdev=stdev))/10)

            noisePlusInterference = 10*log10(10**(self.channel.noisePower/10) + interference)
            SINR = self.listedRSRP[self.servingBS] - noisePlusInterference
            return SINR

    def updateBSList(self):
        for uuid, bs in self.scenarioBasestations.items():
            distance = self.calcDist(bs) 
             
            ### There are 2 reference signal at each OFDM symbol, and 4 OFDM
            ### symbols in a slot carrying reference signals
            referencesignals = bs.slotsPerSubframe*(bs.resourceBlocks*2)*4
             
            #NLOS condition
            if self.lineofsight[uuid][self.env.now] == 1:
                reference = 61.4
                exponent = 2.0
                stdev = 5.8

            #LOS condition
            else:
                reference = 72.0
                exponent = 2.92
                stdev = 8.7

            directionBSUE = bs.calcAngle(self)
            angleBSUE = directionBSUE
            directionUEBS = self.calcAngle(bs)
            angleUEBS = directionUEBS

            RSRP = (bs.txPower + bs.antenna.gain(angle=angleBSUE, direction= directionBSUE) 
                    + self.antenna.gain(angle=angleUEBS, direction=directionUEBS) 
                    - self.channel.pathLossCalc(reference, exponent, 
                    distance, shadowingStdev=stdev))
             
            if RSRP > self.sensibility:
                self.listedRSRP[uuid] = RSRP
            else:
                self.listedRSRP[uuid] = float('-inf') #None



class MobileUser(MeasurementDevice):
    def __init__ (self, scenario, inDict, Vx=0, Vy=0):
        super(MobileUser, self).__init__(scenario, inDict)
        self.mobilityModel = None
        self.packetArrivals = None
        self.snrThreshold = 10
        self.delay = 2

        self.Vx = Vx
        self.Vy = Vy

    def initializeServices(self, **params):
        self.env.process(self.firstAssociation())
        self.env.process(self.measurementEvent())
        self.env.process(self.sendingPackets())

        doppler = hypot(self.Vx, self.Vy)/self.env.wavelength
        self.channel.generateRayleighFading(doppler,self.env.simTime)

        self.channel.switchShadowing = True
        self.channel.switchFading = True

        if issubclass(self.mobilityModel.__class__, MobilityModel):
            self.env.process(self.mobilityModel.move(self))

    # This method schedules the packets transmission
    def sendingPackets(self):
        if self.packetArrivals:
            self.kpi.nPackets = len(self.packetArrivals)

        for t in self.packetArrivals:
            yield self.env.timeout(t - self.env.now)

            # The packet will be sent if and only if the UE is in sync with the
            # serving BS and, of course, if it is associated whatsoever
            if self.sync and self.servingBS != None:
                #snr = self.listedRSRP[self.servingBS] - self.channel['noisePower']
                snr = max(0, self.servingBSSINR())
                temp = self.snrThreshold
                timer = 0

                # There is a 10 milliseconds tolerance to send a packet
                while timer < 10 and self.servingBS != None:
                    if snr > self.snrThreshold or snr > temp:
                        self.kpi.deliveryRate += 1
                        rate = self.scenarioBasestations[self.servingBS].bandwidth*log2(1 + snr)
                        self.kpi.throughput.append(rate)
                        self.kpi.delay.append(self.env.now - t)
                        if self.kpi.delay[-1] < self.delay:
                            self.kpi.partDelay += 1

                        break
                    else:
                        temp *= 0.9
                        yield self.env.timeout(1)
                        timer += 1

class PredictionBaseStation(BaseStation):
    pass


class KPI:
    def __init__(self):
        self.partDelay = 0
        self.handover = 0
        self.handoverFail = 0
        self.pingpong = 0
        self.reassociation = 0
        self.throughput = []
        self.capacity = []
        self.deliveryRate = 0
        self.delay = []
        self.association = []
        self.outofsync = 0
        self.nPackets = 0

    def _print(self):
        if self.throughput:
            meanThroughput = mean(self.throughput)
        else:
            meanThroughput = 0

        if self.capacity:
            meanCapacity = mean(self.capacity)
        else:
            meanCapacity = 0

        if self.delay:
            meanDelay = mean(self.delay)
        else:
            meanDelay = 0

        print('partDelay:', self.partDelay/self.nPackets, 
            '\n handover:',self.handover,
            '\n handoverFail:', self.handoverFail,
            '\n pingpong:',self.pingpong,
            '\n reassociation:', self.reassociation,
            '\n throughput:', meanThroughput,
            '\n capacity:', meanCapacity,
            '\n deliveryRate:',self.deliveryRate/self.nPackets,
            '\n delay:', meanDelay,
            '\n association:',self.association,
            '\n outofsync:',self.outofsync)
    
    def printAsDict(self):
        if self.throughput:
            meanThroughput = mean(self.throughput)
        else:
            meanThroughput = 0

        if self.capacity:
            meanCapacity = mean(self.capacity)
        else:
            meanCapacity = 0

        if self.delay:
            meanDelay = mean(self.delay)
        else:
            meanDelay = 0

        dictionary = {}
        
        dictionary['partDelay'] = self.partDelay/self.nPackets
        dictionary['handover'] = self.handover
        dictionary['handoverFail'] = self.handoverFail
        dictionary['pingpong'] = self.pingpong
        dictionary['reassociation'] = self.reassociation
        dictionary['throughput'] = meanThroughput
        dictionary['capacity'] = meanCapacity
        dictionary['deliveryRate'] = self.deliveryRate/self.nPackets
        dictionary['delay'] = meanDelay
        dictionary['association'] = self.association
        dictionary['outofsync'] = self.outofsync

        print(dumps(dictionary, indent = 4))


class NetworkParameters:
    def __init__(self):
        self.timeToTrigger = 640
        self.timeToMeasure = 40

        self.qualityOut =  -100 #dBm
        self.qualityIn = -90 #dBm
        self.N310 = 10
        self.T310 = 100
        self.N311 = 10
        self.T304 = 100

        self.handoverHysteresys = 0
        self.handoverOffset = 3
        self.handoverThreshold = -90

        self.frameDuration = 10
        self.subframeDuration = 1
        self.SSBurstPeriod = 20
        self.RACHPeriod = 40
        self.SSBurstDuration = 5

class Numerology:
    def __init__(self, subcarrierspacing, networkParameters):
        if subcarrierspacing == 120 or subcarrierspacing == 120e3:
            self.slotDuration = networkParameters.subframeDuration/8
            self.OFDMSymbolDuration = self.slotDuration/14 
            self.minResourceBlocks = 20
            self.maxResourceBlocks = 138
            self.SSBlocks = 64
            self.SSBurstSlots = networkParameters.SSBurstPeriod/self.slotDuration

            self.SSBlockMapping = ((([0 for i in range(8)]+[1 for i in range(16)])*2+
                                    [0 for i in range(8)])*4+
                                    [0 for i in range(14)]*4)*2+[0 for i in range(14)]*32
        self.SSBurstLength = 0


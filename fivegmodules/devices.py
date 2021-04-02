from numpy import hypot
from numpy import log10
from numpy import log2
from numpy import mean
from numpy.random import normal, uniform, choice
from numpy import pi
from numpy import exp
import operator

from json import dumps

from fivegmodules.handover import Handover
from fivegmodules.core import Scenario
from fivegmodules.core import Channel
from fivegmodules.core import WirelessDevice
from fivegmodules.mobility import MobilityModel
from fivegmodules.mobility import StraightRoute
from fivegmodules.plot import PlotRSRP
from fivegmodules.plot import PlotSINR
from fivegmodules.miscellaneous import DataPacket


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
        self.handoverDecisionDelay = 15 # milliseconds (time to process handover decision)
        self.admissionControlDelay = 20 #milliseconds
        self.X2processingDelay = 5
        self.RRCprocessingDelay = 5
        self.preambleDetectionDelay = 3
        self.uplinkAllocationDelay = 5


        # Downlink proportion in relation to the data frame duration
        self.channelProportion = 0.5
        self.onDownlink = False
        self.downlinkDuration = 0 
        self.onUplink = False
        self.uplinkDuration = 0 

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

        self.downlinkDuration = (self.networkParameters.SSBurstPeriod 
                - self.networkParameters.SSBurstDuration)*self.channelProportion
        self.uplinkDuration = (self.networkParameters.SSBurstPeriod   
                - self.networkParameters.SSBurstDuration)*(1 - self.channelProportion)


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
                    self.nextSSB = self.env.now + burstPeriod #(burstPeriod - burstDuration)
                else:
                    ### Yes, the next is a RACH
                    self.nextSSB = self.env.now + 2*burstPeriod #(2*burstPeriod - burstDuration)
                    #print('Next one is a RACH, next SSB in %d #%d' % (self.nextSSB, self.ssbIndex))

                yield self.env.timeout(burstDuration)

                ### Flagging that the SSB is finished
                self.onSSB = False
 
                yield self.env.timeout(burstPeriod - burstDuration)

                '''
                ### Divides the lasting frame time in Downlink and Uplink Channels
                self.onDownlink = True
                yield self.env.timeout(self.downlinkDuration)
                self.onDownlink = False
                self.onUplink = True
                yield self.env.timeout(self.uplinkDuration)
                self.onUplink = False
                '''

 
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
                self.nextRach = self.env.now + rachPeriod #(rachPeriod - rachDuration)
                yield self.env.timeout(rachDuration)
                #print('The rach opportunity  has finished at %d' % self.env.now)

                ### Flagging the end of a RACH opportunity
                self.onRach = False

                yield self.env.timeout(rachPeriod - rachDuration)

    def receivePacket(self, transmitter, packet):
        '''
        mcsReqSinr = 20
        diff = min(mcsReqSinr - transmitter.servingBSSINR(), 5)

        errorProb = 10**(min(diff - 5, 0))

        error = choice([0, 1], p=[errorProb, 1 - errorProb])
        '''

        #if not error: #
        if transmitter.servingBSSINR() > self.sensibility:
            #print('Packet Received', packet.packetId)
            transmitter.kpi.deliveryRate += 1

            delay = self.env.now - packet.arrival
            transmitter.kpi.delay.append(self.env.now - packet.arrival)

            wholeData = packet.payloadLen + self.networkParameters.uplinkOverhead
            SINR = 10**(transmitter.servingBSSINR()/10)
            capacity = self.bandwidth*log2(1 + SINR)
            transmissionTime = wholeData/capacity
            throughput = packet.payloadLen/transmissionTime
            transmitter.kpi.throughput.append(throughput)

            if delay < transmitter.delay:
                transmitter.kpi.partDelay += 1

        '''
        else:
            print('Packet Lost', packet.packetId)
        '''


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
            #if device.listedRSRP[device.servingBS] < device.networkParameters.qualityOut and self.qualityOutCounter == 0:
            if device.listedRSRP[device.servingBS] < device.networkParameters.qualityOut and not device.T310running:
                downcounter = device.networkParameters.T310
                while downcounter > 0:
                    print(device.env.now, device.listedRSRP[device.servingBS], downcounter)

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
                    device.kpi.association[-1].append(device.env.now)
                    try:
                        device.kpi.outofsync.append(device.env.now)
                    except:
                        device.kpi.outofsync = [device.env.now]
 
                    # Need to reassociate with the network
                    device.lastBS.append(device.servingBS)
                    device.servingBS = None
                    device.T310running = False
 
                self.qualityOutCounter = 0
                self.qualityInCounter = 0
 
        '''
        # The user is already out of sync!
        else:
            device.sync = False
            try:
                #device.kpi.outofsync += 1
                device.kpi.outofsync.append(device.env.now)
            except:
                device.kpi.outofsync = [device.env.now]
            #print(device.env.now)
        '''

class HandoverAssessment(SignalAssessmentPolicy):
    def signalAssessment(self, device):
        maxRSRP = max(device.listedRSRP.items(), key=operator.itemgetter(1))[0]
        #for n,i in enumerate(device.listedRSRP.keys()):
        #    device.plotRecvRSRP[n].append(device.listedRSRP[i])
 
        ### FIRST TIME USER ASSOCIATON
        if device.servingBS == None and device.listedRSRP[maxRSRP] > device.sensibility:
            if device.lastBS:
                #device.reassociation(maxRSRP)
                device.env.process(device.connectionReestablishment(maxRSRP))
            #else:
            #    device.env.process(device.firstAssociation(maxRSRP))
 
 
        elif device.servingBS != None and device.sync:
            snr = 10**((device.listedRSRP[device.servingBS] - device.channel[device.servingBS].noisePower)/10)
            #print(self.env.now, maxRSRP, self.listedRSRP[maxRSRP], self.servingBS, self.listedRSRP[self.servingBS], snr)
            rate = device.scenarioBasestations[device.servingBS].bandwidth*log2(1 + snr)
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
        self.reassociationFlag = False
        self.handover = None
        self.handoverCommandProcDelay = 15 #milliseconds
        self.freqReconfigDelay = 20 #milliseconds
        self.uplinkAllocationProcessingDelay = 5
        
        self.syncAssessment = SynchronizationAssessment()
        self.bsAssessment = HandoverAssessment()
        self.kpi = KPI() 
        self.kpi.simTime = self.env.simTime

        self.measOccurring = False
        self.T310running = False
        self.triggerTime = 0

        self.plotRSRP = PlotRSRP(self)
        self.plotSINR = PlotSINR(self)


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
                yield self.env.timeout(3*self.networkParameters.RRCMsgTransmissionDelay)

                self.sync = True
                self.servingBS = baseStation
                self.kpi.association.append([list(self.scenarioBasestations.keys()).index(self.servingBS),
                                                self.env.now])
                #print('Association complete!', self.env.now)
                if self.reassociationFlag:
                    self.reassociationFlag = False
        #print('After', self.sync, self.env.now)

    def connectionReestablishment(self, baseStation):
        if not self.reassociationFlag:
            #print('Before', self.sync, self.env.now)
            self.kpi.reassociation += 1
            self.reassociationFlag = True
            
            yield self.env.timeout(3*self.networkParameters.RRCMsgTransmissionDelay)

            yield self.env.process(self.firstAssociation(baseStation))

    def servingBSSINR(self):
        interference = 0
        servingBSPower = 0

        if self.servingBS == None:
            return 0

        else:
            for uuid, bs in self.scenarioBasestations.items():

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


                if uuid == self.servingBS:
                    directionBSUE = bs.calcAngle(self)
                    angleBSUE = directionBSUE
                    directionUEBS = self.calcAngle(bs)
                    angleUEBS = directionUEBS

                    servingBSPower = (bs.txPower + bs.antenna.gain(angle=angleBSUE, direction= directionBSUE)
                            + self.antenna.gain(angle=angleUEBS, direction=directionUEBS)
                            - self.channel[uuid].pathLossCalc(reference, exponent,
                            distance, shadowingStdev=stdev))

                else:
                    directionBSUE = uniform(0, 2*pi)
                    angleBSUE = bs.calcAngle(self)
                    directionUEBS = self.calcAngle(self.scenarioBasestations[self.servingBS])
                    angleUEBS = self.calcAngle(bs)

                    interference += 10**((bs.txPower + bs.antenna.gain(angle=angleBSUE, direction= directionBSUE) 
                            + self.antenna.gain(angle=angleUEBS, direction=directionUEBS) 
                            - self.channel[uuid].pathLossCalc(reference, exponent, 
                            distance, shadowingStdev=stdev))/10)


            noisePlusInterference = 10*log10(10**(self.channel[uuid].noisePower/10) + interference)

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
                    - self.channel[uuid].pathLossCalc(reference, exponent, 
                    distance, shadowingStdev=stdev, fadingSample = self.env.now))
             
            try:
                oldRSRP = self.listedRSRP[uuid]
                filterCoeff = self.networkParameters.filterA()
                self.listedRSRP[uuid] = (1-filterCoeff)*oldRSRP + filterCoeff*RSRP

            except KeyError:
                if RSRP > self.sensibility:
                    self.listedRSRP[uuid] = RSRP
                else:
                    self.listedRSRP[uuid] = float('-inf') #None

        try:
            value =  self.listedRSRP[self.servingBS]
        except KeyError:
            value = 0
        self.plotRSRP.collectKpi(value)
        self.plotSINR.collectKpi()
        self.kpi.averageSinr.append(self.servingBSSINR())



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
        self.env.process(self.connectivityGap())

        #doppler = hypot(self.Vx, self.Vy)/self.env.wavelength
        #self.channel.generateRayleighFading(doppler,self.env.simTime)

        #self.channel.switchShadowing = True
        #self.channel.switchFading = True

        if issubclass(self.mobilityModel.__class__, MobilityModel):
            self.env.process(self.mobilityModel.move(self))

    def connectivityGap(self):
        while True:
            if not self.sync:
                self.kpi.gap += 1
            yield self.env.timeout(1)

    # This method schedules the packets transmission
    def sendingPackets(self):
        if self.packetArrivals:
            self.kpi.nPackets = len(self.packetArrivals)

        for pktId, t in enumerate(self.packetArrivals):
            packetSent = False
            yield self.env.timeout(t - self.env.now)

            # Generating the packet to be transmitted
            packetLen = 24 * 960 * 720 #* normal(loc=0, scale=1024) 


            try:
                packet = DataPacket(self, self.scenarioBasestations[self.servingBS], 
                                pktId, self.env.now, packetLen)

                if (self.scenarioBasestations[self.servingBS].onSSB or 
                    self.scenarioBasestations[self.servingBS].onRach):

                    if (self.scenarioBasestations[self.servingBS].nextSSB <
                        self.scenarioBasestations[self.servingBS].nextRach):
                        yield self.env.timeout(self.scenarioBasestations[self.servingBS].nextSSB
                                                - (self.networkParameters.SSBurstPeriod
                                                - self.networkParameters.SSBurstDuration)
                                                - self.env.now)

                    elif (self.scenarioBasestations[self.servingBS].nextSSB >
                        self.scenarioBasestations[self.servingBS].nextRach):
                        yield self.env.timeout(self.scenarioBasestations[self.servingBS].nextRach
                                                - (self.networkParameters.SSBurstPeriod
                                                - self.networkParameters.SSBurstDuration)
                                                - self.env.now)

                self.scenarioBasestations[self.servingBS].receivePacket(self, packet)

            except KeyError:
                continue

            '''
            # The packet will be sent if and only if the UE is in sync with the
            # serving BS and, of course, if it is associated whatsoever
            if self.sync or self.handover.handoverExecutionFlag: #and self.servingBS != None:
                #snr = self.listedRSRP[self.servingBS] - self.channel['noisePower']
                snr = max(0, self.servingBSSINR())
                temp = self.snrThreshold
                timer = 0

                self.scenarioBasestations[self.servingBS].receivePacket(self, packet)

                # There is a 10 milliseconds tolerance to send a packet
                while timer < self.networkParameters.retryTimer and self.servingBS != None:
                    if snr > self.snrThreshold or snr > temp:
                        self.kpi.deliveryRate += 1
                        rate = self.scenarioBasestations[self.servingBS].bandwidth*log2(1 + snr)
                        self.kpi.throughput.append(rate)
                        self.kpi.delay.append(self.env.now - t)
                        if self.kpi.delay[-1] < self.delay:
                            self.kpi.partDelay += 1

                        packetSent = True

                        break
                    else:
                        temp *= 0.9
                        yield self.env.timeout(1)
                        timer += 1

                if not packetSent:
                    self.kpi.delay.append(self.env.now - t)
            '''

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
        self.log = {}
        self.averageSinr = []
        self.averageBlockageDuration = []
        self.gap = 0
        self.simTime = 0

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
            '\n outofsync:',self.outofsync,
            '\n Log:',self.log)
    
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

        if self.averageSinr:
            meanSinr = mean(self.averageSinr)
        else:
            meanSinr = 0

        if self.averageBlockageDuration:
            meanBlockageDuration = mean(self.averageBlockageDuration)
        else:
            meanBlockageDuration = 0

        self.association[-1].append(self.simTime)

        dictionary = {}
        
        dictionary['sinr'] = meanSinr
        dictionary['gap'] = self.gap
        dictionary['partDelay'] = self.partDelay/self.nPackets
        dictionary['handover'] = self.handover
        dictionary['handoverRate'] = self.handover/(self.simTime/1e3)
        dictionary['handoverFail'] = self.handoverFail/self.handover
        dictionary['pingpong'] = self.pingpong/self.handover
        dictionary['reassociation'] = self.reassociation
        dictionary['throughput'] = meanThroughput
        dictionary['capacity'] = meanCapacity
        dictionary['deliveryRate'] = self.deliveryRate/self.nPackets
        dictionary['delay'] = meanDelay
        dictionary['association'] = self.association
        dictionary['outofsync'] = self.outofsync
        dictionary['log'] = self.log

        print(dumps(dictionary, indent = 4))


class NetworkParameters:
    def __init__(self):
        self.timeToTrigger = 640
        self.timeToMeasure = 40

        self.RRCMsgTransmissionDelay = 5 #milliseconds

        self.downlinkOverhead = 240 + 800 + 28800 +1000
        self.uplinkOverhead = 240 + 800 + 28800 +1000

        self.qualityOut =  -100 #dBm
        self.qualityIn = -90 #dBm
        self.N310 = 1
        self.T310 = 1000
        self.N311 = 1
        self.T304 = 100

        self.handoverHysteresys = 0
        self.handoverOffset = 3
        self.handoverThreshold = -90

        self.frameDuration = 10
        self.subframeDuration = 1
        self.SSBurstPeriod = 20
        self.RACHPeriod = 40
        self.SSBurstDuration = 5

        self.retryTimer = 10

        self.filterK = 4

    def filterA(self):
        return (1/2**(self.filterK/4))

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


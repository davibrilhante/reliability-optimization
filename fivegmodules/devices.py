from numpy import hypot
from numpy import log10
from numpy import log2
from numpy import mean
from numpy.random import normal, uniform, choice
from numpy import pi
from numpy import exp
from numpy import zeros
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
        self.MCSthreshold = 15.5595597 #dB

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
        if transmitter.servingBSSINR() > transmitter.snrThreshold: #self.MCSthreshold: #self.sensibility:
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
            return True

        else:
            return False

        '''
        else:
            print('Packet Lost', packet.packetId)
        '''


class SignalAssessmentPolicy:
    def signalAssessment(self, device):
        raise NotImplementedError


class T310Monitor(SignalAssessmentPolicy):
    def __init__ (self):
        self.qualityOutCounter = 0
        self.qualityInCounter = 0
        self.T310flag = False
        self.T310ElapsedTime = 0
        self.SINRSample = 0
                           
    # Testing whether the handover did not fail due to be late
    # The UE sync with the serving BS will be tested
    def signalAssessment(self, device):
        device.env.process(self.qualityInMonitor(device))
        device.env.process(self.qualityOutMonitor(device))

    def qualityOutMonitor(self, device):
        while True:
            # every 200 ms will check if SINR < Q_in
            yield device.env.timeout(device.networkParameters.qOutMonitorTimer)

            if not self.T310flag:

                if self.SINRSample < device.networkParameters.qualityOut:
                    # if in Q_out condition, it checks if T310 counter is running
                    self.qualityOutCounter += 1

                    # If it is not running, check if there are N310 Q_out samples
                    # already sampled to see whether it should start T310 or not
                    if self.qualityOutCounter >= device.networkParameters.N310:
                        self.T310flag = True
                        device.T310running = True
            else:

                # Yes, there is a T310 timer on! We should increment it by
                # the time elapsed since the last Q_out check
                self.T310ElapsedTime += device.networkParameters.qOutMonitorTimer

                if self.T310ElapsedTime >= device.networkParameters.T310:
                    # T310 is over, the user is out of sync!
                    device.sync = False
                    device.T310running = False
                    device.lastBS.append(device.servingBS)
                    device.servingBS = None


                    self.T310flag = False
                    self.T310ElapsedTime = 0

                    device.kpi.association[-1].append(device.env.now)
                    try:
                        device.kpi.outofsync.append([device.env.now])
                    except:
                        device.kpi.outofsync = [[device.env.now]]

                else:
                    self.qualityOutCounter = 0
                    self.qualityInCounter = 0



    def qualityInMonitor(self, device):
        while True:
            # every 100 ms will check if SINR >= Q_in
            yield device.env.timeout(device.networkParameters.qInMonitorTimer)

            # When no time passess in the simulation we must use the same SINR sample
            # because the method called for SINR measurement changes every time it is
            # called due to its radomness
            self.SINRSample = device.servingBSSINR()
            #print('T310 flag:', self.T310flag, ' BS SINR: ', self.SINRSample)

            if self.SINRSample >= device.networkParameters.qualityIn:
                #print(self.SINRSample)

                # If in Q_in condition, it checks if the T310 is running
                if not self.T310flag:
                    # If T310 not running, just reset the counters
                    self.qualityOutCounter = 0
                    self.qualityInCounter = 0

                else:

                    # If T310 is counting, then t increments the Q_in counter 
                    # and checks if there are N311 Q_in counters
                    self.qualityInCounter+= 1

                    if self.qualityInCounter >= device.networkParameters.N311:
                        # There are N311 Q_in counters, then it stops T310

                        self.T310flag = False
                        device.T310running = False
                        self.T310ElapsedTime = 0

            
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
        '''
        try:
            print(device.env.now, maxRSRP, device.listedRSRP[maxRSRP],device.servingBS, device.listedRSRP[device.servingBS])
        except KeyError:
            pass
        '''
 
        if device.servingBS == None and device.listedRSRP[maxRSRP] > device.sensibility:
            if device.lastBS:
                #device.reassociation(maxRSRP)
                #device.env.process(device.connectionReestablishment(maxRSRP))
                device.kpi.reassociation += 1
                device.reassociationFlag = True

                device.env.process(device.cellAttachmentProcedure())

            elif device.ignoreFirstAssociation:
                #device.env.process(device.firstAssociation(maxRSRP))
                device.sync = True
                device.servingBS = maxRSRP
                device.kpi.association.append([list(device.scenarioBasestations.keys()).index(device.servingBS),
                                                device.env.now])
 
 
        elif device.servingBS != None and device.sync:
            #snr = 10**((device.listedRSRP[device.servingBS] - device.channel[device.servingBS].noisePower)/10)
            sinr = 10**((device.servingBSSINR()/10))
            #print(device.env.now, maxRSRP, device.listedRSRP[maxRSRP],device.servingBS, device.listedRSRP[device.servingBS], sinr)
            rate = device.scenarioBasestations[device.servingBS].bandwidth*log2(1 + sinr)
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
        
        #self.syncAssessment = SynchronizationAssessment()
        self.syncAssessment = T310Monitor()
        self.bsAssessment = HandoverAssessment()
        self.kpi = KPI() 
        self.kpi.simTime = self.env.simTime

        self.measOccurring = False
        self.T310running = False
        self.triggerTime = 0

        self.cellAttachmentFlag = False

        self.plotRSRP = PlotRSRP(self)
        self.plotSINR = PlotSINR(self)
        self.switchInterference = True
        self.ignoreFirstAssociation = False


    # At each time to measure, the UE updates the list of BS and check
    # if the handover event is happening
    def measurementEvent(self):
        while True:
            self.updateBSList()
            self.bsAssessment.signalAssessment(self)
            yield self.env.timeout(self.networkParameters.timeToMeasure)

    # This method processes the LOS info from the instance
    def addLosInfo(self, los : list, n : int):
        for m,bs in enumerate(self.scenarioBasestations):
            self.lineofsight[bs] = los[m][n]

    def cellAttachmentProcedure(self):
        if not self.cellAttachmentFlag:

            self.cellAttachmentFlag = True
            counterDRX = 0

            # UE runs some DRX cycle to find a suitable BS
            while counterDRX < self.networkParameters.numberDRX:
                yield self.env.timeout(self.networkParameters.DRXlength)
                self.updateBSList()
                baseStation = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]

                if self.listedRSRP[baseStation] >= self.networkParameters.cellSelectionRxLevel:
                    break

            # Once a good BS is found, then starts the uplink synch
            yield self.env.timeout(
                    self.scenarioBasestations[baseStation].nextRach 
                    + self.networkParameters.SSBurstDuration  - self.env.now
                    )

            # Send the cell attach/connection reestablishment request
            if self.listedRSRP[baseStation] >= self.networkParameters.cellSelectionRxLevel:
                yield self.env.timeout(self.networkParameters.RRCMsgTransmissionDelay
                                    + self.scenarioBasestations[baseStation].RRCprocessingDelay)

                if baseStation not in self.lastBS:
                    yield self.env.timeout(self.networkParameters.networkProcessingDelay)

                # Receive a RRC Connection Reconfiguration
                yield self.env.timeout(self.networkParameters.RRCMsgTransmissionDelay
                                        + self.handoverCommandProcDelay
                                        + self.freqReconfigDelay)

                # Send a RRC Reconfiguration Complete
                yield self.env.timeout(self.networkParameters.RRCMsgTransmissionDelay
                                    + self.scenarioBasestations[baseStation].RRCprocessingDelay)

                if self.listedRSRP[baseStation] >= self.networkParameters.cellSelectionRxLevel:
                    self.sync = True
                    self.servingBS = baseStation
                    self.kpi.association.append([list(self.scenarioBasestations.keys()).index(self.servingBS),
                                                    self.env.now])

                    #print('Association complete!', self.env.now)
                    if self.reassociationFlag:
                        self.kpi.outofsync[-1].append(self.env.now)
                        self.reassociationFlag = False


            self.cellAttachmentFlag = False






    def firstAssociation(self, baseStation = None):
        if not self.cellAttachmentFlag:
            self.cellAttachmentFlag = True
            #print(self.env.now, 'cell attachment routine')

            counterDRX = 0 
            while counterDRX < self.networkParameters.numberDRX:
                yield self.env.timeout(self.networkParameters.DRXlength)
                self.updateBSList()
                baseStation = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]
                try:
                    if baseStation == self.lastBS[-1]:
                        break
                except:
                    pass
                counterDRX += 1


            ### FIRST TIME USER ASSOCIATON
            if baseStation == None:
                baseStation = max(self.listedRSRP.items(), key=operator.itemgetter(1))[0]

            #print(self.env.now, self.scenarioBasestations[baseStation].nextSSB)
            #print(self.env.now, self.listedRSRP)

            yield self.env.timeout(             
                        self.scenarioBasestations[baseStation].nextSSB + 
                        self.networkParameters.SSBurstDuration  - self.env.now
                        )

            #if self.listedRSRP[baseStation] > self.networkParameters.qualityOut: 
            if self.listedRSRP[baseStation] > self.sensibility:
                # yields untill the downlink and uplink sync is completed
                yield self.env.timeout(
                        self.scenarioBasestations[baseStation].nextRach 
                        + self.networkParameters.SSBurstDuration  - self.env.now
                        )
                
                #if self.listedRSRP[baseStation] > self.networkParameters.qualityOut: 
                if self.listedRSRP[baseStation] > self.sensibility:
                    # Now, the UE is in sync with the serving BS
                    yield self.env.timeout(3*self.networkParameters.RRCMsgTransmissionDelay
                                + self.handoverCommandProcDelay
                                + 2*self.scenarioBasestations[baseStation].RRCprocessingDelay)

                    #print('picked ', baseStation, ' for new cell at', self.env.now)

                    self.sync = True
                    self.servingBS = baseStation
                    self.kpi.association.append([list(self.scenarioBasestations.keys()).index(self.servingBS),
                                                    self.env.now])
                    #print('Association complete!', self.env.now)
                    if self.reassociationFlag:
                        self.kpi.outofsync[-1].append(self.env.now)
                        self.reassociationFlag = False
            self.cellAttachmentFlag = False
            #print('After', self.sync, self.env.now)


    def connectionReestablishment(self, baseStation):
        if not self.reassociationFlag:
            #print('Before', self.sync, self.env.now)
            self.kpi.reassociation += 1
            self.reassociationFlag = True
            
            yield self.env.timeout(3*self.networkParameters.RRCMsgTransmissionDelay
                        + self.handoverCommandProcDelay                
                        + 2*self.scenarioBasestations[baseStation].RRCprocessingDelay)

            yield self.env.process(self.firstAssociation(baseStation))

    def servingBSSINR(self):
        interference = 0
        servingBSPower = 0

        if self.servingBS == None:
            return 0

        else:
            for uuid, bs in self.scenarioBasestations.items():

                distance = self.calcDist(bs) 

                #LOS condition
                if self.lineofsight[uuid][self.env.now] == 1:
                    reference = 61.4
                    exponent = 2.0
                    stdev = 5.8

                #NLOS condition
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
                            distance, shadowingStdev=stdev, fadingSample = self.env.now))

                else:
                    reference = 72.0
                    exponent = 2.92
                    stdev = 8.7

                    directionBSUE =  bs.calcAngle(self) + pi #uniform(0, 2*pi)
                    angleBSUE = bs.calcAngle(self)
                    if directionBSUE > 2*pi:
                        directionBSUE -= 2*pi
                    directionUEBS = self.calcAngle(self.scenarioBasestations[self.servingBS])
                    angleUEBS = self.calcAngle(bs)

                    interference += 10**((bs.txPower + bs.antenna.gain(angle=angleBSUE, direction= directionBSUE) 
                            + self.antenna.gain(angle=angleUEBS, direction=directionUEBS) 
                            - self.channel[uuid].pathLossCalc(reference, exponent, 
                            distance, shadowingStdev=stdev, fadingSample = self.env.now))/10)


            # Toggles the neighbour BS interference
            if self.switchInterference:
                
                # Noise plus Interference converted to linear
                noisePlusInterference = 10**(self.channel[self.servingBS].noisePower/10) + interference

                #print(self.env.now, 10**(self.listedRSRP[self.servingBS]/10), noisePlusInterference)

                # SINR in dB
                SINR = 10*log10((10**(self.listedRSRP[self.servingBS]/10))/noisePlusInterference)

                #SINR = servingBSPower - noisePlusInterference
                #print('bssinr', servingBSPower)
                #print('bssinr', self.listedRSRP[self.servingBS],'\n n+i', noisePlusInterference)
                #print(SINR)

                return SINR

            else:
                # Noise plus Interference converted to linear
                noisePower = 10**(self.channel[self.servingBS].noisePower/10)

                # SNR in dB
                SNR = 10*log10((10**(self.listedRSRP[self.servingBS]/10))/noisePower)

                #print(SNR)
                return SNR


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
                self.listedRSRP[uuid] = RSRP
                '''
                if RSRP > self.sensibility:
                    self.listedRSRP[uuid] = RSRP
                else:
                    self.listedRSRP[uuid] = float('-inf') #None
                '''

        try:
            value =  self.listedRSRP[self.servingBS]
        except KeyError:
            value = -200 ### Better than -inf???
 
        self.plotRSRP.collectKpi(value)
        #print('update', value)
        self.plotSINR.collectKpi()
        #self.kpi.averageSinr.append(self.servingBSSINR())


class PacketGenerator:
    def __init__(self):
        pass
    def generate(self):
        pass

class MobileUser(MeasurementDevice):
    def __init__  (self, scenario, inDict, Vx=0, Vy=0):
        super(MobileUser, self).__init__(scenario, inDict)
        self.mobilityModel = None
        self.packetArrivals = None
        self.snrThreshold = 10
        self.delay = 2

        self.Vx = Vx
        self.Vy = Vy

        self.max_pkt_retry = 10
        self.packet_generator = None
        self.pred_offset = 0 
        self.pred_probs = zeros(len(self.scenarioBasestations))

    def capacity2snr(self):
        return 10*log10(2**(
            self.capacity/self.scenarioBasestations[self.servingBS].bandwidth)-1)

    def initializeServices(self, **params):
        #self.env.process(self.firstAssociation())
        #self.env.process(self.cellAttachmentProcedure())
        self.env.process(self.measurementEvent())
        self.env.process(self.sendingPackets())
        self.env.process(self.connectivityGap())

        self.syncAssessment.signalAssessment(self)

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
        snr_capacity = self.capacity2snr()
        self.snrThreshold = max(snr_capacity, self.snrThreshold)

        if self.packetArrivals:
            nPackets = len(self.packetArrivals)
            self.kpi.nPackets = nPackets 

        for pktId, t in enumerate(self.packetArrivals):
            retries = 0
            packet_sent = False
            yield self.env.timeout(t - self.env.now)

            # Generating the packet to be transmitted
            packetLen = 24 * 960 * 720 #* normal(loc=0, scale=1024) 

            if pktId < nPackets - 1:
                pkt_time_limit = self.packetArrivals[pktId+1]
            else:
                pkt_time_limit = self.env.simTime - 1

            while retries <= self.max_pkt_retry and t < pkt_time_limit:
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

                    packet_sent = self.scenarioBasestations[self.servingBS].receivePacket(self, packet)

                except KeyError:
                    continue

                if packet_sent:
                    if self.env.now - t > 0:
                        self.kpi.delay.append(self.env.now - t)
                    break

                else:
                    retries += 1
                    yield self.env.timeout(1)

            if not packet_sent:
                self.kpi.delay.append(self.env.now - t)




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
        self.averageRsrp = []
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

        if self.averageRsrp:
            meanRsrp = mean(self.averageRsrp)
        else:
            meanRsrp = 0

        if self.averageBlockageDuration:
            meanBlockageDuration = mean(self.averageBlockageDuration)
        else:
            meanBlockageDuration = 0

        self.association[-1].append(self.simTime)

        dictionary = {}
        
        dictionary['sinr'] = meanSinr
        dictionary['rsrp'] = meanRsrp
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

        self.RRCMsgTransmissionDelay = 5 # milliseconds
        self.networkProcessingDelay = 20 # milliseconds
        self.cellSelectionRxLevel = -65 # dBm

        self.downlinkOverhead = 240 + 800 + 28800 +1000
        self.uplinkOverhead = 240 + 800 + 28800 +1000

        self.qualityOut = -7.2 # dB SINR #-100 # dBm RSRP
        self.qualityIn = -4.8  # dB SINR #-90 # dBm RSRP
        self.N310 = 1
        self.qOutMonitorTimer = 200
        self.T310 = 1000
        self.N311 = 1
        self.qInMonitorTimer = 100
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

        self.DRXlength = 320
        self.numberDRX = 4

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


import fivegmodules.core

__all__ = ['Handover', 'A3Handover', 'PredictionHandover']

class Handover:
    def __init__(self):
        pass

    def triggerMeasurement(self, device, targetBS):
        raise NotImplementedError
 
    def sendMeasurementReport(self, device, targetBS):
        raise NotImplementedError
 
    def handoverFailure(self, device):
        raise NotImplementedError
 
    def switchBaseStation(self, device, targetBS):
        raise NotImplementedError

class A3Handover(Handover):
    def __init__(self):
        super(A3Handover, self).__init__()
        self.handoverExecutionFlag = False
        self.handoverPreparationFlag = False
        self.x2Delay = 5 #milliseconds
        self.handoverCount = 0
        self.handoverPreparationStart = 0
        self.handoverFlag = False
 
    def triggerMeasurement(self, device, targetBS):
        counterTTT = 0
                   
        if not device.sync:
            self.handoverFailure(device)
                   
        # First, check if another measurement is not in progress
        elif not device.measOccurring and not self.handoverFlag:
            # If it is not, check whether it is an A3 event or not
            if  (
                    (
                        device.listedRSRP[targetBS] 
                        - device.networkParameters.handoverHysteresys
                        )
                >= (
                    device.listedRSRP[device.servingBS] 
                    + device.networkParameters.handoverOffset
                    )
                ):

                # Given that it is an A3 event, triggers the measurement
                device.measOccurring = True
                device.triggerTime = device.env.now
                self.handoverCount += 1

                if device.lineofsight[device.servingBS][device.env.now] == 0:
                    # Handover due to non line of sight condition
                    handoverCause = 0

                elif (device.calcDist(device.scenarioBasestations[targetBS]) 
                        < device.calcDist(device.scenarioBasestations[device.servingBS])):
                    # Handover due to user mobility, i.e., closer to target BS than to Serving BS
                    handoverCause = 1

                elif device.listedRSRP[targetBS] > device.listedRSRP[device.servingBS]:
                    # Handover due to severe channel fluctuations
                    handoverCause = 2
                 
                while counterTTT < device.networkParameters.timeToTrigger:
                    yield device.env.timeout(1) #device.networkParameters.timeToMeasure)
                    counterTTT += 1 #device.networkParameters.timeToMeasure
 
                    if device.sync:
                        # The A3 condition still valid? If not, stop the timer
                        if (
                                (
                                    device.listedRSRP[targetBS] 
                                    - device.networkParameters.handoverHysteresys
                                    )
                            <= (
                                device.listedRSRP[device.servingBS] 
                                + device.networkParameters.handoverOffset
                                )
                            ):

                            # Too earlier handover attempt, might have been just
                            # a channel fluctuation
                            try:
                                device.kpi.log['HOF000'] += 1
                            except KeyError:
                                device.kpi.log['HOF000'] = 1

                            break
                    else:
                        # Too late handover, user got out-sync in the middle of it

                        try:
                            device.kpi.log['HOF001'] += 1
                        except KeyError:
                            device.kpi.log['HOF001'] = 1

                        device.kpi.handover +=1
                        self.handoverFailure(device)
                        break

                if counterTTT == device.networkParameters.timeToTrigger:
                    self.handoverFlag = True
                    device.kpi.handover +=1
 
                    if device.sync:
                        if handoverCause == 0:
                            # Handover due to non line of sight condition
                            try:
                                device.kpi.log['HOC000'] += 1
                            except KeyError:
                                device.kpi.log['HOC000'] = 1

                        elif handoverCause == 1:
                            # Handover due to user mobility, i.e., closer to target BS than to Serving BS
                            try:
                                device.kpi.log['HOC001'] += 1
                            except KeyError:
                                device.kpi.log['HOC001'] = 1

                        elif handoverCause == 2:
                            # Handover due to severe channel fluctuations
                            try:
                                device.kpi.log['HOC002'] += 1
                            except KeyError:
                                device.kpi.log['HOC002'] = 1

                        else:
                            # Non specified reason to handover trigger
                            try:
                                device.kpi.log['HOC099'] += 1
                            except KeyError:
                                device.kpi.log['HOC099'] = 1


                        device.env.process(self.sendMeasurementReport(device, targetBS))
     
                    else:
                        try:
                            device.kpi.log['HOF002'] += 1
                        except KeyError:
                            device.kpi.log['HOF002'] = 1
                        self.handoverFailure(device)

                device.measOccurring = False
                device.triggerTime = 0
 
 
 
    def sendMeasurementReport(self, device, targetBS):

        #Check if it is not a reassociation
        if device.listedRSRP[device.servingBS] != None:

            # Holds the time to send the RRC:Measurement Report
            yield device.env.timeout(device.networkParameters.RRCMsgTransmissionDelay)

            # Check if it is a pingpong, just for kpi assessment
            if device.lastBS.count(targetBS)>0:
                device.kpi.pingpong += 1

            #Base stations processing the handover at X2 interface
            self.handoverPreparationFlag = True
            self.handoverPreparationStart = device.env.now
            yield device.env.timeout(
                device.scenarioBasestations[device.servingBS].RRCprocessingDelay
                + device.scenarioBasestations[device.servingBS].handoverDecisionDelay
                + 2*self.x2Delay
                + 2*device.scenarioBasestations[targetBS].X2processingDelay
                + device.scenarioBasestations[targetBS].admissionControlDelay)

            # Switch to the new BS
            device.env.process(self.switchBaseStation(device, targetBS))

 
 
    def handoverFailure(self, device):
        device.lastBS.append(device.servingBS)
        device.servingBS = None
        device.reassociationFlag = False
        device.kpi.handoverFail += 1
 
        device.kpi.association[-1].append(device.env.now)
        self.handoverFlag = False
        device.sync = False

    def switchBaseStation(self, device, targetBS):
        if device.sync:
            device.lastBS.append(device.servingBS)

            # yields for receiving HO Command RRC message and process this message
            yield device.env.timeout(device.networkParameters.RRCMsgTransmissionDelay
                                    + device.handoverCommandProcDelay)

            self.handoverPreparationFlag = False

            # Calculates Handover Preparation time
            try:
                device.kpi.log['HOP'].append(device.env.now 
                                            - self.handoverPreparationStart)
            except KeyError:
                device.kpi.log['HOP'] = [device.env.now 
                                        - self.handoverPreparationStart]

            '''
            # yields untill the downlink sync is completed
            yield device.env.timeout(
                    device.scenarioBasestations[targetBS].nextSSB + 
                    device.networkParameters.SSBurstDuration  - device.env.now
                    )
            #'''

            # Receiving handover command and turning to RRC Idle untill uplink 
            # sync with target BS
            if device.sync and not device.T310running:
                disassociation = device.env.now
                device.sync = False

                self.handoverExecutionFlag = True

                yield device.env.timeout(device.freqReconfigDelay)

                '''
                The time gap between the disassociation from the Serving BS to the
                target BS is known as handover interruption time (HIT) and it is
                the for the UE to get synced with the target BS. There is no data
                connection during this time interval, so the UE remains unsynced
                '''

                # yields untill the uplink sync is completed and a RACH preamble is sent
                yield device.env.timeout(
                        device.scenarioBasestations[targetBS].nextRach + 
                        device.networkParameters.SSBurstDuration  - device.env.now
                        )

                # Wait for receiving uplink Uplink Grant
                yield device.env.timeout(
                    device.scenarioBasestations[targetBS].preambleDetectionDelay
                    + device.scenarioBasestations[targetBS].uplinkAllocationDelay
                    + device.networkParameters.RRCMsgTransmissionDelay
                    + device.uplinkAllocationProcessingDelay
                    )

                # yields to send RRC Reconfiguration complete message
                yield device.env.timeout(device.networkParameters.RRCMsgTransmissionDelay)

                # Checks whether the HO Complete will be successfully sent/received 
                # If not, the handover fails
                if device.listedRSRP[targetBS] > device.networkParameters.qualityOut: 

                    # Once the RRC:HO complete is recieved it needs to be processed
                    yield device.env.timeout(device.scenarioBasestations[targetBS].RRCprocessingDelay)

                    # Now the UE is up and downlink synced
                    device.sync = True
                    self.handoverFlag = False
                    device.servingBS = targetBS
                    device.kpi.association.append(
                        [list(device.scenarioBasestations.keys()).index(device.servingBS), device.env.now])
                    device.kpi.association[-2].append(disassociation)

                    try:
                        device.kpi.log['HIT'].append(device.env.now - disassociation)
                    except KeyError:
                        device.kpi.log['HIT'] = [device.env.now - disassociation]


                else:
                    # Handover fail due to not received handover complete
                    try:
                        device.kpi.log['HOF004'] += 1
                    except KeyError:
                        device.kpi.log['HOF004'] = 1

                    self.handoverFailure(device)
                self.handoverExecutionFlag = False

            # Failed to receive Handover Command
            else:
                try:
                    device.kpi.log['HOF003'] += 1
                except KeyError:
                    device.kpi.log['HOF003'] = 1

                self.handoverFailure(device)

            self.handoverPreparationFlag = False



class PredictionHandover(Handover):
    pass

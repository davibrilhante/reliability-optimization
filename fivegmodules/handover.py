import fivegmodules.core
import numpy as np
from decimal import Decimal

__all__ = ['Handover', 'A3Handover', 'HeuristicHandover']

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

        ### To solve composition problems
        self.parent = self
        #self.handoverDecisionFunction = 

    def handovercause(self, device, targetBS):
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

        return handoverCause

    def loghandovercause(self, device, handoverCause):
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

    def loghandoverfailure(self, device, failurecode, n):
        try:
            device.kpi.log[failurecode] += n
        except KeyError:
            device.kpi.log[failurecode] = n

    def triggerMeasurement(self, device, targetBS):
        counterTTT = 0
        #print(device.env.now, device.sync, device.measOccurring, self.handoverFlag) 
                   
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
                        + device.networkParameters.handoverHysteresys
                    )
                ):

                # Given that it is an A3 event, triggers the measurement
                device.measOccurring = True
                device.triggerTime = device.env.now
                self.parent.handoverCount += 1

                handoverCause = self.handovercause(device,targetBS)
                 
                while counterTTT < device.networkParameters.timeToTrigger:
                    yield device.env.timeout(device.networkParameters.timeToMeasure)
                    counterTTT += device.networkParameters.timeToMeasure
 
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
                                    + device.networkParameters.handoverHysteresys
                                )
                            ):

                            # Too earlier handover attempt, might have been just
                            # a channel fluctuation
                            self.loghandoverfailure(device,'HOF000',1)
                            break
                    else:
                        # Too late handover, user got out-sync in the middle of it
                        self.loghandoverfailure(device,'HOF001',1)

                        device.kpi.handover +=1
                        self.handoverFailure(device)
                        break

                if counterTTT == device.networkParameters.timeToTrigger:
                    self.handoverFlag = True
                    device.kpi.handover +=1
 
                    if device.sync:
                        self.loghandovercause(device, handoverCause)
                        device.env.process(self.parent.sendMeasurementReport(device, targetBS))
     
                    # Too late handover, user got out-sync in the middle of it
                    # and will not be able to communicate with Serving BS to complete
                    else:
                        self.loghandoverfailure(device,'HOF002',1)
                        self.handoverFailure(device)

                device.measOccurring = False
                device.triggerTime = 0
 
 
 
    def sendMeasurementReport(self, device, targetBS):

        #Check if it is not a reassociation
        #if device.listedRSRP[device.servingBS] != None:
        if device.sync:# and (device.servingBSSINR() > device.networkParameters.qualityOut):

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
        '''
        else:
            try:
                device.kpi.log['HOF002'] += 1
            except KeyError:
                device.kpi.log['HOF002'] = 1
            self.handoverFailure(device)
        '''
 
    def handoverFailure(self, device):
        device.lastBS.append(device.servingBS)
        device.servingBS = None
        device.reassociationFlag = False
        device.kpi.handoverFail += 1
 
        #device.kpi.association[-1].append(device.env.now+0.2)
        self.handoverFlag = False
        device.sync = False

        if self.handoverExecutionFlag:
            device.kpi.association[-1].append(device.env.now)

            try:
                device.kpi.outofsync.append([device.env.now])
            except:
                device.kpi.outofsync = [[device.env.now]]


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
            if device.sync: #and not device.T310running:
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
                #if device.listedRSRP[targetBS] > device.networkParameters.qualityOut: 
                device.servingBS = targetBS
                if device.servingBSSINR() > device.networkParameters.qualityOut: 

                    # Once the RRC:HO complete is recieved it needs to be processed
                    yield device.env.timeout(device.scenarioBasestations[targetBS].RRCprocessingDelay)

                    # Now the UE is up and downlink synced
                    device.sync = True
                    self.handoverFlag = False
                    device.kpi.association.append(
                        [list(device.scenarioBasestations.keys()).index(device.servingBS), device.env.now])
                    device.kpi.association[-2].append(disassociation)

                    try:
                        device.kpi.log['HIT'].append(device.env.now - disassociation)
                    except KeyError:
                        device.kpi.log['HIT'] = [device.env.now - disassociation]


                else:
                    # Handover fail due to not received handover complete
                    self.loghandoverfailure(device,'HOF004',1)
                    self.handoverFailure(device)
                self.handoverExecutionFlag = False

            # Failed to receive Handover Command
            else:
                self.loghandoverfailure(device,'HOF003',1)
                self.handoverFailure(device)

            self.handoverPreparationFlag = False



class HeuristicHandover(Handover):
    def __init__(self):
        super(HeuristicHandover, self).__init__()
        self.a3 = A3Handover()
        ### To solve composition problems
        self.a3.parent = self

        self.decisionHelper = None
        self.decisionData = None

        self.handoverExecutionFlag = False
        self.handoverPreparationFlag = False
        self.x2Delay = 5 #milliseconds
        self.handoverCount = 0
        self.handoverPreparationStart = 0
        self.handoverFlag = False


    def triggerMeasurement(self, device, targetBS):
        return self.a3.triggerMeasurement(device, targetBS)
        


    def sendMeasurementReport(self, device, targetBS):
        if device.sync:
            # Holds the time to send the RRC:Measurement Report
            yield device.env.timeout(device.networkParameters.RRCMsgTransmissionDelay)

            # Check if it is a pingpong, just for kpi assessment
            if device.lastBS.count(targetBS)>0:
                device.kpi.pingpong += 1

            #Base stations processing the handover at X2 interface
            self.handoverPreparationFlag = True
            self.a3.handoverPreparationFlag = True

            self.handoverPreparationStart = device.env.now

            yield device.env.timeout(
                device.scenarioBasestations[device.servingBS].RRCprocessingDelay
                + device.scenarioBasestations[device.servingBS].handoverDecisionDelay)

            self.decisionData = self.decisionHelper.getData(device, targetBS)
            if (self.decisionHelper.getDecision(*self.decisionData)):
                yield device.env.timeout(
                    + 2*self.x2Delay
                    + 2*device.scenarioBasestations[targetBS].X2processingDelay
                    + device.scenarioBasestations[targetBS].admissionControlDelay)

                # Switch to the new BS
                device.env.process(self.switchBaseStation(device, targetBS))

            else:
                self.handoverPreparationFlag = False
                self.a3.handoverPreparationFlag = False

                self.handoverFlag = False
                self.a3.handoverFlag = False

                device.kpi.handover -=1

                yield device.env.timeout(
                    device.scenarioBasestations[device.servingBS].RRCprocessingDelay
                    + device.scenarioBasestations[device.servingBS].handoverDecisionDelay)
 
    def handoverFailure(self, device):
        return self.a3.handoverFailure(device)
 
    def switchBaseStation(self, device, targetBS):
        return self.a3.switchBaseStation(device, targetBS)

class DecisionHelper:
    def __init__(self):
        pass

    def getDecision(self,*args,**kwargs):
        raise NotImplementedError

    def getData(self,device,targetBS):
        raise NotImplementedError

class PredictionHelper(DecisionHelper):
    def __init__(self):
        super(PredictionHelper, self).__init__()
        self.prediction_window = 0
        self.deteriorate=False

    def getDecision(self,serving_prediction, target_prediction, angles):
        serving_score = self.scoringFunction([serving_prediction,angles[0], angles[-1]])
        target_score = self.scoringFunction([target_prediction,angles[1],angles[-1]])
        #print(serving_score, target_score)

        if target_score <= serving_score:
            return True
        else:
            return False

    def getData(self, device, targetBS):
        init = device.env.now
        end = device.env.now + self.prediction_window
        if not self.deteriorate:
            s_prediction = device.lineofsight[device.servingBS][init:end]
            t_prediction = device.lineofsight[targetBS][init:end]

        ue_angle = np.arctan2(device.Vy,device.Vx)
        s_angle = abs(ue_angle + 
                np.arctan2(device.scenarioBasestations[device.servingBS].y - device.y,
                            device.scenarioBasestations[device.servingBS].x - device.x))
        t_angle = abs(ue_angle + 
                np.arctan2(device.scenarioBasestations[targetBS].y - device.y,
                            device.scenarioBasestations[targetBS].x - device.x))


        ue_speed = np.hypot(device.Vx,device.Vy) 
        s_distance = device.calcDist(device.scenarioBasestations[device.servingBS])
        t_distance = device.calcDist(device.scenarioBasestations[targetBS])

        return [s_prediction, t_prediction, [s_angle, t_angle, ue_speed, s_distance, t_distance, device.pred_offset]]


    def scoringFunction(self, prediction):
        score = 0
        burst = False
        n=0

        for p, i in enumerate(reversed(prediction[0])):
            if i == 1:
                burst=False
                n=0
            else:
                burst=True
                n+=1

            if burst:
                score += (1+np.log2(p+1))*(2**(n/10))

        '''
        Calculate the angular score
        '''
        angular_f = prediction[2] + prediction[1]/(2*np.pi)
        #angular_f = abs(np.log2(prediction[1])/np.log2(2*np.pi))



        return angular_f*np.ceil(score)


class ProbabilityHelper(DecisionHelper):
    def __init__(self):
        super(ProbabilityHelper, self).__init__()
        self.ho_prob = 0
        self.step = 0

    def getDecision(self):
        if np.random.rand() <= self.ho_prob:
            return True
        else:
            return False

    def getData(self, device, targetBS):
        if device.lastBS == targetBS:
            self.updateProb(-1)
        else:
            self.updateProb(+1)

        return None

    def updateProb(self, sense=1):
        self.ho_prob += sense*self.step

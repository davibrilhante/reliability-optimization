import fivegmodules.core
import numpy as np
from decimal import Decimal
import operator

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
        
        self.operator = ''

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
            decision, whichBS = self.decisionHelper.getDecision(device, self.decisionData)

            if decision:
            #if (self.decisionHelper.getDecision(*self.decisionData)):
                yield device.env.timeout(
                    + 2*self.x2Delay
                    #+ 2*device.scenarioBasestations[targetBS].X2processingDelay
                    #+ device.scenarioBasestations[targetBS].admissionControlDelay)
                    + 2*device.scenarioBasestations[whichBS].X2processingDelay
                    + device.scenarioBasestations[whichBS].admissionControlDelay)

                # Switch to the new BS
                #device.env.process(self.switchBaseStation(device, targetBS))
                device.env.process(self.switchBaseStation(device, whichBS))

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
        self.operator = ''

    def getDecision(self,device,BSdata,*args,**kwargs):
        scores = {}
        dists = {}
        decision = True
        ue_angle = np.arctan2(device.Vy,device.Vx)

        for BS, prediction in BSdata.items():

            bs_angle = np.arctan2(device.scenarioBasestations[BS].y - device.y,
                                device.scenarioBasestations[BS].x - device.x)

            angle = bs_angle - ue_angle

            #print(device.x, device.y, BS, device.scenarioBasestations[BS].x, 
            #        device.scenarioBasestations[BS].y, np.degrees(angle))


            if angle > np.pi/2 or angle < -np.pi/2:
                angle = -1
            else:
                angle = 1

            dists[BS] = angle*device.calcDist(device.scenarioBasestations[BS])

            '''
            if dists[BS] < 0:
                scores[BS] = self.scoringFunction(prediction,chemgrid=False)
            else:
            '''
            scores[BS] = self.scoringFunction(prediction,distance=dists[BS],chemgrid=False)

        try:
            device.kpi.log['score'][device.env.now] = scores #{i  : [scores[i], dists[i]] for i in scores.keys()}
        except KeyError:
            device.kpi.log['score'] = {}
            device.kpi.log['score'][device.env.now] = scores #{i  : [scores[i], dists[i]] for i in scores.keys()} 


        '''
        chosenBS = max(scores.items(),key=operator.itemgetter(1))[0]
        '''
        if any(n>0 for n in dists.values()):
            negs = {i : scores[i] for i in scores.keys() if dists[i] >= 0}
            chosenBS = max(negs.items(),key=operator.itemgetter(1))[0]

        else:
            chosenBS = max(scores.items(),key=operator.itemgetter(1))[0]

        if ((chosenBS == device.servingBS)):# or
                #(device.listedRSRP[chosenBS] < -90)):
                    #+ device.networkParameters.handoverOffset
                    #+ device.networkParameters.handoverHysteresys)):
            decision = False


        return decision, chosenBS


    def getData(self, device, targetBS):
        init = device.env.now
        end = device.env.now + self.prediction_window

        predictions = {}
        '''
        Considers the possibility of deteriorating the prediction to make it more real
        '''
        if not self.deteriorate:
            '''
            It will iterate over all the BS and store the predictions on the
            dictionary by the uuid of each BS
            '''
            for bs in device.scenarioBasestations:
                predictions[bs] = device.lineofsight[bs][init:end]
        else:
            raise NotImplementedError

        '''
        Returns the predictions to the decision function
        '''
        return predictions



    def scoringFunction(self, prediction,*args,**kwargs):
        W = len(prediction)

        score = np.zeros(W)
        rsrp = np.zeros(W)

        k = 4
        l = 4

        attraction = 0
        blocks = []
        free = []


        try:
            distance = kwargs['distance']
        except KeyError:
            distance = 0

        try:
            bsradius = kwargs['bsradius']
        except KeyError:
            bsradius = 150


        if self.operator=='' or self.operator=='score':
            n=0
            for i, p in enumerate(prediction):
                if p == 1:
                    n=0
                else:
                    n+=1

                score[i] = (1 - p)*(1+np.log10(100*(W-i)/W))*(2**(n/W))

            return np.mean(score)

        elif self.operator=='root':
            return ((W - sum(prediction))/W)**(1/8)

        elif self.operator=='avgduration':
            return np.mean(self.burst_process(prediction))

        elif self.operator=='nepisodes':
            return len(self.burst_process(prediction))

        elif self.operator=='shortdist':
            return abs(distance)

        elif self.operator=='minmax':
            return max(self.burst_process(prediction))

        elif self.operator=='chemgrid':
            return self.chemgrid(prediction)

        elif self.operator=='maxmin':
            return min(self.rsrp_calc(prediction,distance))

        elif self.operator=='mean':
            return np.mean(self.rsrp_calc(prediction,distance))

        elif self.operator=='meandev':
            return np.dev(self.rsrp_calc(prediction,distance))

        elif self.operator=='movavg':
            return self.movingavgerage(self.rsrp_calc(prediction,distance))

        elif self.operator=='invavg':
            return self.inverseaverage(self.rsrp_calc(prediction,distance))

    def movingaverage(self, rsrp):
        raise NotImplementedError

    def inverseaverage(self, rsrp):
        ordered = sorted(rsrp, reverse=True)
        weights = list(range(len(rsrp)))
        return np.average(ordered,weights=weights)

    def chemgrid(self, prediction):
        raise NotImplementedError

    def burst_process(self,prediction):
        n = 0
        blocks = []
        burst = False
        for i, p in enumerate(prediction):
            if p == 0:
                burst = True
                n+=1
            else:
                if burst:
                    blocks.append(n)
                burst = False
                n = 0
        return blocks

    def rsrp_calc(self, prediction, distance, Filter=False, num=1, den=4): 
        alfa = np.exp(-1*num/den)
        rsrp = np.zeros(len(prediction))
        linkbudget = 23 + 10 + 10 # tx power + bs gain + ue gain
        for p, i in enumerate(prediction):

            rsrp[p] = linkbudget - (72 + 10*2.92*np.log10(abs(distance)) + np.random.normal(0,8.7))
            if i == 1:
                rsrp[p] = linkbudget - (61.4 + 10*2*np.log10(abs(distance)) + np.random.normal(0,5.8))

            if Filter and p != 0:
                rsrp[p] = (1 - alfa)*rsrp[p-1] + alfa*rsrp[p]

        return rsrp



class ProbabilityHelper(DecisionHelper):
    def __init__(self):
        super(ProbabilityHelper, self).__init__()
        self.ho_prob = 0
        self.step = 0

    def getDecision(self, device, targetBS, probability):
        if np.random.rand() <= probability: #self.ho_prob:
            self.updateProb(device, targetBS)
            return True
        else:
            return False

    def getData(self, device, targetBS):
        return [device, targetBS, device.pred_probs[targetBS]]
        


    def updateProb(self, device, targetBS):
        if device.lastBS == targetBS:
            #self.updateProb(-1)
            device.pred_probs[targetBS] = max(0, device.pred_probs[targetBS] - self.step)
        else:
            device.pred_probs[targetBS] = min(1, device.pred_probs[targetBS] + self.step)

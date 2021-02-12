import fivegmodules.core

__all__ = ['Handover', 'A3Handover', 'PredictionHandover']

class Handover:
    def triggerMeasurement(self, device, targetBS):
        raise NotImplementedError
 
    def sendMeasurementReport(self, device, targetBS):
        raise NotImplementedError
 
    def handoverFailure(self, device):
        raise NotImplementedError
 
    def switchBaseStation(self, device, targetBS):
        raise NotImplementedError

class A3Handover(Handover):
 
    def triggerMeasurement(self, device, targetBS):
        counterTTT = 0
                   
        if not device.sync:
            self.handoverFailure()
                   
        # First, check if another measurement is not in progress
        elif not device.measOccurring:
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
                   
                   
                while counterTTT < device.networkParameters.timeToTrigger:
                    yield device.env.timeout(1)
                    counterTTT += 1
 
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
                            break
                    else:
                        # Too late handover, user got out-sync in the middle of it
                        print('Handover failure call 1')
                        self.handoverFailure(device)
                        break

                if counterTTT == device.networkParameters.timeToTrigger:
                    device.kpi.handover +=1
 
                    if device.sync:
                        self.sendMeasurementReport(device, targetBS)
 
                    else:
                        print('Handover failure call 2')
                        self.handoverFailure(device)

                device.measOccurring = False
                device.triggerTime = 0
 
 
 
    def sendMeasurementReport(self, device, targetBS):
        if (device.env.now >= device.triggerTime + device.networkParameters.timeToTrigger) and device.measOccurring:
 
            #Check if it is not a reassociation
            if device.listedRSRP[device.servingBS] != None:
 
                # Check if it is a pingpong, just for kpi assessment
                if device.lastBS.count(targetBS)>0:
                    device.kpi.pingpong += 1
 
                # Switch to the new BS
                device.env.process(self.switchBaseStation(device, targetBS))

 
 
    def handoverFailure(self, device):
        device.lastBS.append(device.servingBS)
        device.servingBS = None
        device.reassociationFlag = False
        device.kpi.handoverFail += 1
        print('Handover Failure!!!')
 
        # It seems there is a sync=False here
        device.sync = False

    def switchBaseStation(self, device, targetBS):
        if device.sync:
            #print('Switching from %s to %s'%(self.servingBS, targetBS))
            device.lastBS.append(device.servingBS)

            # The time gap between the disassociation from the Serving BS to the
            # target BS is known as handover interruption time (HIT) and it is
            # the for the UE to get synced with the target BS. There is no data
            # connection during this time interval, so the UE remains unsynced

            # yields for the BS association delay/HO preparation time
            yield device.env.timeout(device.scenarioBasestations[targetBS].associationDelay)

            '''
            print(self.listBS[self.servingBS].nextSSB, self.listBS[self.servingBS].frameIndex, self.env.now)
            print(defs.BURST_DURATION)
            print(self.listBS[self.servingBS].nextSSB + defs.BURST_DURATION  - self.env.now)

            # yields untill the downlink sync is completed
            yield device.env.timeout(
                    device.scenarioBasestations[targetBS].nextSSB + 
                    device.networkParameters.SSBurstDuration  - device.env.now
                    )
            #'''

            # Receiving handover command and turning to RRC Idle untill uplink 
            # sync with target BS
            if device.sync and not device.T310running:
                device.sync = False

                #print(self.listBS[self.servingBS].nextRach, self.env.now)
                # yields untill the uplink sync is completed
                yield device.env.timeout(
                        device.scenarioBasestations[targetBS].nextRach + 
                        device.networkParameters.SSBurstDuration  - device.env.now
                        )


                if device.listedRSRP[targetBS] > device.networkParameters.qualityOut: 
                    # Now the UE is up and downlink synced
                    device.sync = True
                    device.servingBS = targetBS
                    device.kpi.association.append(
                        [list(device.scenarioBasestations.keys()).index(device.servingBS), device.env.now])
                else:
                    print('Handover Failure call 3')
                    self.handoverFailure(device)

            # Failed to receive handover command
            else:
                print('Handover Failure call 4')
                self.handoverFailure(device)



class PredictionHandover(Handover):
    pass

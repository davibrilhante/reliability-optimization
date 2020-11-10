#! /usr/bin/env python3

# -*- coding : utf8 -*-

import numpy as np
from matplotlib import pyplot as plt
from json import load, dump
from argparse import ArgumentParser

parser = ArgumentParser()
#parser.add_argument('-i','--input')
#parser.add_argument('-j','--json')

parser.add_argument('-s','--speed')
parser.add_argument('-b','--block')
args = parser.parse_args()



def avgBlockageDuration(blockageInfo : list):
    result = []
    for basestation in blockageInfo:
        avgBaseStation = []
        for userEquip in basestation:
            counter = 0
            blockFlag = False
            for block in userEquip:
                if block == 1 and blockFlag:
                    avgBaseStation.append(counter)
                    blockFlag = False
                    counter = 0

                if block == 0:
                    blockFlag = True
                    counter += 1

            #avgBaseStation.append(counter/len(userEquip))
        result.append(np.nanmean(avgBaseStation))
    return np.nanmean(result), np.std(result)


def avgAntiBlockageDuration(blockageInfo : list):
    result = []
    for basestation in blockageInfo:
        avgBaseStation = []
        for userEquip in basestation:
            counter = 0
            blockFlag = False
            for block in userEquip:
                if block == 0 and blockFlag:
                    avgBaseStation.append(counter)
                    blockFlag = False
                    counter = 0

                if block == 1:
                    blockFlag = True
                    counter += 1

            #avgBaseStation.append(counter/len(userEquip))
        result.append(np.nanmean(avgBaseStation))
    return np.nanmean(result), np.std(result)


def avgSightArea(simulationTime : int, userEquip : dict, basestation: list, ttt : int, step = 1):
    result = []
    time = 0

    
    while time < simulationTime:
        result.append([])
        for bs in basestation:
            bsX = bs['position']['x']
            bsY = bs['position']['y']

            ueX = userEquip['position']['x'] + (userEquip['speed']['x']/3.6)*time*1e-3
            ueY = userEquip['position']['y'] + (userEquip['speed']['y']/3.6)*time*1e-3
            dist = np.hypot(bsX-ueX, bsY-ueY)

            #if dist <= 200:
                #print(bs['uuid'])
            '''

            finalX = ueX + (userEquip['speed']['x']/3.6)*ttt*1e-3
            finalY = ueY + (userEquip['speed']['y']/3.6)*ttt*1e-3

            sideB = np.hypot(ueX-finalX, ueY-finalY)

            sideC = np.hypot(bsX-finalX, bsY-finalY)

            halfPerimeter = (dist + sideB + sideC)/2

            #      link budget - path loss (in LOS condition)
            receivedPower = 40 - (61.4 + 20.0*np.log10(dist))
            print(receivedPower)
            if receivedPower > -90:
                area = np.sqrt(
                        halfPerimeter*(halfPerimeter - dist)*(halfPerimeter-sideB)*(halfPerimeter-sideC)
                        )
            else:
                area = float('inf')
            '''
            #angle = np.arctan2(bsY-ueY,bsX-ueX)
            area = dist*(userEquip['speed']['x']/3.6)#*np.cos(angle)))

            
            result[-1].append(area)

        time += step

    return result 


def handoverProbability(areaSBS, areaNBS, numberNeighbours, blockDuration, 
        freeDuration, blockDensity, tau):

    if blockDuration == 0:
        return 0

    expSBS = blockDensity*areaSBS
    expNBS = blockDensity*areaNBS

    expBlock = (-1/blockDuration)*tau
    #expFree = (-1/Duration)*tau

    blockDurationProb = np.exp(expBlock) #/blockDuration
    oneOrMoreObstaclesProb = (1 - np.exp(-1*expSBS))

    #freeDurationProb = np.exp(expFree)
    freeDurationProb = 1/mu
    noObstacleProb = np.exp(-1*expNBS)

    #probability = (1 - np.exp(-1*expSBS))*blockDurationProb
    #print(probability)

    #probability *= (1 - blockDurationProb)*numberNeighbours*np.exp(expNBS) 
    #probability *= numberNeighbours*np.exp(-1*expNBS)*freeDurationProb 
    print(blockDurationProb,oneOrMoreObstaclesProb,freeDurationProb,noObstacleProb)

    probability = blockDurationProb*oneOrMoreObstaclesProb
    probability *= freeDurationProb*noObstacleProb

    return probability



def handoverExpectation(areaSBS, areaNBS, numberNeighbours, blockDuration, 
        freeDuration, blockDensity, tau):

    handoverProb = handoverProbability(areaSBS, areaNBS, numberNeighbours,
            blockDuration, freeDuration, blockDensity, tau)

    #print(handoverProb)

    expectedValue = handoverProb/(1 - handoverProb)**2

    return expectedValue

class MarkovChain(object):
    def __init__(self, n):
        self.nStates = n

    def populateMatrix(self, step):
        self.transitionMatrix = np.zeros((self.nStates, self.nStates))
        for actual in range(self.nStates):
            for _next in range(self.nStates):
                if actual == _next:
                    continue
                else:
                    #dist = np.hypot(self.bspos[actual][0] - self.bspos[_next][0],
                    #        self.bspos[actual][1] - self.bspos[_next][1])

                    #if dist <= 500: 
                    self.transitionMatrix[actual][_next] = handoverProbability(
                            self.areas[step][actual], self.areas[step][_next],
                            1, self.blockDuration, self.freeDuration,
                            self.blockageDensity, self.timeToTrigger
                            )
                    #else:
                    #    self.transitionMatrix[actual][_next] = 0

            self.transitionMatrix[actual][actual] = 1 - sum(self.transitionMatrix[actual])

    def setHandoverParams(self, areas, mu, csi, density, tau, basestations):
        '''
        if len(areas) < self.nStates:
            print("Invalid area array passed!")
            exit()
        
        else:
        '''
        self.areas = areas
        self.blockDuration = mu
        self.freeDuration = csi
        self.blockageDensity = density
        self.timeToTrigger = tau

        self.bspos = np.zeros((len(basestations), 2))

        for n, bs in enumerate(basestations):
            self.bspos[n][0] = bs['position']['x']
            self.bspos[n][1] = bs['position']['y']


    
    def chainRandomWalk(self, steps):
        state = 0 #initial state
        handovers=0

        self.populateMatrix(0)
        wheight = self.transitionMatrix[state]
        for n in range(1,steps):
            print(self.areas[n])
            print(state, wheight)
            #print(wheight)
            nextState = np.random.choice(self.nStates,1, p=wheight)
            #nextState = np.random.choice(self.nStates,1)

            if nextState != state:
                handovers+=1

            state = nextState
            self.populateMatrix(n)
            wheight = self.transitionMatrix[state][0]

        return handovers



    def monteCarloSimulation(self, simulations, steps):
        results = np.zeros(simulations)
        for n in range(simulations):
            results[n] = self.chainRandomWalk(steps)

        return np.mean(results), np.std(results)


TTT = [(2**i)*10 for i in range(3,10)]
ueSpeed = [str(7*i + 22) for i in range(3,10)]
ueSpeed = [args.speed]
blockageDensity = ['{:.3f}'.format(0.001 + i*0.001) for i in range(10)]
blockageDensity = [args.block]
execs = 30

evaluation = {str(i) : {j : {k : [] for k in blockageDensity} for j in ueSpeed} for i in TTT}

for s in ueSpeed:
    for b in blockageDensity:
        tempMu = []
        tempCsi = []
        for x in range(execs):
            #filename = 'out/'+s+'/'+b+'/'+str(x)
            filename = 'instances/'+s+'/'+b+'/'+str(x)
            try:
                with open(filename, 'r') as jsonfile:
                    data = load(jsonfile)
            except Exception as e:
                print(e)
                continue

            #tau = 640
            numberBS = len(data['baseStation'])
            numberUE = len(data['userEquipment'])
            simulationTime = data['scenario']['simTime']

            parameterMu, muStd = avgBlockageDuration(data['blockage'])
            parameterCsi, csiStd = avgAntiBlockageDuration(data['blockage'])
            #print(parameterMu, muStd)
            tempMu.append(parameterMu)
            tempCsi.append(parameterCsi)

        area = 58.457
        mu = np.mean(tempMu)*1e-3
        csi = np.mean(tempCsi)*1e-3
        print(mu, csi)

        for tau in TTT:
            print(s,b,tau)
            step = tau 
            #print("Calc Area!")
            parameterArea = avgSightArea(simulationTime, data['userEquipment'][0],
                                data['baseStation'], tau, step)
            #parameterArea = [area for i in data['baseStation']]

            nStates = len(data['baseStation'])
            chain = MarkovChain(nStates)

            chain.setHandoverParams(parameterArea, mu, csi, float(b), tau*1e-3, data['baseStation'])
            #chain.populateMatrix()
            #print(chain.transitionMatrix)
            evaluation[str(tau)][s][b], std = chain.monteCarloSimulation(int(1e4), int(simulationTime/step))

            #print(parameterArea)

            #expectedHO = handoverExpectation(area, area, 6, mu, csi, float(b), tau*1e-3)

            #evaluation[str(tau)][s][b] = expectedHO

            print(tau, evaluation[str(tau)][s][b], std)

outname = args.speed+'-'+args.block
with open(outname,'w') as outfile:
    dump(evaluation, outfile)

'''
for speed in ueSpeed:
    for block in blockageDensity:
        temp = []
        for t in TTT:
            temp.append(evaluation[str(t)][s][b])
        plt.plot(TTT, temp)

plt.xlabel("Time to Trigger")
plt.ylabel("Expected number of Hand Over")
plt.grid()
plt.legend()
plt.show()
'''

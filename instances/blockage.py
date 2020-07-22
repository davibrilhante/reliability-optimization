import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString




def blockers_gen(rate, scenario, Plot = False):
    #The blockage rate is given in square meters blocked
    blockers = {}
    blockers['objects'] = []
    totalArea = (scenario['boundaries']['xmax']-scenario['boundaries']['xmin'])*(
                scenario['boundaries']['ymax']-scenario['boundaries']['ymin'])

    totalBlockers = np.random.poisson(rate*totalArea)

    nBlockers = 0 
    while(nBlockers < totalBlockers):#rate*totalArea):
        blockers['objects'].append({})
        blockers['objects'][-1]['nVertices'] = 4 #All objects are quadrilaterals

        centroidX = np.random.randint(scenario['boundaries']['xmin'],scenario['boundaries']['xmax'])
        centroidY = np.random.randint(scenario['boundaries']['ymin'],scenario['boundaries']['ymax'])
        blockers['objects'][-1]['centroid'] = {
                'x':centroidX,
                'y':centroidY
                } 

        blockers['objects'][-1]['polygon'] = Polygon([(centroidX - 0.5, centroidY + 0.5),
            (centroidX + 0.5, centroidY + 0.5),
            (centroidX + 0.5, centroidY - 0.5),
            (centroidX - 0.5, centroidY - 0.5)])

        if Plot:
            x, y = blockers['objects'][-1]['polygon'].exterior.xy
            plt.plot(x, y, color='red')

        nBlockers += 1
    return blockers


def blockage(rate, scenario, baseStations, userEquipments, Filter=1, Plot = False):
    blockers = blockers_gen(rate, scenario,Plot)

    blockage = []
    gamma = []
    for m, bs in enumerate(baseStations):
        #print('\n')
        #print(m,end='')
        blockage.append([])
        gamma.append([])
        for n, user in enumerate(userEquipments):
            #print(n,end='')
            blockage[m].append([])
            gamma[m].append([])
            listOfBlockers = []
            for t in range(scenario['simTime']):
                gamma[m][n].append(0)
                ueNewX = user['position']['x'] + (user['speed']['x']/3.6)*(t*1e-3)
                ueNewY = user['position']['y'] + (user['speed']['y']/3.6)*(t*1e-3)
                block = False
                line = LineString([(bs['position']['x'], bs['position']['y']), (ueNewX, ueNewY)])
                for b in blockers['objects']:
                    if line.crosses(b['polygon']):
                        index = blockers['objects'].index(b)
                        if listOfBlockers.count(index) == 0:
                            listOfBlockers.append(index)
                        block = True
                if block:
                    blockage[m][n].append(0)
                else:
                    blockage[m][n].append(1)
                if t == 0:
                    gamma[m][n][t] = (1 - Filter)*gamma[m][n][t] + Filter*blockageScore(blockers, listOfBlockers, [ueNewX, ueNewY])
                else:
                    gamma[m][n][t] = blockageScore(blockers, listOfBlockers, [ueNewX, ueNewY])

    if Plot:
        for i, bs in enumerate(baseStations):
            plt.scatter(bs['position']['x'], bs['position']['y'], marker='D', color='blue')
        plt.show()
        colors = ['blue', 'green', 'orange', 'red']
        for i, bs in enumerate(baseStations):

            plt.plot(gamma[i][0], label='BS '+str(i), color=colors[i])
        plt.legend()
        plt.show()


    return blockage, gamma

def blockageScore(blockers, blockerList, uePos, delta=None):
    if delta == 1 or delta == None:
        delta = lambda x1, y1, x2, y2: 1/np.hypot(x1-x2, y1-y2)
    elif delta == 2:
        delta = lambda x1, y1, x2, y2: np.exp(-1*np.hypot(x1-x2, y1-y2))

    gamma = 0
    for i in blockerList:
        gamma += delta(blockers['objects'][i]['polygon'].centroid.x, uePos[0], 
                blockers['objects'][i]['polygon'].centroid.y, uePos[1])
    return gamma


def blockage2(rate, scenario, baseStations, userEquipments, Plot = False, prediction=50):
    blockers = blockers_gen(rate, scenario,False)#Plot)

    blockage = []
    gamma = []
    for m, bs in enumerate(baseStations):
        blockage.append([])
        gamma.append([])
        for n, user in enumerate(userEquipments):
            blockage[m].append([])
            gamma[m].append([])
            listOfBlockers = []
            for t in range(scenario['simTime']):
                gamma[m][n].append(0)
                futureBlockers = []

                block = False
                window = min(prediction, scenario['simTime']-t)
                ueNewX = user['position']['x'] + (user['speed']['x']/3.6)*(t)*1e-3
                ueNewY = user['position']['y'] + (user['speed']['y']/3.6)*(t)*1e-3
                for i in range(window): 
                    futureBlockers.append([])
                    tempX = ueNewX + (user['speed']['x']/3.6)*(i)*1e-3
                    tempY = ueNewY + (user['speed']['y']/3.6)*(i)*1e-3

                    line = LineString([(bs['position']['x'], bs['position']['y']), (ueNewX, ueNewY)])
                    for b in blockers['objects']:
                        cross = line.crosses(b['polygon'])
                        if cross:
                            index = blockers['objects'].index(b)
                            if i == 0:
                                if listOfBlockers.count(index) == 0:
                                    listOfBlockers.append(index)
                                block = True
                            else:
                                if futureBlockers[:i].count(index) == 0:
                                    futureBlockers[i].append(index)

                if block:
                    blockage[m][n].append(0)
                else:
                    blockage[m][n].append(1)

                #gamma[m][n][t] = blockageScore(blockers, listOfBlockers, [ueNewX, ueNewY])
                gamma[m][n][t] = blockageScore2(blockers, listOfBlockers, futureBlockers, 
                        [ueNewX, ueNewY], [user['speed']['x'],user['speed']['y']], prediction)

    if Plot:
        colors = ['blue', 'green', 'orange', 'red']
        '''
        for i, bs in enumerate(baseStations):
            plt.scatter(bs['position']['x'], bs['position']['y'], marker='D', color=colors[i], label='BS '+str(i))
        plt.legend(ncol=4, pos=0)
        plt.show()
        '''
        for i, bs in enumerate(baseStations):
            plt.plot(gamma[i][0], label='BS '+str(i), color=colors[i])
        plt.legend()
        plt.show()

    return blockage, gamma


def blockageScore2(blockers, blockerList, futureList, uePos, speed, prediction, delta=None):
    if delta == 1 or delta == None:
        delta = lambda x1, y1, x2, y2: 1/np.hypot(x1-x2, y1-y2)
    elif delta == 2:
        delta = lambda x1, y1, x2, y2: np.exp(-1*np.hypot(x1-x2, y1-y2))

    gamma = 0

    for t in range(prediction):
        x = uePos[0] + t*speed[0]
        y = uePos[1] + t*speed[1]
        if t == 0:
            for i in blockerList:
                gamma += (1/prediction) * delta(blockers['objects'][i]['polygon'].centroid.x, x, 
                        blockers['objects'][i]['polygon'].centroid.y, y)
        else:
            for i in futureList[:t]:
                #gamma += (1- t/prediction) * delta(blockers['objects'][i]['polygon'].centroid.x, x, 
                #        blockers['objects'][i]['polygon'].centroid.y, y)
                for j in i:
                    gamma += (1/prediction) * delta(blockers['objects'][j]['polygon'].centroid.x, x, 
                            blockers['objects'][j]['polygon'].centroid.y, y)

    return gamma


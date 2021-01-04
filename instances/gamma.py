#! /usr/bin/env python3
# -*- coding : utf8 -*-

from json import load, dump, dumps
from argparse import ArgumentParser
from shapely.geometry import Polygon, LineString
from numpy import mean, hypot, exp

from decompressor import decompressor

def blockageScore(blockers, blockerList, ue, timeslot, opt=None):
    if opt == 1 or opt == None:
        delta = lambda x1, y1, x2, y2: 1/hypot(x1-x2, y1-y2)

    elif opt == 2:
        delta = lambda x1, y1, x2, y2, s: exp(-1*hypot(x1-x2, y1-y2))

    elif not callable(opt):
        print('Error: Not a function or a valid default option given')
        exit()

    ueNewX = ue['position']['x'] + (ue['speed']['x']/3.6)*(timeslot*1e-3)
    ueNewY = ue['position']['y'] + (ue['speed']['y']/3.6)*(timeslot*1e-3)

    gamma = 0
    speed = hypot(ue['speed']['x'], ue['speed']['y'])
    for blocker in blockerList:
        gamma += delta(blocker.centroid.x, blocker.centroid.y,
                ueNewX, ueNewY, speed)
    return gamma


parser = ArgumentParser()
parser.add_argument('-s','--speed')
parser.add_argument('-b','--block')
#parser.add_argument('-i','--instance')

args = parser.parse_args()

seeds = [0] #[i for i in range(30)]

for s in seeds:
    #print('Actual seed: ', s)
    filename = 'out/'+args.speed+'/'+args.block+'/'+str(s)
    with open(filename,'r') as jsonfile:
        data = load(jsonfile)

    decompressor(data)

    #print('Number of blockers: ', len(data['blockers']))

    # Trasform the obstacles in polygons
    blockers = []
    for obstacle in data['blockers']:
        centroidX = obstacle['x']
        centroidY = obstacle['y']
        blockers.append(
                Polygon(
                [(centroidX - 0.5, centroidY + 0.5),
                (centroidX + 0.5, centroidY + 0.5),
                (centroidX + 0.5, centroidY - 0.5),
                (centroidX - 0.5, centroidY - 0.5)]
                        )
                )

    gamma = []
    for m, bs in enumerate(data['baseStation']):
        gamma.append([])
        for n, ue in enumerate(data['userEquipment']):
            gamma[m].append([])
            listOfBlockers = []
            for t in range(data['scenario']['simTime']):

                gamma[m][n].append(0)

                ueNewX = ue['position']['x'] + (ue['speed']['x']/3.6)*(t*1e-3)
                ueNewY = ue['position']['y'] + (ue['speed']['y']/3.6)*(t*1e-3)

                line = LineString([(bs['position']['x'], bs['position']['y']), (ueNewX, ueNewY)])

                if data['blockage'][m][n][t] == 0:
                    #Yep, there is a blockage but where?
                    for b in blockers:
                        if line.intersects(b):
                            listOfBlockers.append(b) if b not in listOfBlockers else listOfBlockers
                            #break

                #gamma[m][n][t] = blockageScore(blockers, listOfBlockers, [ueNewX, ueNewY])
                gamma[m][n][t] = blockageScore(blockers, listOfBlockers, ue, t, 2)
                '''
                if t%1000 == 0:
                    print(t, gamma[m][n][t], ueNewX, ueNewY)
                '''
    y = dumps(gamma)
    print(y)
    '''
    with open('gamma/'+args.speed+'/'+args.block+'/'+str(s), 'w') as out:
        try:
            dump(gamma, out)
            out.close()
        except Exception as e:
            print(e)
            exit()
    '''

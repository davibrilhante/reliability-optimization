from json import load
from argparse import ArgumentParser
from shapely.geometry import Polygon, LineString
from numpy.random import randint
from numpy import mean

parser = ArgumentParser()
parser.add_argument('-s','--speed')
parser.add_argument('-b','--block')
#parser.add_argument('-i','--instance')

args = parser.parse_args()

seeds = ['20']#randint(0,30, 3)

avgErrors = []
for s in seeds:
    print('Chosen seed: ', s)
    filename = args.speed+'/'+args.block+'/'+str(s)
    with open(filename,'r') as jsonfile:
        data = load(jsonfile)

    print('Number of blockers: ', len(data['blockers']))

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

    # Check if the blockage match in time
    errors = 0
    ones = 0
    zeros = 0
    for m, bs in enumerate(data['baseStation']):
        for n, ue in enumerate(data['userEquipment']):
            for t in range(data['scenario']['simTime']):
                ueNewX = ue['position']['x'] + (ue['speed']['x']/3.6)*(t*1e-3)
                ueNewY = ue['position']['y'] + (ue['speed']['y']/3.6)*(t*1e-3)

                line = LineString([(bs['position']['x'], bs['position']['y']), (ueNewX, ueNewY)])

                isBlockage = True
                for b in blockers:
                    isBlockage = line.intersects(b)
                    if isBlockage:
                        break

                if isBlockage != 1 - data['blockage'][m][n][t]:
                    if data['blockage'][m][n][t]==1:
                        ones += 1
                    else:
                        zeros += 1

                    errors+= 1

    print("Errors in blockage: ", errors)
    print("Errors to 1:", ones)
    print("Errors to 0:", zeros)
    print("Percentage errors in blockage: ", 
            errors/(data['scenario']['simTime']*len(data['baseStation'])*len(data['userEquipment'])))
    avgErrors.append(errors)

print('Average Errors in this Conf: ',mean(avgErrors))

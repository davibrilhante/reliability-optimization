import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import MultiPoint


class SuperLine(LineString):
    def __init__(self, points):
        super().__init__(points)

    def slopecoefficient(self):
        coords = list(self.coords)
        if coords[1][0] == coords[0][0]:
            # Vertical line
            return float('inf')
        elif coords[1][1] == coords[0][1]:
            # Horizontal line
            return 0
        else:
            return (coords[1][1]-coords[0][1])/(coords[1][0]-coords[0][0])

    def slopeangle(self):
        coords = list(self.coords)
        if coords[1][0] == coords[1][0]:
            # Vertical line
            return float(np.pi/2)
        elif coords[1][1] == coords[0][1]:
            # Horizontal line
            return 0
        else:
            return np.arctan2(coords[1][1]-coords[0][1],coords[1][0]-coords[0][0])

    def linearequation(self, form=0):
        coords = list(self.coords)
        if form == 0:
            slope = self.slopecoefficient()
            y0 = coords[1][1] - slope*coords[1][0]
            return [slope, y0]

        elif form == 1:
            # return the line equation given two points in the form ax + by + c
            a = coords[0][1] - coords[1][1]
            b = coords[1][0] - coords[0][0]
            c = coords[0][0]*coords[1][1] - coords[1][0]*coords[0][1]
            return [a, b, c]

    def whereintersects(self, geometry, infinity=True):
        if geometry.geom_type == 'Point':
            if self.intersects(geometry):
                #In this case, the intersection is the point itself
                return geometry
            else:
                return False

        elif geometry.geom_type == 'LineString':
            coords = list(geometry.coords)
            equation1 = self.linearequation()

            #In the case you dont want to consider the line infinity
            if not infinity:
                if not self.intersects(geometry):
                    return False

            if self.contains(geometry):
                x = coords[0][0]
                y = coords[0][1]
                return Point([x,y])

            # Vertical line
            elif coords[1][0] == coords[0][0]:
                #They are parallel
                if equation1[0] == float('inf'):
                    return False

                x = coords[1][0]
                y = equation1[0]*coords[1][0] + equation1[1]
                #print('v', [x,y])
                return Point([x,y])

            # Horizontal line
            elif coords[1][1] == coords[0][1]:
                #They are parallel
                if equation1[0] == 0:
                    return False

                y = coords[1][1]
                x = (coords[1][1] - equation1[1])/equation1[0]
                #print('h', [x,y])
                return Point([x,y])

            else:
                slope = (coords[1][1]-coords[0][1])/(coords[1][0]-coords[0][0])
                b = coords[1][1] - slope*coords[1][0]
                equation2 = [slope, b]

                if slope == equation1[0]:
                    # They are parallel
                    return False

                else:
                    x = (equation2[1] - equation1[1])/(equation1[0] - equation2[0])
                    y = slope*x + b
                    return Point([x, y])

        elif geometry.geom_type == 'Polygon':
            if self.intersects(geometry):
                #generate the polygon boundaries
                intersectionpoints = []
                polypoints = list(geometry.exterior.coords)
                for i in range(len(polypoints)-1):
                    polygonside = LineString([polypoints[i]]+[polypoints[i+1]])
                    temp = self.whereintersects(polygonside, False)
                    if temp:
                        intersectionpoints.append(temp)

                return intersectionpoints

            else:
                return False

        else:
            print('Not a valid geometry type')

    def plot(self, limits):
        equation = self.linearequation()
        x = np.linspace(0,limits,100)
        y = equation[0]*x + equation[1]
        plt.plot(x, y)




def blockers_gen(rate, scenario, route, Plot = False, tolerance = 150):
    #The blockage rate is given in square meters blocked
    blockers = {}
    blockers['objects'] = []
    '''
    totalArea = (scenario['boundaries']['xmax']-scenario['boundaries']['xmin'])*(
                scenario['boundaries']['ymax']-scenario['boundaries']['ymin'])
    routeBuffer = route.buffer(tolerance, cap_style=3)
    totalArea = routeBuffer.area - 2*tolerance*np.sqrt(((tolerance**2)/2) - (tolerance/2)**2)
    #'''

    routeBuffer = MultiPoint([(scenario['boundaries']['xmin'], scenario['boundaries']['ymin']),
        (scenario['boundaries']['xmin'] + tolerance, scenario['boundaries']['ymin']),
        (scenario['boundaries']['xmin'], scenario['boundaries']['xmin'] + tolerance),
        (scenario['boundaries']['xmax'] - tolerance, scenario['boundaries']['ymax']),
        (scenario['boundaries']['xmax'], scenario['boundaries']['xmax'] - tolerance),
        (scenario['boundaries']['xmax'], scenario['boundaries']['ymax'])]).convex_hull

    totalArea = routeBuffer.area

    totalBlockers = np.random.poisson(rate*totalArea)
    #print(totalArea)
    #print(totalBlockers)

    nBlockers = 0 
    while(nBlockers < totalBlockers):#rate*totalArea):
        centroidX = np.random.randint(scenario['boundaries']['xmin'],scenario['boundaries']['xmax'])
        centroidY = np.random.randint(scenario['boundaries']['ymin'],scenario['boundaries']['ymax'])

        point = Point([centroidX, centroidY])

        '''    
        if type(route) == type(LineString([(0,0),(1,1)])):
            upperlimit = route.parallel_offset(tolerance,'left')
            lowerlimit = route.parallel_offset(tolerance,'right')

            if not (upperlimit.crosses(point) or lowerlimit.crosses(point)):
                continue
        '''
        if not (routeBuffer.contains(point)):
            continue

        blockers['objects'].append({})
        blockers['objects'][-1]['nVertices'] = 4 #All objects are quadrilaterals



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
    plt.show()
    return blockers


def advanceintime(polygon, uex, uey, ue, bs):
    bspoint = Point(bs['position']['x'], bs['position']['y'])
    uepoint = Point(uex, uey)
    uetobs = SuperLine(list(uepoint.coords)+list(bspoint.coords))
    ueroute = SuperLine([(uex, uey),(uex+ue['speed']['x'], uey+ue['speed']['y'])])

    #print(list(bspoint.coords))

    uedirectionx = ue['speed']['x']/abs(ue['speed']['x'])
    uedirectiony = ue['speed']['y']/abs(ue['speed']['y'])

    points = uetobs.whereintersects(polygon)
    #print([list(i.coords) for i in points])
    #uetobs.plot(20)

    polypoints = list(polygon.exterior.coords)
    polypoints.pop()
    dist = []
    for i in range(len(polypoints)):
        bstovertex = SuperLine(list(bspoint.coords)+[polypoints[i]])
        intersectionpoint = ueroute.whereintersects(bstovertex)
        intersectionpoint = list(intersectionpoint.coords)[0]
        if (uedirectionx*(intersectionpoint[0] - uex) >= 0 and
                uedirectiony*(intersectionpoint[1] - uey) >= 0):
            dist.append(np.hypot(intersectionpoint[0]-uex, intersectionpoint[1]-uey))
        else:
            dist.append(-1*np.hypot(intersectionpoint[0]-uex, intersectionpoint[1]-uey))

    darkline = max(dist)
    extremepoint = polypoints[dist.index(darkline)]
    #print(extremepoint)
    #print(darkline)
    bstoextreme = SuperLine(list(bspoint.coords)+[extremepoint])
    #bstoextreme.plot(20)

    vfinal = (np.hypot(ue['speed']['x'], ue['speed']['y'])/3.6)

    return (darkline/vfinal)*1e3



def blockage(rate, scenario, baseStations, userEquipments, Filter=1, tolerance = 150, Plot = False):
    '''
    ueRoute = LineString([(userEquipments[0]['position']['x'],userEquipments[0]['position']['y']),
            (userEquipments[0]['position']['x'] + (userEquipments[0]['speed']['x']/3.6)*(scenario['simTime']*1e-3),
            userEquipments[0]['position']['y'] + (userEquipments[0]['speed']['y']/3.6)*(scenario['simTime']*1e-3))])
    #'''

    ueRoute = LineString([(scenario['boundaries']['xmin'], scenario['boundaries']['ymin']), 
        (scenario['boundaries']['xmax'], scenario['boundaries']['ymax'])])

    blockers = blockers_gen(rate, scenario, ueRoute, Plot, tolerance)#Plot)

    blockage = []
    gamma = []
    for m, bs in enumerate(baseStations):
        #print('\n')
        #print(m,end='')
        blockage.append([])
        gamma.append([])
        for n, user in enumerate(userEquipments):
            '''
            ### Sort the obstacles by their distances to the UE route
            ueRoute = LineString([(user['position']['x'],user['position']['y']),
                    (user['position']['x'] + (user['speed']['x']/3.6)*(scenario['simTime']*1e-3),
                    user['position']['y'] + (user['speed']['y']/3.6)*(scenario['simTime']*1e-3))])
            #origin = Point(0,0)
            origin = Point(bs['position']['x'],bs['position']['y'])

            dists = [origin.distance(i['polygon']) for i in blockers['objects']]
            temp = list(zip(dists,blockers['objects']))

            sortblockers = [x for _,x in sorted(temp, key=lambda pair:pair[0])]
            '''

            #print(n,end='')
            blockage[m].append([])
            #gamma[m].append([])
            listOfBlockers = []
            advance = float('-inf')
            for t in range(scenario['simTime']):
                block = False
                #if t%1000 == 0:
                #    print(t)

                #gamma[m][n].append(0)
                if t <= advance:
                    block = True

                else:
                    ueNewX = user['position']['x'] + (user['speed']['x']/3.6)*(t*1e-3)
                    ueNewY = user['position']['y'] + (user['speed']['y']/3.6)*(t*1e-3)

                    line = LineString([(bs['position']['x'], bs['position']['y']), (ueNewX, ueNewY)])

                    #for b in sortblockers:
                    for b in blockers['objects']:
                        '''
                        center = list(b['polygon'].centroid.coords)[0]
                        if ((center[0] + 10 - bs['position']['x'] < 0) and  
                                (ueNewX - bs['position']['x'] > center[0] + 10 - bs['position']['x'])):
                            continue
                        '''

                        #if line.crosses(b['polygon']):
                        if line.intersects(b['polygon']):
                            #print(t, ueNewX, ueNewY)
                            advance = t + advanceintime(b['polygon'], ueNewX, ueNewY, user, bs)
                            #print(advance)
                            '''
                            index = blockers['objects'].index(b)
                            if listOfBlockers.count(index) == 0:
                                listOfBlockers.append(index)
                            '''
                            block = True


                        '''
                        elif ((center[0] - 10 - bs['position']['x'] > 0) and  
                                (center[0] - 10 - bs['position']['x'] > ueNewX - bs['position']['x'])):
                            break
                        '''

                if block:
                    blockage[m][n].append(0)
                else:
                    blockage[m][n].append(1)
                '''
                if t == 0:
                    gamma[m][n][t] = (1 - Filter)*gamma[m][n][t] + Filter*blockageScore(blockers, listOfBlockers, [ueNewX, ueNewY])
                else:
                    gamma[m][n][t] = blockageScore(blockers, listOfBlockers, [ueNewX, ueNewY])
                '''

    if Plot:
        for i, bs in enumerate(baseStations):
            plt.scatter(bs['position']['x'], bs['position']['y'], marker='D', color='blue')
        plt.show()
        colors = ['blue', 'green', 'orange', 'red']
        '''
        for i, bs in enumerate(baseStations):

            plt.plot(gamma[i][0], label='BS '+str(i), color=colors[i])
        plt.legend()
        plt.show()
        '''


    return [i['centroid'] for i in blockers['objects']], blockage#, gamma

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


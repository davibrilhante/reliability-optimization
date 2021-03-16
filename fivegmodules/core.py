from numpy import hypot
from numpy import arctan2
from simpy import Environment

from fivegmodules.plot import PlotNetwork


class Scenario(Environment):
    def __init__ (self, simTime):
        super(Scenario, self).__init__()
        self.simTime = simTime
         
        self.userEquipments = None
        self.baseStations = None

        self.frequency = None
        self.wavelength = None

         
    def addBaseStations(self, baseStations : dict):
        '''
        BaseStation must be a dictionary like
        { 'uuid' : <__class__.BaseStation>}
        '''
        self.baseStations = baseStations
         
    def addUserEquipments(self, userEquipments : dict):
        '''
        BaseStation must be a dictionary like
        { 'uuid' : <__class__.MobileUser>}
        '''
        self.userEquipments = userEquipments

    def plot(self):
        self.plotter = PlotNetwork(self.baseStations, self.userEquipments)
        self.plotter.plot()


class Channel:
    def __init__ (self):
        self.switchShadowing = None # Bool
        self.switchFading = None # Bool
        self.noisePower = None # dBm
        self.channelGain = None # dB

    def pathLossCalc(self, reference, exponent, distance, **params):
        raise NotImplementedError



class WirelessDevice:
    def __init__ (self, scenario, inDict):
        self.x = inDict['position']['x'] if inDict else 0
        self.y = inDict['position']['y'] if inDict else 0
        self.id = inDict['uuid'] if inDict else None
        self.uuid = inDict['uuid'] if inDict else None
        self.env = scenario
        self.channel = None
 
        self.txPower = inDict['txPower'] if inDict else 30 #dBm
        self.antenna = None
        self.sensibility = inDict['sensibility'] if inDict or inDict['sensibility'] else -110 #dBm
 
        self.lineofsight = { }
 
    ### Initialize Services Interface
    def initializeServices(self, **params):
        raise NotImplementedError
 
    # This method processes the LOS info from the instance
    def addLosInfo(self, los : list, n : int) -> list:
        raise NotImplementedError
 
    def calcDist(self, device) -> float :
        '''
        Returns the distance between this and some other device
        '''
        return hypot(self.x - device.x, self.y - device.y)

    def calcAngle(self, device) -> float:
        '''
        Returns the the angle between user and cartesian plane
        defined with base station at the center
        '''
        return arctan2(device.y - self.y, device.x - self.x)

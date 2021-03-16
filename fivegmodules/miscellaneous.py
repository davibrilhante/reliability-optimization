from numpy import log10, sqrt, pi, zeros
from numpy import sin, cos
from numpy.random import normal, uniform
from simpy import Environment

from fivegmodules.core import Scenario
from fivegmodules.core import Channel

__all__ = ['Scenario', 'Channel', 'AWGNChannel', 'Antenna', 'UniformLinearArray',
            'UniformPlanarArray', 'DummyAntenna', 'ConePlusCircle', 'Packet', 
            'DataPacket']

class AWGNChannel(Channel):
    def __init__(self):
        super(AWGNChannel, self).__init__()
        self.fadingComponents = 64
        self.fadingSamples = None

    def generateRayleighFading(self, maxDoppler, samples):
        self.fadingSamples = zeros(samples)
        for t in range(samples):
            am = uniform(0, 2*pi, self.fadingComponents)
            bm = uniform(0, 2*pi, self.fadingComponents)
            alpham = uniform(0, 2*pi, self.fadingComponents)

            ri = (1/sqrt(self.fadingComponents))*sum(cos(2*pi*maxDoppler*cos(alpham)*t + am))
            rq = (1/sqrt(self.fadingComponents))*sum(sin(2*pi*maxDoppler*cos(alpham)*t + bm))

            rt = ri + 1j*rq

            self.fadingSamples[t] = 10*log10(abs(rt)**2)
        #print(self.fadingSamples)

    def pathLossCalc(self, reference, exponent, distance, **params):
        if self.switchShadowing:
            shadowingStdev = params.get('shadowingStdev',5.8)
            shadowingSample = normal(loc=0, scale=shadowingStdev)
            
        else:
            shadowingSample = 0

        if self.switchFading:
            tsample = params.get('fadingSample', 0)
            fading = self.fadingSamples[tsample]
        else:
            fading = 0


        return reference + 10*exponent*log10(distance) + shadowingSample + fading

class Antenna:
    def __init__ (self, **params):
        self.nElementsX = params.get('xelements',0)
        self.nElementsY = params.get('yelements',0)

    def gain(self, **params):
        raise NotImplementedError

class UniformPlanarArray(Antenna):
    def __init__ (self):
        super(UniformPlanarArray, self).__init__()

    def gain(self, **params):
        pass

class UniformLinearArray(Antenna):
    def __init__ (self):
        super(UniformLinearArray, self).__init__()


class DummyAntenna(Antenna):
    def __init__ (self, gain):
        super(DummyAntenna, self).__init__()
        self._gain = gain

    def gain(self, **params):
        return self._gain

class ConePlusCircle(Antenna):
    def __init__ (self, gain, apperture):
        super(ConePlusCircle, self).__init__()
        self._gain = gain
        self.apperture = apperture

    def gain(self, **params):
        # angle which the is user is positioned related to antenna
        angle = params.get('angle')

        # Angle that the antenna is pointed at
        direction = params.get('direction')

        if (angle <= direction + self.apperture/2) or (angle >= direction + self.apperture/2):
            return self._gain*(2*pi/self.apperture)

        else:
            return self._gain*(2*pi/ (2*pi - self.apperture))


class Packet:
    def __init__(self, source, dest, pId : int, arrv : int, payloadLen : int):
        self.arrival = arrv
        self.payloadLen = payloadLen
        self.timetolive = 80
        self.source = source
        self.dest = dest
        self.packetId = pId
        self.flowId = 0

    def acknowledge(self, nextpkt):
        raise NotImplementedError


class DataPacket(Packet):
    def __init__(self, source, dest, pId : int, arrv : int, 
                payloadLen : int, data = None):

        super(DataPacket, self).__init__(source, dest, pId, arrv, payloadLen)
        self.data = data

    def acknowledge(self, nextpkt):
        pass

class ACK(Packet):
    def __init__(self, source, dest, pId : int, arrv : int, nextPkt : int):
        self.payloadLen = 512
        self.nextPkt = nextPkt
        super(DataPacket, self).__init__(source, dest, pId, arrv, self.payloadLen)

    def acknowledge(self, nextpkt):
        pass


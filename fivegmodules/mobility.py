
__all__ = ["MobilityModel", "RandomWalk", "StraightRoute", "Trace"]

class MobilityModel:
    def __init__(self):
        self.kmph = True
        self.mps = False

        self.timestep = 1e-3

    def move(self, device):
        raise NotImplementedError

    def calcDist(self, device):
        raise NotImplementedError

class RandomWalk(MobilityModel):
    def __init__ (self):
        super(StraightRoute, self).__init__()

    def move(self, device):
        pass

    def calcDist(self, device):
        pass

class StraightRoute(MobilityModel):
    def __init__ (self):
        super(StraightRoute, self).__init__()

    def calcDist(self, device):
        ### assumes the speed in km/h and the time in milliseconds
        if self.kmph:
            x = (device.Vx/3.6)*self.timestep
            y = (device.Vy/3.6)*self.timestep

        else:
            x = (device.Vx)*self.timestep
            y = (device.Vy)*self.timestep

        return x, y

    def move(self, device):
        while True:
            yield device.env.timeout(1) #updates at the minimal time cell
            newx, newy = self.calcDist(device)
            device.x += newx
            device.y += newy


class Trace(MobilityModel):
    def __init__ (self):
        super(StraightRoute, self).__init__()

    def move(self, device):
        pass

    def calcDist(self, device):
        pass

from matplotlib import pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def plot(self):
        raise NotImplementedError

class PlotKpi(Plotter):
    def __init__(self, device):
        self.device = device
        self.metric = []

    def collectKpi(self, value):
        raise NotImplementedError

    def plot(self):
        pass


class PlotRSRP(PlotKpi):
    def __init__(self, device):
        super(PlotRSRP, self).__init__(device)

    def collectKpi(self, value = None):
        self.metric.append(value)

    def plot(self):
        plt.plot(self.metric)
        plt.grid()
        plt.ylabel('RSRP [dBm]')
        plt.xlabel('Sample')
        plt.show()

class PlotSINR(PlotKpi):
    def __init__(self, device):
        super(PlotSINR, self).__init__(device)

    def collectKpi(self, value = None):
        self.metric.append(self.device.servingBSSINR())

    def plot(self):
        plt.plot(self.metric)
        plt.grid()
        plt.ylabel('SINR [dB]')
        plt.xlabel('Sample')
        plt.show()


class PlotNetwork(Plotter):
    def __init__(self, baseStations : dict, userEquipments : dict):
        self.basestations = baseStations
        self.userequipments = userEquipments

    def plotBaseStations(self):
        for m, bs in enumerate(self.basestations.values()):
            plt.scatter(bs['position']['x'], bs['position']['y']+20, marker = '^')
            plt.text(bs['position']['x']+10, bs['position']['y']+10, 'BS '+str(m)+'\n'+str(bs['txPower'])+'dBm')

    def plotUserEquipments(self):
        for n, ue in enumerate(self.userequipments.values()):
            plt.scatter(ue['position']['x'], ue['position']['y']+20, color=colors[n+1], marker='s')
            plt.text(ue['position']['x']-40, ue['position']['y']+10, str(ue['speed']['y'])+'Km/h')

    def plotBSRange(self, radius):
        pass

    def plotUERoute(self):
        pass

    def plotUEAssociation(self, association):
        pass

    def plot(self):
        self.plotBaseStations()
        self.plotUserEquipments()
        plt.show()

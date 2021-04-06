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
        self.device.kpi.averageRsrp.append(self.metric[-1])

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
        self.device.kpi.averageSinr.append(self.metric[-1])

    def plot(self):
        plt.plot(self.metric)
        plt.grid()
        plt.ylabel('SINR [dB]')
        plt.xlabel('Sample')
        plt.show()


class PlotNetwork(Plotter):
    def __init__(self, baseStations : dict, userEquipments : dict):
        super(PlotNetwork, self).__init__()
        self.basestations = baseStations
        self.userequipments = userEquipments

    def plotBaseStations(self):
        for m, bs in enumerate(self.basestations.values()):
            plt.scatter(bs.x, bs.y+20, marker = '^')
            plt.text(bs.x+10, bs.y+10, 'BS '+str(m)+'\n'+str(bs.txPower)+'dBm')

    def plotUserEquipments(self):
        for n, ue in enumerate(self.userequipments.values()):
            plt.scatter(ue.x, ue.y+20, marker='s')
            plt.text(ue.x-40, ue.y+10, str(ue.Vy)+'Km/h')

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

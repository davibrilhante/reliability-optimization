import numpy as np
from json import load
import argparse
import sys
from decompressor import decompressor
from matplotlib import pyplot as plt

from optimization import todb, tolin, load_inputFile, calc_recv



parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputFile', help='Instance json input file')

args = parser.parse_args()



if __name__ == "__main__":
    scenario, channel, LOS, network, nodes, R = load_inputFile(args.inputFile)

    m_bs = len(network)
    n_ue = len(nodes)

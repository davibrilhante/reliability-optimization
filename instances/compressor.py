#! /usr/bin/env python3
# -*- coding : utf-8 -*-

from json import load, dump

'''
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-i', '--input', help='input file')
parser.add_argument('-o', '--output', help='input file')

args = parser.parse_args()
'''

def compressor(data : dict) -> dict:
    for bs, bsblock in enumerate(data['blockage']):
        for ue, block in enumerate(bsblock):
            temp = []
            for t, los in enumerate(block):
                if (t == 0 and los == 1):
                    temp.append(0)
                elif (los == 1 and block[t-1] == 0):
                    temp.append(t)
                elif (los == 0 and block[t-1] == 1 and t > 0):
                    temp.append(t-1)
       
            data['blockage'][bs][ue] = temp
       
    return data

    


if __name__ == "__main__":
    try:
        with open(args.input, 'r') as jsonfile:
            data = load(jsonfile)
            jsonfile.close()

    except Exception as e:
        print(e)
        exit()

    compressor(data)

    try:    
        with open(args.output, 'w') as jsonfile:
            dump(data, jsonfile, indent=4)
            jsonfile.close()

    except Exception as e:
        print(e)
        exit()

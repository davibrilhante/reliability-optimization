#! /usr/bin/env python3
# -*- coding : utf-8 -*-

from json import load, dump
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-i', '--input', help='input file')
parser.add_argument('-o', '--output', help='input file')

args = parser.parse_args()

    
def decompressor(data : dict) -> dict:
    for bs, bsblock in enumerate(data['blockage']):
        for ue, block in enumerate(bsblock):
            temp = []
            for t, time in enumerate(block):
                if t % 2 == 1:
                    start = block[t-1]
                    stop = time+1

                    for i in range(start, stop):
                        temp.append(1)
                else:
                    if t == 0:
                        start = 0
                        stop = time
                        
                    else:
                        start = block[t-1]+1
                        stop = time

                    for i in range(start,stop):
                        temp.append(0)

            if len(block) % 2 == 1:
                final = block[-1]
            else:
                final = block[-1]+1
     
            for i in range(final, data['scenario']['simTime']):
                temp.append(1 - temp[final-1])

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

    decompressor(data)

    try:    
        with open(args.output, 'w') as jsonfile:
            dump(data, jsonfile, indent=4)
            jsonfile.close()

    except Exception as e:
        print(e)
        exit()

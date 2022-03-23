#! /usr/bin/env python3
# -*- coding : utf-8 -*-

from json import load, dump
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()

    parser.add_argument('-i', '--input', help='input file')
    parser.add_argument('-o', '--output', help='input file')

    args = parser.parse_args()
    return args


def decompressor(data : dict) -> dict:
    for bs, bsblock in enumerate(data['blockage']):
        for ue, block in enumerate(bsblock):
            temp = []

            if not block:
                data['blockage'][bs][ue] = [0 for i in range(data['scenario']['simTime'])]
                #data['blockage'][bs][ue] = [1 for i in range(data['scenario']['simTime'])]

            else:
                for t, time in enumerate(block):
                    # The odd t's are the endings of the LOS periods
                    # Thus, the beginnings are the even t's (or the previous t of an odd t)
                    if t % 2 == 1:
                        start = block[t-1]
                        stop = time+1

                        for i in range(start, stop):
                            temp.append(1)
                    else:
                        #If it is the first t, the NLOS period starts at 0 and ends
                        #in t. If the time is 0, then there is no blockage at the beginning
                        if t == 0:
                            start = 0
                            stop = time
                        
                        # The even t's determine the beginning of a LOS period
                        # so the NLOS started previus t time+1 and lasts untill time-1
                        else:
                            start = block[t-1]+1
                            stop = time

                        for i in range(start,stop):
                            temp.append(0)

                #At the end, if the last one is a LOS beginning, so untill the end
                # of the simulation there will be a LOS period
                if len(block) % 2 == 1:
                    final = block[-1]

                #Otherwise, if it is a LOS ending, then untill the end of simulation
                # there will be a NLOS period
                else:
                    final = block[-1]+1
         
                for i in range(final, data['scenario']['simTime']):
                    temp.append(1 - temp[final-1])

                data['blockage'][bs][ue] = temp
     
    return data


if __name__ == "__main__":
    args = get_args()

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

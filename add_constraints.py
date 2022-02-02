#!/usr/bin/env python3
# -*- coding : utf8 -*-
                      
                      
import numpy as np    
import gurobipy as gb 
from gurobipy import GRB
import logging

def gen_constraint_1(x, m_bs, nodes, resources, SNR,simTime, gen_dict):
    n_ue = len(nodes)
    generator = (nodes[n]['capacity']*x[m,n,t] <=
                        resources[m]*np.log2(1+SNR[m][n][t])
                for n in range(n_ue)
                    for m in range(m_bs)
                        for t in nodes[n][m,'available'])

    gen_dict['constrs_1'] = generator


def gen_constraint_2(x, y, m_bs, nodes, simTime, gen_dict):
    n_ue = len(nodes)

    
    generator = (y[m,n,t] <= x[m,n,t] 
                for n in range(n_ue)
                    for m in range(m_bs)
                        #for t in range(simTime))
                        for arrival in nodes[n]['packets']
                            for t in range(arrival,min(arrival+nodes[n]['delay'], simTime))
                                if t in nodes[n][m,'available'])

    gen_dict['constrs_2'] = generator


def gen_constraint_3(x, m_bs, nodes, SNR, simTime, gen_dict):
    n_ue = len(nodes)

    generator = (gb.quicksum((SNR[m][n][t] - nodes[n]['threshold'])*x[m,n,t] for m in range(m_bs))>=0
                for t in range(simTime)
                    for n in range(n_ue))

    gen_dict['constrs_3'] = generator

def gen_auxiliar_4(sumbeta, beta, m_bs, n_ue, simTime, gen_dict):
    M =  {i for i in range(m_bs)}

    generator = (sumbeta[p,n,t] == gb.quicksum(beta[p][q][n][t] for q in M - {p})
                    for t in range(simTime)
                        for n in range(n_ue)
                            for p in M)

    gen_dict['auxiliar_4'] = generator

def gen_constraint_4(x, beta, m_bs, n_ue, simTime, gen_dict, interval=1):
    M =  {i for i in range(m_bs)}

    betap = {}
    betaq = {}
    for p in M:
        betap[p] = {}
        betaq[p] = {}
        for n in range(n_ue):
            betap[p][n] = {}
            betaq[p][n] = {}
            for t in range(simTime):
                betap[p][n][t] = 1
                betaq[p][n][t] = 0
                for q in M - {p}:
                    if beta[p][q][n][t] == 1:
                        betap[p][n][t] = 0

                    if beta[q][p][n][t] == 1:
                        betaq[p][n][t] = 1


    generator = (x[p,n,t] >= x[p,n,t-1]*betap[p][n][t] #(1 - gb.quicksum(beta[p][q][n][t] for q in M - {p}))
                for t in range(1, simTime)
                    for n in range(n_ue)
                        for p in M)

    gen_dict['constrs_4'] = generator

def gen_constraint_5(y, m_bs, nodes, simTime, gen_dict):
    n_ue = len(nodes)

    '''
    generator = (sum(sum(y[m,n,t] for t in range(simTime))
                for m in range(m_bs)) <= nodes[n]['nPackets']
                    for n in range(n_ue))
    '''

    intervals = {}
    for n, ue in enumerate(nodes):
        intervals[n] = []
        for arrival in ue['packets']:
            for t in range(arrival, arrival+ue['delay']):
                intervals[n].append(t)

    generator = (gb.quicksum(gb.quicksum(y[m,n,t] for t in intervals[n]) for m in range(m_bs)) <=
                    nodes[n]['nPackets'] for n in range(n_ue))

    gen_dict['constrs_5'] = generator

def gen_constraint_6(y, m_bs, nodes,  gen_dict):
    n_ue = len(nodes)

    '''
    generator = (sum(sum(y[m,n,t] for t in range(arrival, arrival + nodes[n]['delay']))
                for m in range(m_bs)) <= 1
                    for n in range(n_ue)
                        for arrival in nodes[n]['packets'])
    
    '''
     
    generator = (gb.quicksum(gb.quicksum(y[m,n,t] for t in range(arrival, arrival + nodes[n]['delay'])) for m in range(m_bs)) <= 1
                    for n in range(n_ue)
                        for arrival in nodes[n]['packets'])

    gen_dict['constrs_6'] = generator
    

def gen_constraint_7(y, m_bs, nodes, gen_dict):
    n_ue = len(nodes)

    generator = (sum(sum(y[m,n,t] for t in range(nodes[n]['packets'][0])) for m in range(m_bs)) == 0
                for n in range(n_ue))

    gen_dict['constrs_7'] = generator
    

def gen_constraint_8(y, m_bs, nodes, simTime, gen_dict):
    n_ue = len(nodes)

    generator = (sum(sum(y[m,n,t] for t in range(nodes[n]['packets'][-1]+nodes[n]['delay'],simTime)) 
                for m in range(m_bs)) == 0
                    for n in range(n_ue))

    gen_dict['constrs_8'] = generator

def gen_constraint_9(y, m_bs, nodes, gen_dict):
    n_ue = len(nodes)

    generator = (sum(sum(y[m,n,t] for t in range(nodes[n]['packets'][p]+nodes[n]['delay'], nodes[n]['packets'][p+1])) 
                for m in range(m_bs)) == 0
                    for n in range(n_ue)
                        for p in range(nodes[n]['nPackets']-1))

    gen_dict['constrs_9'] = generator

def gen_constraint_10(Var, m_bs, nodes, simTime, gen_dict, Key='constrs_10'):
    n_ue = len(nodes)
    valid_bs = {}
    for n in range(n_ue):
        for t in range(simTime):
            valid_bs[t] = set()
            for m in range(m_bs):
                if t in nodes[n][m,'available']:
                    valid_bs[t].add(m)
        
    generator = (sum(Var[m,n,t] for m in valid_bs[t]) <= 1 
                for t in range(simTime)
                    for n in range(n_ue))

    gen_dict[Key] = generator

def add_all_constraints(model, Vars, nodes, network, SNR, beta, R, scenario, interval=1):
    m_bs = len(network) 
    n_ue = len(nodes)
    M = {i for i in range(m_bs)}

    x = Vars[0]
    y = Vars[1]
    u = Vars[2]
    constrs_dict = {}


    gen_constraint_1(x, m_bs, nodes, R, SNR, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_1'], name='capcity_constr')
        logging.info('Constraints #1 added...')

    except Exception as error:
        logging.warning('E301: Error adding constraints #1')
        logging.warning(error)
        exit()


    gen_constraint_2(x, y, m_bs, nodes, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_2'], name='delay_constr')
        logging.info('Constraints #2 added...')

    except Exception as error:
        logging.warning('E302: Error adding constraints #2')
        logging.warning(error)
        exit()


    '''
    gen_constraint_3(x, m_bs, nodes, SNR, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_3'], name='snr_constr')
        print('Constraints #3 added...')

    except Exception as error:
        print('Error adding constraints #3')
        print(error)
        exit()


    try:
        aux = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.INTEGER, ub=scenario['ttt'], name ='aux')
        generator = (aux[p,n,t] == scenario['ttt'] - gb.quicksum(x[p,n,k] for k in range(t - scenario['ttt'], t))
                        for t in range(scenario['ttt'], scenario['simTime'])
                            for n in range(n_ue)
                                for p in M)

        model.addConstrs(generator, name='aux_const')


        generator = (u[p,n,t] == gb.min_(1, aux[p,n,t])
                        for t in range(scenario['ttt'], scenario['simTime'])
                            for n in range(n_ue)
                                for p in M)

        model.addConstrs(generator, name='u_const')

        print('u_const added')

    except Exception as error:
        print('u_const not added to model')
        print(error)
        exit()
    '''

    betap = {}
    betaq = {}
    for p in M:
        betap[p] = {}
        betaq[p] = {}
        for n in range(n_ue):
            betap[p][n] = {}
            betaq[p][n] = {}
            for t in range(scenario['simTime']):
                betap[p][n][t] = 1
                betaq[p][n][t] = 0
                for q in M - {p}:
                    if beta[p][q][n][t] == 1:
                        betap[p][n][t] = 0
                        break
                for q in M - {p}:
                    if beta[q][p][n][t] == 1:
                        betaq[p][n][t] = 1


    aux = {}
    for p in M:
        for n in range(n_ue):
            if 0 in nodes[n][p, 'available']:
                model.addConstr(x[p,n,0] >= 0, 'initial_constr[{ind[0]},0,0]'.format(ind=[p]))

            ho_range = {k for k in range(scenario['simTime'])}
            for t in range(1,scenario['simTime']):
                if (t in nodes[n][p,'available']) and (t-1 in nodes[n][p,'available']):
                    if betap[p][n][t] == 1:
                        try:
                            # If it is not going to receive a handover, 
                            # then it should stay at the same state
                            if betaq[p][n][t] == 0:
                                model.addConstr(x[p,n,t] - x[p,n,t-1] == 0, 'stay_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[p,n,t]))

                            # Otherwise, if it is a candidate to receive a handover
                            # Then it may or not change the state
                            else:
                                model.addConstr(x[p,n,t] - x[p,n,t-1] >= 0, 'stay_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[p,n,t]))
                        except Exception as error:
                            logging.warning('E303: ', error)


                    else:
                        candidates = []
                        for q in M - {p}:
                            if (beta[p][q][n][t] == 1) and (t in nodes[n][q,'available']):
                                candidates.append(q)
                        try:
                            aux[p,n,t] = model.addVar(vtype=GRB.INTEGER, ub=scenario['ttt'],
                                    name='aux[{bs},{ue},{tempo}]'.format(bs=p,ue=n,tempo=t))

                            extra = 0
                            valid = set()

                            for k in range(t - scenario['ttt'], t):
                                if k in nodes[n][p,'available']:
                                    valid.add(k)
                                else:
                                    extra += 1

                            # if valid_range:
                            #if ho_range.issubset(nodes[n][p,'available']):
                            model.addConstr(aux[p,n,t] == scenario['ttt'] - extra - gb.quicksum(x[p,n,k] for k in valid),
                                            'aux_constr[{bs},{ue},{tempo}]'.format(bs=p,ue=n,tempo=t))
                            #else:
                            #    model.addConstr(aux[p,n,t] == scenario['ttt'], 'aux_constr[{bs},{ue},{tempo}]'.format(bs=p,ue=n,tempo=t))

                        except Exception as error:
                            logging.warning('E304: ', error)

                        try:
                            model.addGenConstrMin(u[p,n,t],[1, aux[p,n,t]], name='u_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[p,n,t]))

                        except Exception as error:
                            logging.warnning('E307: ', error)

                        try:
                            model.addGenConstrIndicator(
                                    u[p,n,t], False, gb.quicksum(x[q,n,t] for q in candidates + [p]), 
                                    GRB.EQUAL, 1, 'handover_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[p,n,t]))

                            '''
                            model.addGenConstrIndicator(
                                    u[p,n,t], False, x[p,n,t] - x[p,n,t-1], 
                                    GRB.GREATER_EQUAL, 0, 'stay_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[p,n,t]))

                            for q in candidates:
                                model.addGenConstrIndicator(
                                        u[p,n,t], False, x[q,n,t] - x[q,n,t-1], 
                                        GRB.GREATER_EQUAL, 0, 'target_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[q,n,t]))
                            '''

                        except Exception as error:
                            logging.warning('E308: ', error)

                        try:
                            model.addGenConstrIndicator(
                                    u[p,n,t], True, x[p,n,t] - x[p,n,t-1], 
                                    GRB.EQUAL, 0, 'stay_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[p,n,t]))

                            '''
                            for q in candidates:
                                model.addGenConstrIndicator(
                                        u[p,n,t], True, x[q,n,t] - x[q,n,t-1], 
                                        GRB.EQUAL, 0, 'target_constr[{ind[0]},{ind[1]},{ind[2]}]'.format(ind=[q,n,t]))
                            '''

                        except Exception as error:
                            logging.warning('E309: ', error)

                            ho_range.remove(t - scenario['ttt'])
                            ho_range.add(t)

        



    gen_constraint_5(y, m_bs, nodes, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_5'], name='max_pkt_constr')
        logging.info('Constraints #5 added...')

    except Exception as error:
        logging.warning('E305: Error adding constraints #5')
        logging.warning(error)
        exit()


    gen_constraint_6(y, m_bs, nodes, constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_6'], name='delay_pkt_constr')
        logging.info('constraints #6 added...')

    except Exception as error:
        logging.warning('E306: error adding constraints #6')
        logging.warning(error)
        exit()

    '''
    gen_constraint_7(y, m_bs, nodes, constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_7'], name='delay_start_constr')
        print('constraints #7 added...')

    except Exception as error:
        print('error adding constraints #7')
        print(error)
        exit()

    gen_constraint_8(y, m_bs, nodes, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_8'], name='delay_end_constr')
        print('Constraints #8 added...')

    except Exception as error:
        print('Error adding constraints #8')
        print(error)
        exit()

    gen_constraint_9(y, m_bs, nodes, constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_9'], name='post_delay_constr')
        print('constraints #9 added...')

    except Exception as error:
        print('error adding constraints #9')
        print(error)
        exit()
    '''

    gen_constraint_10(x, m_bs, nodes, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_10'], name='x_constr')
        logging.info('Constraints #10 added...')

    except Exception as error:
        logging.warning('E310: Error adding constraints #10')
        loggung.warning(error)
        exit()

    '''
    gen_constraint_10(y, m_bs, n_ue, scenario['simTime'], constrs_dict, 'constrs_11')

    try:
        model.addConstrs(constrs_dict['constrs_11'], name='y_constr')
        print('Constraints #11 added...')

    except Exception as error:
        print('Error adding constraints #11')
        print(error)
        exit()
    '''

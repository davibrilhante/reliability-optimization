#!/usr/bin/env python3
# -*- coding : utf8 -*-
                      
                      
import numpy as np    
import gurobipy as gb 
from gurobipy import GRB

def gen_constraint_1(x, m_bs, nodes, resources, SNR,simTime, gen_dict):
    n_ue = len(nodes)
    generator = (nodes[n]['capacity']*x[m,n,t] <=
                        resources[m]*np.log2(1+SNR[m][n][t])
                for n in range(n_ue)
                    for m in range(m_bs)
                        for t in range(simTime))

    gen_dict['constrs_1'] = generator


def gen_constraint_2(x, y, m_bs, nodes, simTime, gen_dict):
    n_ue = len(nodes)

    '''
    generator = (sum(x[m,n,k] for k in range(arrival,arrival+nodes[n]['delay']))
                        == sum(y[m,n,k] for k in range(arrival,arrival+nodes[n]['delay']))
                for n in range(n_ue)
                    for m in range(m_bs)
                        for arrival in nodes[n]['packets'])
    '''
    generator = (y[m,n,t] <= x[m,n,t] 
                for n in range(n_ue)
                    for m in range(m_bs)
                        for t in range(simTime))
                        #for arrival in nodes[n]['packets']
                        #    for t in range(arrival,arrival+nodes[n]['delay']))

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

    generator = (sum(sum(y[m,n,t] for t in range(simTime))
                for m in range(m_bs)) <= nodes[n]['nPackets']
                    for n in range(n_ue))

    gen_dict['constrs_5'] = generator

def gen_constraint_6(y, m_bs, nodes,  gen_dict):
    n_ue = len(nodes)

    generator = (sum(sum(y[m,n,t] for t in range(arrival, arrival + nodes[n]['delay']))
                for m in range(m_bs)) <= 1
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

def gen_constraint_10(Var, m_bs, n_ue, simTime, gen_dict, Key='constrs_10'):
    generator = (sum(Var[m,n,t] for m in range(m_bs)) <= 1 
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
        print('Constraints #1 added...')

    except Exception as error:
        print('Error adding constraints #1')
        print(error)
        exit()


    gen_constraint_2(x, y, m_bs, nodes, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_2'], name='delay_constr')
        print('Constraints #2 added...')

    except Exception as error:
        print('Error adding constraints #2')
        print(error)
        exit()


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



    sumu = model.addVars(m_bs, m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='sum_u')
    sumx = model.addVars(m_bs, m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='sum_x')

    #generator = (sumu[q,n,t] == gb.quicksum((1 - u[p,n,t])*beta[p][q][n][t]  for p in M - {q})
    generator = (sumu[p, q,n,t] == (1 - u[p,n,t])*beta[p][q][n][t]  
                    #for t in range(scenario['ttt'], scenario['simTime'])
                    for t in range(scenario['simTime'])
                        for n in range(n_ue)
                            for q in M
                                for p in M - {q})

    model.addConstrs(generator, name='sum_u_const')

    
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
                        print('betap[',p,',',t,'] = 0')
                        break
                for q in M - {p}:
                    if beta[q][p][n][t] == 1:
                        print('betaq[',p,',',t,'] = 1')
                        betaq[p][n][t] = 1

    model.addConstr(x[0,0,0] == 1, 'initial_constr')

    for p in M:
        for n in range(n_ue):
            for t in range(1,scenario['simTime']):
                if betap[p][n][t] == 1:
                    if betaq[p][n][t] == 0:
                        model.addConstr(x[p,n,t] - x[p,n,t-1] == 0, 'stay_constr')
                    else:
                        model.addConstr(x[p,n,t] - x[p,n,t-1] >= 0, 'stay_constr')
                        #model.addGenConstrIndicator(x[p,n,t-1], True, x[p,n,t] - x[p,n,t-1], GRB.EQUAL, 0, 'move_constr')
                else:
                    candidates = []
                    for q in M - {p}:
                        if beta[p][q][n][t] == 1:
                            print(p, q, t, 'handover_constr')
                            candidates.append(q)
                    try:
                        #model.addConstr((sumu[p,q,n,t] == 1) >> (x[q,n,t] == 1), 'move_constr')
                        model.addGenConstrIndicator(u[p,n,t], False, gb.quicksum(x[q,n,t] for q in candidates), GRB.EQUAL, 1, 'handover_constr')
                        #model.addGenConstrIndicator(u[p,n,t], True, gb.quicksum(x[q,n,t] for q in candidates), GRB.EQUAL, 0, 'handover_constr')
                        model.addGenConstrIndicator(u[p,n,t], True, x[p,n,t] - x[p,n,t-1], GRB.EQUAL, 0, 'handover_constr')
                    except Exception as error:
                        print(error)
        

    '''

    parc1 = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='parc1')
    generator = (parc1[p,n,t] == x[p,n,t-1]*betap[p][n][t] 
                for t in range(1,scenario['simTime'])
                    for n in range(n_ue)
                        for p in M)

    model.addConstrs(generator, name='parc1_const')

    #generator = (x[q,n,t] == gb.or_(parc1[q,n,t],sumu[q,n,t])
    #generator = ((sumu[p,q,n,t] == 1) >> (x[q,n,t] == 1)
    generator = (x[q,n,t] >= parc1[q,n,t] + sumu[p,q,n,t]
                    for t in range(1,scenario['simTime'])
                        for n in range(n_ue)
                            for q in M
                                for p in M - {q})

    model.addConstrs(generator, name='test_constr')

    generator = (sumx[p,q,n,t] == gb.and_(x[p,n,t-1], x[q,n,t])
                    #for t in range(scenario['ttt'], scenario['simTime'])
                    for t in range(scenario['ttt'],scenario['simTime'])
                        for n in range(n_ue)
                            for q in M
                                for p in M - {q})

    model.addConstrs(generator, name='sumx_constr')

    generator = ((sumx[p,q,n,t] == 1) >> (sumu[p,q,n,t] == 1)
                    #for t in range(scenario['ttt'], scenario['simTime'])
                    for t in range(scenario['ttt'],scenario['simTime'])
                        for n in range(n_ue)
                            for q in M
                                for p in M - {q})

    model.addConstrs(generator, name='handover_constr')
    '''
    #gen_constraint_4(x, beta, m_bs, n_ue, scenario['simTime'], constrs_dict)

    try:
        #model.addConstrs(constrs_dict['constrs_4'], name='handover_constr')
        print('Constraints #4 added...')  

    except Exception as error:
        print('Error adding constraints #4')
        print(error)
        exit()


    gen_constraint_5(y, m_bs, nodes, scenario['simTime'], constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_5'], name='max_pkt_constr')
        print('Constraints #5 added...')

    except Exception as error:
        print('Error adding constraints #5')
        print(error)
        exit()


    gen_constraint_6(y, m_bs, nodes, constrs_dict)

    try:
        model.addConstrs(constrs_dict['constrs_6'], name='delay_pkt_constr')
        print('constraints #6 added...')

    except Exception as error:
        print('error adding constraints #6')
        print(error)
        exit()

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

    gen_constraint_10(x, m_bs, n_ue, scenario['simTime'], constrs_dict, 'constrs_10')

    try:
        model.addConstrs(constrs_dict['constrs_10'], name='x_constr')
        print('Constraints #10 added...')

    except Exception as error:
        print('Error adding constraints #10')
        print(error)
        exit()

    gen_constraint_10(y, m_bs, n_ue, scenario['simTime'], constrs_dict, 'constrs_11')

    try:
        model.addConstrs(constrs_dict['constrs_11'], name='y_constr')
        print('Constraints #11 added...')

    except Exception as error:
        print('Error adding constraints #11')
        print(error)
        exit()

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

    generator = (sum((SNR[m][n][t] - nodes[n]['threshold'])*x[m,n,t] for m in range(m_bs))>=0
                for t in range(simTime)
                    for n in range(n_ue))

    gen_dict['constrs_3'] = generator

def gen_auxiliar_4(B, beta, m_bs, n_ue, simTime, gen_dict):
    bs_pairs = []
    for i in range(m_bs):
        for j in range(m_bs):
            if i != j:
                bs_pairs.append([i,j])

    generator = (B[p,q,n,t] == beta[p][q][n][t]
                    for t in range(simTime)
                        for n in range(n_ue)
                            for p, q in bs_pairs)

    gen_dict['auxiliar_4'] = generator

def gen_constraint_4(x, z, B, u, m_bs, n_ue, beta, simTime, gen_dict, interval=1):
    bs_pairs = []
    for i in range(m_bs):
        for j in range(m_bs):
            if i != j:
                bs_pairs.append([i,j])

    
    generator = (z[p,q,n,t] == gb.and_(x[q,n,t],u[p,n,t],B[p,q,n,t])
    #generator = (z[p,q,n,t] == gb.and_(u[p,n,t],B[p,q,n,t])
                for t in range(interval,simTime)
                    for n in range(n_ue)
                        for p, q in bs_pairs )

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
    x = Vars[0]
    y = Vars[1]
    z = Vars[2]
    B = Vars[3]
    w = Vars[4]
    u = Vars[5]
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



    n_ue = len(nodes)
    gen_auxiliar_4(B, beta, m_bs, n_ue, scenario['simTime'], constrs_dict)
    gen_constraint_4(x, z, B, u, m_bs, n_ue, beta, scenario['simTime'], constrs_dict)
    
    try:
        model.addConstrs(constrs_dict['auxiliar_4'], name='handover_auxiliar')
        model.addConstrs(constrs_dict['constrs_4'], name='handover_constr')
        print('Constraints #4 added...')

    except Exception as error:
        print('Error adding constraints #4')
        print(error)
        exit()

    
    V = {i for i in range(m_bs)}

    generator = (w[p,n,t] == 1 - sum(beta[p][q][n][t+interval] for q in V - {p})
                    for t in range(scenario['simTime'] - interval)
                        for n in range(n_ue)
                            for p in V)

    model.addConstrs(generator, name='another_auxiliar')


    generator = (u[p,n,t] == gb.and_(x[p,n,k] for k in range(t - scenario['ttt'], t))
                    for t in range(scenario['ttt'], scenario['simTime'])
                        for n in range(n_ue)
                            for p in V)

    model.addConstrs(generator, name='yet_another_auxiliar')


    a = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='a')
    b = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='b')
    c = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='c')
    d = model.addVars(m_bs, n_ue, scenario['simTime'], vtype=GRB.BINARY, name='d')

    generator = (c[p,n,t] == sum(z[q,p,n,t+interval] for q in V - {p})
                    for t in range(scenario['simTime'] - interval)
                        for n in range(n_ue)
                            for p in V)

    model.addConstrs(generator, name='yet_another_auxiliar')


    generator = (a[p,n,t] == gb.and_(x[p,n,t],w[p,n,t])
                for t in range(scenario['simTime'] - interval)
                    for n in range(n_ue)
                        for p in V)

    model.addConstrs(generator, name='yet_another_auxiliar')
    

    generator = (d[p,n,t] == 1 - x[p,n,t]
                    for t in range(scenario['simTime'])
                        for n in range(n_ue)
                            for p in V)

    model.addConstrs(generator, name='negation_constraint')


    generator = (b[p,n,t] == gb.and_(d[p,n,t],c[p,n,t])
                for t in range(scenario['simTime'] - interval)
                    for n in range(n_ue)
                        for p in V
                            for q in V - {p})

    model.addConstrs(generator, name='yet_another_auxiliar')


    generator = (x[p,n,t+interval] == gb.or_(a[p,n,t],b[p,n,t])
                for t in range(scenario['simTime'] - interval)
                    for n in range(n_ue)
                        for p in V)

    model.addConstrs(generator, name='yet_another_auxiliar_again')


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




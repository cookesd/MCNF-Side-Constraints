# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:02:43 2020

@author: dakar
"""

import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd

#%% Setup input data
edge_df = pd.DataFrame({'source':[1,1,2,2,3],
                          'target':[2,3,4,3,4],
                          'weight':[0,0,1,0,1],
                          'capacity':[1]*5,
                          'con_c':[0,0,0,1,0]})
mult_ind_edge_df = edge_df.set_index(['source','target'])
node_demands = {1:-2,2:0,3:0,4:2}
nodes = set(edge_df.source.unique()).union(set(edge_df.target.unique()))
edges = [(u,v) for u,v in zip(edge_df.source.values,edge_df.target.values)]

#%% Build Model

class SCModel(object):
    def __init__(self,edgelist_df,node_demand_dict):
        self.super_source = -1
        self.super_sink = 0
        self.edgelist_df = edgelist_df
        self.multi_ind_edgelist_df = edgelist_df.set_index(['source','target'])
        self.node_demand_dict = node_demand_dict
        self.edges = [(u,v) for u,v in zip(edgelist_df.source.values,
                                           edgelist_df.target.values)]
        self.nodes = set(edgelist_df.source.unique()).union(set(edgelist_df.target.unique()))
        self.base_supply_nodes_demands = {node: self.node_demand_dict[node]
                                     for node in self.nodes
                                     if self.node_demand_dict[node] < 0}
        self.base_demand_nodes_demands = {node: self.node_demand_dict[node]
                                     for node in self.nodes
                                     if self.node_demand_dict[node] > 0}
        self.sum_pos_demand = max([sum([-d for n,d in self.base_supply_nodes_demands.items()]),
                                   sum([d for n,d in self.base_demand_nodes_demands.items()])])
        self.aug_multi_ind_edgelist_df = self.make_aug_edgelist()
        self.aug_node_demand_dict = self.make_aug_node_demand_dict()
        self.aug_edges = list(self.aug_multi_ind_edgelist_df.index)
        self.aug_nodes = set(list(self.nodes) + [self.super_source,self.super_sink])
    
    def make_aug_node_demand_dict(self):
        nd = {node:0 for node in self.node_demand_dict.keys()}
        nd[self.super_source] = -self.sum_pos_demand
        nd[self.super_sink] = self.sum_pos_demand
        return(nd)
    
    def make_aug_edgelist(self):
        max_weight = self.edgelist_df.weight.max()
        new_edge_dicts = [
            # source to supply node edges
            [{'source':self.super_source,'target':node,
              'weight':0,'capacity':abs(demand)}
             for node,demand in self.base_supply_nodes_demands.items()] +
            # demand to sink nodes
            [{'source':node,'target':self.super_sink,
              'weight':0,'capacity':abs(demand)}
             for node,demand in self.base_demand_nodes_demands.items()] +
            # source to sink edge
            [{'source':self.super_source,'target':self.super_sink,
              'weight':max_weight * len(self.edgelist_df.index) * self.sum_pos_demand,
              'capacity':self.sum_pos_demand}]]
        
        new_edge_df = pd.DataFrame({i:edge_dict
                                    for i,edge_dict
                                    in enumerate(new_edge_dicts[0])}).transpose()
        aug_mi_edgelist_df = pd.concat([self.edgelist_df,new_edge_df],
                                     axis='index').set_index(['source','target'])
        return(aug_mi_edgelist_df)
        
    def _make_model(self,nodes,edges,multi_ind_edgelist_df,node_demand_dict):
        model = pyo.ConcreteModel()
        model.nodes = pyo.Set(nodes)
        
        def edge_caps(model,u,v):
            # return(0,1)
            return((0,multi_ind_edgelist_df.loc[(u,v),'capacity']))
        model.x_ij = pyo.Var(edges,bounds=edge_caps)
        
        def get_node_balance_lhs(model,node):
            lhs = (sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
            sum([model.x_ij[(u,v)] for u,v in edges if u == node]))
            return(lhs)
        
        def node_balance_rule(model,node):
            # return((sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
            # sum([model.x_ij[(u,v)] for u,v in edges if u == node])) <= node_demands[node])
            return(get_node_balance_lhs(model,node) <= node_demand_dict[node])
        
        model.node_balance_cons = pyo.Constraint(nodes,rule=node_balance_rule)
        
        return(model)
    
    
    def build_base_model(self):
        model = self._make_model(self.nodes,self.edges,
                                          self.multi_ind_edgelist_df,
                                          self.node_demand_dict)
        def obj_rule(model):
            z = sum([float(self.multi_ind_edgelist_df.loc[(u,v),:'weight']) * model.x_ij[(u,v)]
                     for u,v in edges])
            return(z)
        model.obj = pyo.Objective(rule=obj_rule,sense=1)
        self.base_model = model
    
    def build_aug_model(self):
        model = self._make_model(self.aug_nodes,self.aug_edges,
                                          self.aug_multi_ind_edgelist_df,
                                          self.aug_node_demand_dict)
        # model.alpha = pyo.Param()
        def obj_rule(model):
            z = sum([float(self.aug_multi_ind_edgelist_df.loc[(u,v),:'weight']) * model.x_ij[(u,v)]
                     for u,v in self.aug_edges])
            return(z)
        model.obj = pyo.Objective(rule=obj_rule,sense=1)
        self.aug_model = model
    
    
    
    def solve(self,model):
        solver = pyo.SolverFactory('glpk')
        res = solver.solve(model)


#%% test class
a = SCModel(edge_df,node_demands)
a.build_aug_model()
a.solve(a.aug_model)
a.aug_model.pprint()
#%% Example MCNF Model    
model = pyo.ConcreteModel()
model.nodes = pyo.Set(nodes)

def edge_caps(model,u,v):
    # return(0,1)
    return((0,mult_ind_edge_df.loc[(u,v),'capacity']))
model.x_ij = pyo.Var(edges,bounds=edge_caps)

def obj_rule(model):
    z = sum([float(mult_ind_edge_df.loc[(u,v),:'weight']) * model.x_ij[(u,v)]
             for u,v in edges])
    return(z)
model.obj = pyo.Objective(rule=obj_rule,sense=1)

def get_node_balance_lhs(model,node):
    lhs = (sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
    sum([model.x_ij[(u,v)] for u,v in edges if u == node]))
    print(lhs)
    return(lhs)

def node_balance_rule(model,node):
    return((sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
    sum([model.x_ij[(u,v)] for u,v in edges if u == node])) <= node_demands[node])
    # return(get_node_balance_lhs(model,node) <= node_demands[node])

model.node_balance_cons = pyo.Constraint(nodes,rule=node_balance_rule)


#%% Solve model
solver = pyo.SolverFactory('glpk')
res = solver.solve(model)
print('The obj value is {}'.format(pyo.value(model.obj)))
print('The flow is below:')
model.x_ij.pprint()

flow_s = pd.Series({(u,v):pyo.value(model.x_ij[(u,v)]) for u,v in edges},name='flow')
print(flow_s)

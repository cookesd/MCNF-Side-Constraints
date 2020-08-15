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
        
        # Should probably make 'update' methods for these
        # So that they can be updated when user adds/deletes nodes/edges
        self.edges = [(u,v) for u,v in zip(edgelist_df.source.values,
                                           edgelist_df.target.values)]
        self.nodes = set(edgelist_df.source.unique()).union(set(edgelist_df.target.unique()))
        
        # Get base graph data
        self.base_supply_nodes_demands = {node: self.node_demand_dict[node]
                                     for node in self.nodes
                                     if self.node_demand_dict[node] < 0}
        self.base_demand_nodes_demands = {node: self.node_demand_dict[node]
                                     for node in self.nodes
                                     if self.node_demand_dict[node] > 0}
        self.sum_pos_demand = max([sum([-d for n,d in self.base_supply_nodes_demands.items()]),
                                   sum([d for n,d in self.base_demand_nodes_demands.items()])])
        
        # Get augmenting graph data
        self.aug_multi_ind_edgelist_df = self.make_aug_edgelist()
        self.aug_node_demand_dict = self.make_aug_node_demand_dict()
        self.aug_edges = list(self.aug_multi_ind_edgelist_df.index)
        self.aug_nodes = set(list(self.nodes) + [self.super_source,self.super_sink])
    
    def make_aug_node_demand_dict(self):
        '''
        Make node demand dict for augmentation graph of the base graph
        
        Make all original nodes into transshipment nodes (demand = 0).
        Add a super source and super sink node that supplies and 
        receives (respectively) the sum of the total demand in the original
        graph.

        Returns
        -------
        nd : dict
            Node demand dict for the aumentation graph

        '''
        nd = {node:0 for node in self.node_demand_dict.keys()}
        nd[self.super_source] = -self.sum_pos_demand
        nd[self.super_sink] = self.sum_pos_demand
        return(nd)
    
    def make_aug_edgelist(self):
        '''
        Make edgelist dataframe for augmentation graph of the base graph
        
        Keep all edges from original graph and their initial attributes (weight/capacity).
        Add edge from super source to all source nodes from the base graph
        with zero cost (weight) and capacities equal to their initial supply.
        Add edge from each demand node from the base graph to super sink
        with zero cost (weight) and capacities equal to their initial demand.
        

        Returns
        -------
        aug_mi_edgelist_df : pd.DataFrame
            Edgelist dataframe for the aumentation graph.
            Columns ('source','target','weight','capacity')

        '''
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
        '''
        Make pyomo linear programming model to solve MCNF problem

        Parameters
        ----------
        nodes : set
            The set of nodes in the graph including super source and super sink.
        edges : list of tuples
            The (u,v) node pairs defining the edges for the graph to build.
        multi_ind_edgelist_df : pd.DataFrame
            DataFrame containing edge information (source, target, weight, capacity).
        node_demand_dict : dict
            The node, demand pairs for nodes in the graph.

        Returns
        -------
        model : pyomo.ConcreteModel
            Object containing the decision variables, constraint, and objective function
            for the MCNF problem for the graph.

        '''
        model = pyo.ConcreteModel()
        model.nodes = pyo.Set(nodes)
        
        def edge_caps(model,u,v):
            '''
            Set the flow capacity on the edge from u to v

            Parameters
            ----------
            model : pyo.Model
                The model this constraint is defined on.
            u : TYPE
                The source node of the edge.
            v : TYPE
                The target node of the edge.

            Returns
            -------
            bounds : tuple
                Lower bound and upper bound of the edge capacity.

            '''
            # return(0,1)
            bounds = (0,multi_ind_edgelist_df.loc[(u,v),'capacity'])
            return(bounds)
        model.x_ij = pyo.Var(edges,bounds=edge_caps)
        
        def get_node_balance_lhs(model,node):
            '''
            Calaculate the node balance (inflow - outflow) for the given node
            
            This function is separate from the constraint so we can access the
            value outside of the model
            
            Parameters
            ----------
            model : pyo.Model
                The model to get the flow from.
            node : TYPE
                The node to compute the node balance for.

            Returns
            -------
            lhs : numeric
                The inflow for the node in the given solution.

            '''
            # Can make this set to sum over smaller by filtering the edge_df
            # to rows where the source / target is 'node'
            lhs = (sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
            sum([model.x_ij[(u,v)] for u,v in edges if u == node]))
            return(lhs)
        
        def node_balance_rule(model,node):
            '''
            Computes whether or not the node meets the balance restrictions.
            
            Says whether or not the inflow - outflow <= node demand. We can
            use <= rather than == because the sum of demands must be zero, so
            if one node is < its demand, then another must be > its demand
            therefore that constraint will not be satisfied.

            Parameters
            ----------
            model : pyo.Model
                The model to add the constraint to.
            node : TYPE
                The node to check node balance for.

            Returns
            -------
            boolean True if constraint satisfied.

            '''
            # return((sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
            # sum([model.x_ij[(u,v)] for u,v in edges if u == node])) <= node_demands[node])
            return(get_node_balance_lhs(model,node) <= node_demand_dict[node])
        
        model.node_balance_cons = pyo.Constraint(nodes,rule=node_balance_rule)
        
        return(model)
    
    
    def build_base_model(self):
        '''
        Build pyomo LP model for MCNF on the base graph. Calls _make_model and
        adds the objective function to it. Sets model to the base_model attribute.

        Returns
        -------
        None.

        '''
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
        '''
        Build pyomo LP model for MCNF on the augmentation graph. Calls _make_model and
        adds the objective function to it. Sets model to the aug_model attribute.

        Returns
        -------
        None.

        '''
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
        '''
        Solve the provided model. The result will be stored in the model object.

        Parameters
        ----------
        model : pyo.Model
            The model we want to solve.

        Returns
        -------
        None.

        '''
        solver = pyo.SolverFactory('glpk')
        res = solver.solve(model)


#%% test class
a = SCModel(edge_df,node_demands)
a.build_aug_model()
a.solve(a.aug_model)
a.aug_model.pprint()

print('The obj value is {}'.format(pyo.value(a.aug_model.obj)))
print('The flow is below:')
a.aug_model.x_ij.pprint()

flow_s = pd.Series({(u,v):pyo.value(a.aug_model.x_ij[(u,v)]) for u,v in a.aug_edges},name='flow')
flow_s.index.names = ['source','target']
print(flow_s)
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

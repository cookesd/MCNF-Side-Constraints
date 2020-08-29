# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:02:43 2020

@author: dakar
"""

import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd


#%% Build Model

class SCModel(object):
    def __init__(self,edgelist_df,node_demand_dict,constraint_df = None,max_iterations=10):
        self.super_source = -1
        self.super_sink = 0
        self.edgelist_df = edgelist_df
        self.multi_ind_edgelist_df = edgelist_df.set_index(['source','target'])
        self.node_demand_dict = node_demand_dict
        self.constraint_df = constraint_df
        # set dual variables to 0 in first iteration for all constraints
        self.alpha_dict = {con:0 for con in self.constraint_df.constraint.unique()}
        self.alpha_df = pd.DataFrame(self.alpha_dict,index=[0])
        self.max_iterations = max_iterations
        
        # Should probably make 'add','update',and 'remove' methods for these
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
            # Make super source and super sink nodes to handle unequal supply/demand
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
    
    def lagrangian_obj_rule(self,model):
            z = (sum([float(self.aug_multi_ind_edgelist_df.loc[(u,v),:'weight']) * model.x_ij[(u,v)]
                     for u,v in self.aug_edges]) +
                 # get the dual adjusted portion of the objective to satisfy the constraints
                 # $\alpha_k * [(a_{i,j,k} *x_{i,j}) - b_k]$
            sum([self.alpha_dict[con] * sum([(self.constraint_df.loc[i,'coeff'] *
                                              model.x_ij[self.constraint_df.loc[i,'edge']]) + 
                                             self.constraint_df.loc[i,'rhs']
                                         for i in self.constraint_df[self.constraint_df['constraint']==con].index])
                 for con in self.constraint_df.constraint.unique()]))
            return(z)
    def update_aug_obj(self,model,verbose=False):
        '''Delete previous objective (if exist) and make new objective with updated values'''
        for obj in model.component_objects(pyo.Objective):
            model.del_component(obj)
        model.obj = pyo.Objective(rule=self.lagrangian_obj_rule,sense=pyo.minimize)
        if verbose:
            self.aug_model.obj.pprint()
        
        
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
        # Save this as the object's model (without an objective)
        self.aug_model = model
        # Add the objective to the model
        self.update_aug_obj(self.aug_model)
    
    
    def solve_mcnf_side_con(self):
        '''
        Solve the MCNF side constraint problem by iteratively updating dual variables
        associated with the constraints to ensure they're satisfied

        Returns
        -------
        None.

        '''
        
        self.solve(self.aug_model)
        gradient = self._get_gradient()
        epsilon = .05
        iteration = 1
        max_iterations = self.max_iterations
        while any([abs(partial) >= epsilon for partial in gradient.values()]):
            # some constraint is not satisfied so must change the dual variables
            print('Gradient for iteration {}:'.format(iteration))
            print(pd.Series(gradient))
            
            # determine step size of change
            gamma = self._get_step_size(gradient_dict=gradient)
            
            # set the new alpha values as the previous minus the gradient times the step size
            new_alphas = {key:val + gamma*gradient[key] for key,val in self.alpha_dict.items()}
            self.alpha_dict = new_alphas
            # add this new set of alphas to the dictionary for record
            self.alpha_df = pd.concat([self.alpha_df,
                                       pd.DataFrame(new_alphas,index=[0])],
                                      ignore_index=True)
            
            # Delete previous objective function and update with new alpha values
            self.update_aug_obj(self.aug_model)
            
            
            # Re-solve model with updated objective
            self.solve(self.aug_model)
            # Get new gradient values
            gradient = self._get_gradient()
            if iteration == max_iterations:
                print('Max iterations ({}) reached. Current solution saved'.format(max_iterations))
                break
            iteration += 1
        
            
    def _get_step_size(self,gradient_dict,verbose=False):
        min_step_size = .05
        
        def step_size_obj_rule(model):
            z = sum([(self.alpha_dict[con] + model.step_size*gradient_dict[con]) * sum([(self.constraint_df.loc[i,'coeff'] *
                                              pyo.value(self.aug_model.x_ij[self.constraint_df.loc[i,'edge']])) + 
                                             self.constraint_df.loc[i,'rhs']
                                         for i in self.constraint_df[self.constraint_df['constraint']==con].index])
                 for con in self.constraint_df.constraint.unique()])
            return(z)
        step_size_model = pyo.ConcreteModel()
        step_size_model.step_size = pyo.Var(domain=pyo.NonNegativeReals,initialize=min_step_size)
        step_size_model.obj = pyo.Objective(rule = step_size_obj_rule,sense=pyo.minimize)
        # Solve the model
        step_size_solver = pyo.SolverFactory('glpk')
        res = step_size_solver.solve(step_size_model)
        
        # Determine the step size (set to something small if solution says it should be 0)
        step_size = max(min_step_size,pyo.value(step_size_model.step_size))
        if verbose:
            print('Argmin solution: {}\nValue Used: {}'.format(pyo.value(step_size_model.step_size),
                                                               step_size))
        
        return(step_size)
    
    
    def _get_gradient(self):
        '''
        Get the gradient of the objective function w.r.t. the dual variables (alphas)
        
        The alphas are multiplied by the side constraints (ax-b) so we just get the
        value of ax-b for the current iteration and that's the partial derivative
        with respect to that component of alpha

        Parameters
        ----------
        None.

        Returns
        -------
        grad_dict : dict
            The gradient with respect to the alpha associated with each side constraint

        '''
        
        grad_dict = {key:0 for key in self.alpha_dict.keys()}
        for key in self.alpha_dict.keys():
            c_df = self.constraint_df.loc[self.constraint_df['constraint']==key,:]
            grad_dict[key] = float(sum([a*x for a,x in zip(c_df['coeff'].values,
                                                     [pyo.value(self.aug_model.x_ij[edge])
                                                      for edge in c_df['edge'].values])]) -
            c_df['rhs'].unique())
                                                      
        
        return(grad_dict)
        
    
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
if __name__ == '__main__':
    # Setup input data
    # edge_df = pd.DataFrame({'source':[1,1,2,2,3],
    #                           'target':[2,3,3,4,4],
    #                           'weight':[0,0,0,1,1],
    #                           'capacity':[1]*5,
    #                           'con_c':[0,0,0,1,0]})
    edge_df = pd.DataFrame({'source':[1,1,2,2,3],
                              'target':[2,3,3,4,4],
                              'weight':[0,0,0,1,0], # costs 1 to go 2-4
                              'capacity':[1,1,1,1,2], # can send 2 units on edges (3,4) 1 on others
                              'con_c':[0,0,0,1,0]})
    # Single constraint saying edges (2,4) and (3,4) must sent same amount of flow (x_{2,4} - x_{3,4} = 0)
    constraint_df = pd.DataFrame({'constraint':['c1','c1'],
                                  'edge':[(2,4),(3,4)],
                                  'coeff':[1,-1],
                                  'rhs':[0,0],
                                  'sense':['=','=']})
    mult_ind_edge_df = edge_df.set_index(['source','target'])
    # 1 is source node, 4 is sink and all others are transshipment nodes
    node_demands = {1:-2,2:0,3:0,4:2}
    nodes = set(edge_df.source.unique()).union(set(edge_df.target.unique()))
    edges = [(u,v) for u,v in zip(edge_df.source.values,edge_df.target.values)]

    # Build object and model
    a = SCModel(edge_df,node_demands,constraint_df)
    a.build_aug_model()
    a.solve_mcnf_side_con()
    # a.solve(a.aug_model)
    a.aug_model.pprint()
    
    # # Print results
    # print('The obj value is {}'.format(pyo.value(a.aug_model.obj)))
    # print('The flow is below:')
    # a.aug_model.x_ij.pprint()
    
    # flow_s = pd.Series({(u,v):pyo.value(a.aug_model.x_ij[(u,v)]) for u,v in a.aug_edges},name='flow')
    # flow_s.index.names = ['source','target']
    # print(flow_s)
# #%% Example MCNF Model    
# model = pyo.ConcreteModel()
# model.nodes = pyo.Set(nodes)

# def edge_caps(model,u,v):
#     # return(0,1)
#     return((0,mult_ind_edge_df.loc[(u,v),'capacity']))
# model.x_ij = pyo.Var(edges,bounds=edge_caps)

# def obj_rule(model):
#     z = sum([float(mult_ind_edge_df.loc[(u,v),:'weight']) * model.x_ij[(u,v)]
#              for u,v in edges])
#     return(z)
# model.obj = pyo.Objective(rule=obj_rule,sense=1)

# def get_node_balance_lhs(model,node):
#     lhs = (sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
#     sum([model.x_ij[(u,v)] for u,v in edges if u == node]))
#     print(lhs)
#     return(lhs)

# def node_balance_rule(model,node):
#     return((sum([model.x_ij[(u,v)] for u,v in edges if v == node]) -
#     sum([model.x_ij[(u,v)] for u,v in edges if u == node])) <= node_demands[node])
#     # return(get_node_balance_lhs(model,node) <= node_demands[node])

# model.node_balance_cons = pyo.Constraint(nodes,rule=node_balance_rule)


# #%% Solve model
# solver = pyo.SolverFactory('glpk')
# res = solver.solve(model)
# print('The obj value is {}'.format(pyo.value(model.obj)))
# print('The flow is below:')
# model.x_ij.pprint()

# flow_s = pd.Series({(u,v):pyo.value(model.x_ij[(u,v)]) for u,v in edges},name='flow')
# print(flow_s)

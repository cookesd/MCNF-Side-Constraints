# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:22:55 2020

@author: dakar
"""

import networkx as nx
import pandas as pd
from graphs import BaseGraph,AugGraph # the classes for base graphs the augmentation graph

class SideConstraintGraph(object):
    def __init__(self,edgelist_df,node_demand_dict,constraint_df=None):
        # makes the DiGraph with edgelist data if edgelist_df != None
        # makes an empy DiGraph if edgelist_df = None
        self.edgelist_df = edgelist_df
        self.node_demand_dict = node_demand_dict
        self.g = BaseGraph(edgelist_df,node_demand_dict)
        self.aug_g = AugGraph(edgelist_df = self.g.edgelist_df,
                              node_demand_dict = self.g.node_demand_dict,
                              base_graph = self.g)
        # self.graph = nx.DiGraph(incoming_graph_data=edgelist_df)
        self.constraints = pd.DataFrame()
        if constraint_df:
            self.add_constraints(constraint_df)
         
    
        
    
    
    def add_constraints(self,constriant_df):
        pass
        # concats the new constaint_df to the objects constraint_df
        # the constraint_df has columns 
            # con_name : the constraint name
            # source : the source node of the edge
            # target : the target node of the edge
            # coeficient : the coeficient for the edge in the constraint
            # direction : {'>=','<=','=='} the direction of the constraint
            # rhs : the rhs value for the constraint
            # lambda : optional. if not supplied, set to 0
        # lambda is the dual variable we change as we iteratively solve
            
    def remove_constraints(self,con_names):
        # removes all rows where the constraint name is in con_names
        self.constraints = self.constraints.loc[self.constraints.con_name.drop(con_names),:]
        
    def update_node_demands(self,node_demand_dict):
        self.g.update_node_demands(node_demand_dict)
        self.aug_g.update_node_demands(node_demand_dict)
        for node in node_demand_dict.keys():
            if node not in self.nodes():
                self.add_node(node)
            self.nodes[node]['demand'] = node_demand_dict[node]
            
    def solve(self):
        pass
        # should iteratively call nx.min_cost_flow
        # adjusting alpha parameters from in the 
        
    def draw_flow(self,flow_dict,which_g='aug'):
        if which_g == 'aug':
            self.aug_g.draw_flow(flow_dict)
        elif which_g == 'base':
            # removes the flow from the source and sink if present
            self.g.draw_flow({node:flow_dict[node] for node in self.g.nodes})
        # should draw the graph
        # color the supply (demand < 0), demand (demand > 0) and
        # transshippment (demand == 0) nodes different colors
        # distinguish edges that are in unsatisfied constraints???
            
if __name__ == '__main__':
    edges = pd.DataFrame({'source': [0, 1, 2],
                      'target': [2, 2, 3],
                      'weight': [3, 4, 5],
                      'color': ['red', 'blue', 'blue'],
                      'capacity': [5,9,10]})
    node_demands = dict(zip(list(range(6)),
                            [-2,-1,-1,1,2,1]))
    
    g = SideConstraintGraph(edges,node_demands)
    g.g.draw_graph_with_edges()
    g.aug_g.base_graph.draw_graph_with_edges()
    g.aug_g.draw_graph_with_edges(edge_attrs=['weight','capacity','con_c'])


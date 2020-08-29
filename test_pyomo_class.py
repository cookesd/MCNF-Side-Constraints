# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:52:23 2020

@author: dakar
"""

from classes.pyomo_class import SCModel
import pandas as pd

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

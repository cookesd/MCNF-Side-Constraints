# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:57:12 2020

@author: dakar
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

# g = nx.DiGraph()

edge_df = pd.DataFrame({'source':[1,1,2,2,3],
                          'target':[2,3,4,3,4],
                          'weight':[0,0,1,0,1],
                          'capacity':[1]*5,
                          'con_c':[0,0,0,1,0]})
g = nx.from_pandas_edgelist(edge_df,edge_attr=True,
                            create_using=nx.DiGraph)

# edges = [(1,2,0),(1,3,0),
#          (2,4,1),(2,3,0),
#          (3,4,1)]
# g.add_weighted_edges_from(edges)
print(g.edges(data=True))


gpos = {1:(-1,0),
        2:(0,1),
        3:(0,-1),
        4:(1,0),
        'super_source':(-1,1),
        'super_sink':(1,1)}
demands = {1:-2,2:0,3:0,4:2}
for node in g.nodes():
    g.nodes[node]['demand']=demands[node]


#%% Drawing functions
def draw_graph_with_edges(graph,pos):
    node_demands = [d for n,d in graph.nodes(data='demand')]
    # node_colors = [abs(i)/max(node_demands) for i in node_demands]
    cmap = plt.cm.jet
    fig,ax = plt.subplots()
    # nx.draw_networkx(graph,pos,#cmap=plt.cm.get_cmap('jet'),
    #                   node_color=node_colors,#vmin=0,vmax=1,
    #                   ax=ax)
    nx.draw_networkx_edges(graph,pos=pos,
                            ax=ax,
                            label='(weight,capacity,con coeff)')
    nx.draw_networkx_edge_labels(graph,pos=pos,
                                  edge_labels={(u,v):
                                              tuple(d.values())
                                              # dict(zip(['w','cap','con'],
                                                          # d.values()))
                                          for u,v,d in graph.edges(data=True)},
                                  ax=ax)
    nx.draw_networkx_nodes(graph,pos,node_color=node_demands,
                            cmap = cmap,
                            alpha=.8)
    nx.draw_networkx_labels(graph,pos,
                            ax=ax)
    norm = matplotlib.colors.Normalize(vmin=min(node_demands),vmax=max(node_demands))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm,ax=ax,alpha=.8);
    
def draw_flow(graph,flow_dict,pos):
    flow_labels = {(k,v1):'flow: '+str(v2) 
                   for k,v in flow_dict.items()
                   for v1,v2 in v.items()}
    nx.draw_networkx(graph,pos=pos)
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=flow_labels)
    plt.show();



#%% Solve max flow without constraints

draw_graph_with_edges(g,gpos)
unconstrained = nx.max_flow_min_cost(g,s=1,t=4)
draw_flow(g,unconstrained,gpos)

#%% Augment with source and sink
aug_g = g.copy()
aug_gpos = gpos.copy()
aug_gpos['super_source'] = (min([x for x,y in gpos.values()])-1,
                            max([y for x,y in gpos.values()])+1)
aug_gpos['super_sink'] = (max([x for x,y in gpos.values()])+1,
                            max([y for x,y in gpos.values()])+1)
sum_demand = sum([d for n,d in g.nodes(data='demand') if d>0])
max_weight = max([w for u,v,w in g.edges(data='weight')])
# Add edge with 0 weight from super source to nodes with negative demand (demand nodes)
aug_g.add_weighted_edges_from([('super_source',n,0)
                            for n in g.nodes()
                          if g.nodes[n]['demand'] < 0])
aug_g.nodes['super_source']['demand'] = -sum_demand
# Add edge with 0 weight to super source from nodes with positive demand (supply nodes)
aug_g.add_weighted_edges_from([(n,'super_sink',0)
                            for n in g.nodes()
                            if g.nodes[n]['demand'] > 0])

aug_g.nodes['super_sink']['demand'] = sum_demand

# We will only use the edge from super source to sink if we can't get any other path
aug_g.add_weighted_edges_from([('super_source','super_sink',
                                max_weight*len(g.edges()))])

# Set capacities on the new edges
    # If it's from super source to node in original graph
    # then it equals the nodes demand from original graph
    # If it's the super_source to super_sink, it's the max demand
    # It it's from a node in original graph to super_sink
    # then it's the nodes supply from the original graph
    # Then set those node demands to zero (only super source and sink have demands/supply)
for u,v in aug_g.edges():
    if u == 'super_source':
        aug_g.edges[(u,v)]['con_c'] = 0
        if v != 'super_sink':
            # set the capacity to the nodes demand from the original graph
            aug_g.edges[(u,v)]['capacity'] = -g.nodes[v]['demand']
            aug_g.nodes[v]['demand'] = 0 # set demand to 0
        else: # this it the super_sink
            aug_g.edges[(u,v)]['capacity'] = sum_demand
    elif v == 'super_sink':
        aug_g.edges[(u,v)]['con_c'] = 0
        aug_g.edges[(u,v)]['capacity'] = g.nodes[u]['demand']
        aug_g.nodes[u]['demand'] = 0 # set demand to 0
print(aug_g.nodes(data=True))
print(aug_g.edges(data='capacity'))
draw_graph_with_edges(aug_g,aug_gpos)

#%% Iteratively solve with updated alphas
base_res = nx.network_simplex(aug_g,demand='demand',capacity='capacity',
                              weight='weight')
draw_flow(aug_g,flow_dict=base_res[1],pos=aug_gpos)

con_lhs = base_res[1][2][3]
con = con_lhs < 1

summary = {0:{'l_cost':base_res[0],'flow':con_lhs,
              'true_cost':sum([base_res[1][u][v] * aug_g.edges()[(u,v)]['weight']
                               for u,v in aug_g.edges()]),
              'mu':0}}
mu = -1
i = 1
while con == True:
    for u,v in aug_g.edges():
        aug_g.edges[(u,v)]['l_weight'] = (aug_g.edges[(u,v)]['weight'] + 
        (mu * aug_g.edges[(u,v)]['con_c']))
    aug_res = nx.network_simplex(aug_g,demand='demand',capacity='capacity',
                                 weight='l_weight')
    con_lhs = aug_res[1][2][3]
    con = con_lhs < 1
    summary[i] = {'l_cost':aug_res[0],
                  'flow':con_lhs,
                  'true_cost':sum([aug_res[1][u][v] * aug_g.edges()[(u,v)]['weight']
                               for u,v in aug_g.edges]),
                  'mu':mu}
    print(pd.Series(summary[i],name=i))
    mu = mu - 1
    i += 1
print(pd.DataFrame(summary))
draw_flow(aug_g,flow_dict=aug_res[1],pos=aug_gpos)
#%% Process
# 1. Get original graph with demands
# 2. Add super source (and sink) with demands equal to the min (max) of
    # previous demand nodes (negative demand means it's a supply node)
# 3. Add edges from super source to all previous source nodes and the super sink node
    # and edges from all previous demand nodes to the super sink
# 3. Set demands of all other nodes to 0
# 4. Solve (network simplex) and check if constraint satisfied
# 5. Iteratively adjust the lagrangian coefficient to force constraint to be
    # satisfied, but tune so not 'overly satisfied' (decreases actual obj too much)
# The flow from the super source node to the super sink node is the demand
# that could have been sent in the max flow scenario but can't because of the constraints
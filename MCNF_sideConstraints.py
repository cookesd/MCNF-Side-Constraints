# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 17:53:03 2019

@author: cookesd
"""

import networkx as nx
import matplotlib.pyplot as plt
import time

def makeNodeCols(graph):
    '''Makes suply nodes green, demand nodes red, transshipment nodes white'''
    for v in graph.nodes():
        if graph.nodes[v]['demand']< 0:
            graph.nodes[v]['color'] = 'green'
        elif graph.nodes[v]['demand'] > 0:
            graph.nodes[v]['color'] = 'red'
        else: graph.nodes[v]['color'] = 'white'
def drawSol(graph,flowDict,pos = None):
    plt.close()
    if pos==None:
        pos=nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_color=[graph.nodes[v]['color'] for v in graph.nodes()],edgecolors='k')
    nx.draw_networkx_labels(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=flowDict)
    plt.show()

gDict = {0:{1:{'capacity':6},
            2:{'capacity':2},
            15:{'capacity':8}},
         1:{2:{'capacity':1},
            3:{'capacity':3}},
         2:{4:{'capacity':7}},
         3:{2:{'capacity':3},
           5:{'capacity':2}},
         4:{5:{'capacity':7}},
         5:{15:{'capacity':1}},
         10:{5:{'capacity':8},
             15:{'capacity':8}}}
         
G = nx.from_dict_of_dicts(gDict,create_using=nx.DiGraph)
#nx.draw_networkx(G)


for n in G.nodes():
    G.nodes[n]['demand'] = 0
G.nodes[0]['demand'] = -8
G.nodes[10]['demand']=-9
G.nodes[5]['demand'] = 9
G.nodes[15]['demand']=8

makeNodeCols(G)
gpos = nx.circular_layout(G)


#
#print(G.nodes(data=True))
#print(G.edges())


#Prints the capacities for each edge in the graph
#for u,v in G.edges():
#    print(G[u][v])


mu = 0   
## made up side constraint (flow from 3 to sink (5) == flow from 4 to sink (5))
for u,v in G.edges():
    G[u][v]['weight'] = 0 #no weights in original, just set them to 0 so they're all the same
    if u == 3 and v == 5:
        G[3][5]['c1'] = 1
    elif u == 4 and v == 5:
        G[4][5]['c1'] = -1
    else:
        G[u][v]['c1']=0
    G[u][v]['lCost'] = G[u][v]['weight'] - mu*G[u][v]['c1']

nx.draw_networkx_nodes(G,gpos,node_color=[G.nodes[v]['color'] for v in G.nodes()],edgecolors='k')
nx.draw_networkx_labels(G,gpos)
nx.draw_networkx_edges(G,gpos)
nx.draw_networkx_edge_labels(G,gpos,edge_labels={(u,v):(d['capacity'],d['weight'],d['c1']) for (u,v,d) in G.edges(data=True)})
plt.show()



res = nx.network_simplex(G,weight='lCost') #gives tuple [0]=cost, [1] = flowdict
print(res)
flowDict = {}
for node in res[1]:
    for key in res[1][node]:
        flowDict[(node,key)]=res[1][node][key]
drawSol(G,flowDict,gpos)
#print edge data
#for u,v in G.edges():
#    print(G[u][v])
#res = nx.max_flow_min_cost(G,0,5)
#
b = 0
con1 = sum([res[1][u][v]*G[u][v]['c1'] for u,v in G.edges()])-b #Ax-b
z = [res[0]]
constraint = [con1]
mus = [mu]
theta = 1.05
thetas = []
#con1

for k in range(1,50):
    print('iteration',k)
    theta = theta-.05
    mu = mu+theta*(con1)
    thetas.append(theta)
    mus.append(mu)
    
    res = nx.network_simplex(G,weight='lCost')
    con1 = sum([res[1][u][v]*G[u][v]['c1'] for u,v in G.edges()])-b #Ax-b
    print('Ax-b =',con1)
    for u,v in G.edges():
        G[u][v]['lCost'] = G[u][v]['weight']+mu*G[u][v]['c1'] #gets combined coeff for x (c+mu*A)
    print(res)
    z.append(res[0])
    constraint.append(con1)
    flowDict = {}
    for node in res[1]:
        for key in res[1][node]:
            flowDict[(node,key)]=res[1][node][key]
    #drawSol(G,flowDict,gpos)
#    end = input('press enter to continue,-1 to quit')
#    if end == str(-1):
#        print('quitting')
#        break
    print('mu =',mu)
    
    
    
    


while con1 != b:
    if con1 > b: #flow on (4,5)>(3,5) so make (4,5) cost more
        for u,v in G.edges():
            G[u][v]['weight'] = G[u][v]['weight']+step*G[u][v]['c1']
    if con1 < b: #flow on (4,5)<(3,5) so make (3,5) cost more
        for u,v in G.edges():
            G[u][v]['weight'] = G[u][v]['weight']-step*G[u][v]['c1']
    print('The weights are, \n', {(3,5):G[3][5]['weight'], (4,5): G[4][5]['weight']})
    start = time.time()
    res = nx.network_simplex(G)
    print('simplex took ', time.time()-start, 'seconds')
    con1 = sum([res[1][u][v]*G[u][v]['c1'] for u,v in G.edges()])
    print('Diff (4,5)-(3,5) = ',con1)
    print('Flow is \n',res)
    flowLabels = {(u,v):res[1][u][v] for u,v in G.edges()}
    
    plt.figure()
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_labels(G,pos)
    nx.draw_networkx_edges(G,pos,arrows=True)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=flowLabels)
    plt.axis('off')
    plt.show()
    step = float(input('How large to step, 0 to quit:'))
    if step == 0.0:
        con1=0
        print('User decided to quit')
    

#for u,v in G.edges():
#    G[u][v]['weight'] = G[u][v]['weight']-1000*G[u][v]['c1'] # negative increases flow from 4 to 5; want decrese
#res2 = nx.network_simplex(G)
#print(res2)
#
#con1_2 = sum([res2[1][u][v]*G[u][v]['c1'] for u,v in G.edges()]) #lhs for c1
#con2_2 = sum([res2[1][u][v]*G[u][v]['c2'] for u,v in G.edges()]) # lhs for c2
#con1_2
#con2_2
#
#flowLabels = {(u,v):res2[1][u][v] for u,v in G.edges()}
#plt.figure()
#pos = nx.circular_layout(G)
#nx.draw_networkx_nodes(G,pos)
#nx.draw_networkx_labels(G,pos)
#nx.draw_networkx_edges(G,pos,arrows=True)
#nx.draw_networkx_edge_labels(G,pos,edge_labels=flowLabels)
#plt.axis('off')
#plt.show()
##run original nx.max_flow_min_cost
## Then check if sumproduct (cons,flow) = 0
## while not, increase 'dual var' and rerun flow
## need to print out (or save) flow and checks each iteration to check it
#
#### for generality, would need an extra edge attribute for each constraint
#### this is a equal to constraint so really need two with opposite signs to complete
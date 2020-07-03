# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:09:35 2020

@author: dakar
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import copy

#%% BaseGraph
class BaseGraph(nx.DiGraph):
    def __init__(self,edgelist_df,node_demand_dict,name='Base Graph'):
        '''
        

        Parameters
        ----------
        edgelist_df : pd.DataFrame
            The edgelist for the graph. Each row represents an edge.
            The df Must contain columns, {'source','target','weight','capacity'}
            to set the attributes for the edges. Any other columns  will be
            additional attributes in the graph
        node_demand_dict : dict
            Dict with node-demand, key-value pairs for the graph. Nodes with
            zero demand can be ommited, but there must be at least one node each
            with positive and negative demand. The demands must sum to 0. Positive
            demand means the node wants to receive flow (demand node).
            Negative demand means, the node wants to send flow (supply node).
            Nodes with zero demand must have inflow - outflow = 0 (transshipment).
            This is opposite of most network flows, but networkx handles flow
            this way.

        Returns
        -------
        None.

        '''
        if edgelist_df.shape[0] == edgelist_df.shape[1]:
            # The __init__ treats df as an adjacency list if it has symmetric shape
            # so we remove an edge, then add it to the graph and replace it
            # in the edgelist_df
            first_edge = edgelist_df.loc[0,:]
            edgelist_df = edgelist_df.iloc[1:,:]
        # Make initial graph (with one edge removed if df has symmetric shape
        nx.DiGraph.__init__(self,incoming_graph_data=edgelist_df)
        # Adds the removed edge it df has symmetric shape
        if edgelist_df.shape[0] == edgelist_df.shape[1]:
            s = first_edge.source
            t = first_edge.target
            self.add_weighted_edges_from([(first_edge.source,
                                           first_edge.target,
                                           first_edge.weight)])
            for attr in first_edge.drop[['source','target']].index:
                self.edges[s][t][attr] = first_edge[attr]
            # add the edge back to the edgelist_df
            edgelist_df = pd.concat([pd.DataFrame(first_edge).transpose(),
                                    edgelist_df],axis='index')
        
        self.edgelist_df = edgelist_df
        self.node_demand_dict = node_demand_dict
        self.set_node_demands(self.node_demand_dict)
        self.pos = self.make_pos()
        
        self.name = name
        
    def __deepcopy__(self, memo):
        copy_g = BaseGraph(copy.deepcopy(self.edgelist_df,memo),
                          copy.deepcopy(self.node_demand_dict,memo),
                          copy.deepcopy(self.name,memo))
        return(copy_g)
            
    def set_node_demands(self,node_demand_dict):
        '''
        Set the demand for all nodes in the graph.
        
        Set the node demands for all nodes based on the dict values. Nodes
        not in the dict keys will default to 0, but there must be at least one
        node each with positive and negative demand. The demands must sum to 0.
        If your demands don't sum to zero, you can set up artifical supply/demand
        nodes to account for the surplus.'

        Parameters
        ----------
        node_demand_dict : dict
            Dict with node-demand, key-value pairs for the graph. Nodes with
            zero demand can be ommited, but there must be at least one node each
            with positive and negative demand. The demands must sum to 0. Positive
            demand means the node wants to receive flow (demand node).
            Negative demand means, the node wants to send flow (supply node).
            Nodes with zero demand must have inflow - outflow = 0 (transshipment).
            This is opposite of most network flows, but networkx handles flow
            this way.

        Returns
        -------
        None.

        '''
        
        for node in self.nodes():
            # if node not in the dict then default to 0
            _demand = node_demand_dict.get(node,0)
            self.nodes()[node]['demand'] = _demand
            self.nodes()[node]['type'] = self._determine_node_type(_demand)
            self.node_demand_dict[node] = _demand
    
    def make_pos(self):
        '''
        Make a position dictionary stating location of nodes in the graph
        
        Currently just uses the nx.spring_layout, but can define own method
        to have supply nodes at left, demand at right and all others in between
        based on path lengths to supply/demand nodes

        Returns
        -------
        None.

        '''
        pos = nx.spring_layout(self)
        return(pos)
            
    def update_node_demands(self,update_dict):
        '''
        Update the demands for nodes in the dict

        Parameters
        ----------
        update_dict : dict
            Dict with node-demand, key-value pairs for the graph. Positive
            demand means the node wants to receive flow (demand node).
            Negative demand means, the node wants to send flow (supply node).
            Nodes with zero demand must have inflow - outflow = 0 (transshipment).
            This is opposite of most network flows, but networkx handles flow
            this way.

        Returns
        -------
        None.

        '''
        for node in update_dict.keys():
            # probably don't need to store it in two places
            # I'm not sure which is the easier way to use
            # for plotting, I still have to make it a dict
            # The nx functionality is more cumbersome, but
            # accomodates multiple attributes.
            _demand = update_dict[node]
            self.node_demand_dict[node] = _demand
            self.nodes()[node]['type'] = self._determine_node_type(_demand)
            self.nodes()[node]['demand'] = _demand
            
    def _determine_node_type(self,demand):
        '''
        Determines if a node is a supply, demand, or transshipment node
        '''
        if demand > 0:
            return('demand')
        elif demand < 0:
            return('supply')
        else:
            return('transshipment')
        
        
    def add_sc_edges(self,edgelist_df):
        '''
        Add edges to the graph from a DataFrame edgelist

        Parameters
        ----------
        edgelist_df : pd.DataFrame
            DataFrame where each row represents an edge. It must have columns
            named 'source' and 'target' representing the from and to location
            of the edge. Other columns represent attributes for the edge.
            Potential entries are 'weight', 'cost', 'time', etc.

        Returns
        -------
        None.

        '''
        # adds the edges to the graph
        self.add_edges_from([(u,v)
                             for u,v in edgelist_df[['source','target']].values])
        
        # sets attributes to the edges if attribute columns present in the df
        attr_cols = edgelist_df.drop(columns=['source','target']).columns
        if len(attr_cols) > 0:
            attr_df = edgelist_df.set_index(['source','target'])
            nx.set_edge_attributes(self,values={(u,v):{col:attr_df.loc[(u,v),col]
                                                       for col in attr_cols}
                                                for u,v in attr_df.index.values})
            
    def draw_flow(self,flow_dict,title=None):
        flow_labels = {(k,v1):'flow: '+str(v2) 
                   for k,v in flow_dict.items()
                   for v1,v2 in v.items()}
        nx.draw_networkx(self,pos=self.pos)
        nx.draw_networkx_edge_labels(self,self.pos,edge_labels=flow_labels)
        if title:
            plt.title(title)
        elif title != '':
            plt.title(self.name)
        plt.show();
        
    def draw_graph_with_edges(self,title=None,edge_attrs = ['weight','capacity']):
        node_demands = [d for n,d in self.nodes(data='demand')]
        # node_colors = [abs(i)/max(node_demands) for i in node_demands]
        cmap = plt.cm.jet
        fig,ax = plt.subplots()
        # nx.draw_networkx(graph,pos,#cmap=plt.cm.get_cmap('jet'),
        #                   node_color=node_colors,#vmin=0,vmax=1,
        #                   ax=ax)
        nx.draw_networkx_edges(self,pos=self.pos,
                                ax=ax,
                                label='(weight,capacity,con coeff)')
        nx.draw_networkx_edge_labels(self,pos=self.pos,
                                      edge_labels={(u,v):
                                                  tuple([d.get(attr,0) for attr in edge_attrs])
                                                  #d.values())
                                                  # dict(zip(['w','cap','con'],
                                                              # d.values()))
                                              for u,v,d in self.edges(data=True)},
                                      ax=ax)
        nx.draw_networkx_nodes(self,self.pos,node_color=node_demands,
                                cmap = cmap,
                                alpha=.8)
        nx.draw_networkx_labels(self,self.pos,
                                ax=ax)
        ax.plot(np.NaN, np.NaN, '-', color='none',
                label='({})'.format(','.join(edge_attrs)))
        norm = matplotlib.colors.Normalize(vmin=min(node_demands),
                                           vmax=max(node_demands))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm,ax=ax,alpha=.8)
        
        if title:
            plt.title(title)
        elif title != '':
            plt.title(self.name)
        plt.legend();
        # plt.legend(el,'({})'.format(','.join(edge_attrs)));

#%% AugGraph
class AugGraph(BaseGraph):
    def __init__(self,edgelist_df,node_demand_dict,base_graph,name='Augmentation Graph'):
        # Makes the copy of the base graph and initializes the AugGraph
        ### Not sure if I actually want to hold onto the base graph
        ### because it causes multiple copies of the same object
        ### I might just want to use the __init__ to copy
        ### and let the sideconstraintgraph keep the base_graph info
        ### Keeping the copy allows me to keep track of the demands if they change
        self.base_graph = copy.deepcopy(base_graph)#.copy()
        BaseGraph.__init__(self,self.base_graph.edgelist_df,self.base_graph.node_demand_dict,name=name)
        
        # Adds the super source and sink nodes
        super_nodes = ['super_source','super_sink']
        self.add_nodes_from(super_nodes)
        # self.node_demand_dict['super_source'] = 0
        # self.node_demand_dict['super_sink'] = 0
        
        ### get the total weight and demand info from the base_graph
        # get the max weight
        max_weight = max([w for u,v,w in base_graph.edges(data='weight')])
        self.base_supply_nodes_demands = {} # set in base_star_nodes_demands
        self.base_demand_nodes_demands = {} # set in base_star_nodes_demands
        self.sum_pos_demand = None # set in base_star_nodes_demands
        self.set_base_star_nodes_demands()
        
        
        ### add new edges ###
        new_edge_dicts = [
            # source to supply node edges
            [{'source':'super_source','target':node,
              'weight':0,'capacity':demand}
             for node,demand in self.base_supply_nodes_demands.items()] +
            # demand to sink nodes
            [{'source':node,'target':'super_sink',
              'weight':0,'capacity':demand}
             for node,demand in self.base_demand_nodes_demands.items()] +
            # source to sink edge
            [{'source':'super_source','target':'super_sink',
              'weight':max_weight * len(base_graph.edges) * self.sum_pos_demand,
              'capacity':self.sum_pos_demand}]]
        
        new_edge_df = pd.DataFrame({i:edge_dict
                                    for i,edge_dict
                                    in enumerate(new_edge_dicts[0])}).transpose()
        self.edgelist_df = pd.concat([self.edgelist_df,new_edge_df],
                                     axis='index')
        self.add_sc_edges(new_edge_df)
        
        ### Update Demands ###
        for node in self.base_graph.nodes():
            self.nodes[node]['demand'] = 0 # original node demands go to 0
        self.set_super_demands()
        # self.nodes['super_source']['demand'] = - sum_pos_demand
        # self.nodes['super_sink']['demand'] = sum_pos_demand
        
        ### Update pos ###
        self.pos = self.base_graph.pos.copy()
        self.pos['super_source'] = (min([x for x,y in self.pos.values()])-.5,
                            max([y for x,y in self.pos.values()])+.5)
        self.pos['super_sink'] = (max([x for x,y in self.pos.values()])+.5,
                            max([y for x,y in self.pos.values()])+.5)
    
    def set_base_star_nodes_demands(self):
        self.base_supply_nodes_demands = {node: self.base_graph.nodes[node]['demand']
                                     for node in self.base_graph
                                     if self.base_graph.nodes[node]['type'] == 'supply'}
        self.base_demand_nodes_demands = {node: self.base_graph.nodes[node]['demand']
                                     for node in self.base_graph
                                     if self.base_graph.nodes[node]['type'] == 'demand'}
        self.sum_pos_demand = max([sum([-d for n,d in self.base_supply_nodes_demands.items()]),
                                   sum([d for n,d in self.base_demand_nodes_demands.items()])])
    
    def update_node_demands(self,update_dict):
        '''
        Update the demands for nodes in the dict

        Parameters
        ----------
        update_dict : dict
            Dict with node-demand, key-value pairs for the graph. Positive
            demand means the node wants to receive flow (demand node).
            Negative demand means, the node wants to send flow (supply node).
            Nodes with zero demand must have inflow - outflow = 0 (transshipment).
            This is opposite of most network flows, but networkx handles flow
            this way.

        Returns
        -------
        None.

        '''
        self.base_graph.update_node_demands(update_dict)
        self.set_base_star_nodes_demands()
        self.set_super_demands()
        
    def set_super_demands(self):
        
        ### gets the total inflow/outflow ###
        self.nodes['super_source']['demand'] = -self.sum_pos_demand
        self.nodes['super_sink']['demand'] = self.sum_pos_demand
        self.node_demand_dict['super_source'] = -self.sum_pos_demand
        self.node_demand_dict['super_sink'] = self.sum_pos_demand
        
    
if __name__ == '__main__':        
    edge_df = pd.DataFrame({'source':[1,1,2,2,3],
                          'target':[2,3,4,3,4],
                          'weight':[0,0,1,0,1],
                          'capacity':[1]*5,
                          'con_c':[0,0,0,1,0]})
    node_demands = {1:-2,2:0,3:0,4:2}
    g = BaseGraph(edge_df,node_demands)
    g.draw_graph_with_edges()
    aug_g = AugGraph(edgelist_df=g.edgelist_df,
                      node_demand_dict=g.node_demand_dict,base_graph=g)
    aug_g.draw_graph_with_edges(edge_attrs=['weight','con_c','capacity'])
    print(g.edges(data=True))
    print(g.nodes())
    
    print(g.node_demand_dict)
    print(aug_g.node_demand_dict)

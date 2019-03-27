import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uuid
import itertools


class World:
    def __init__(self,  agents, init_edge_proba=0, agent_power=3):
        self.G=nx.Graph()
        
        for agent in agents:
            self.G.add_node(agent)
            
        for target in self.G.nodes:
            for other in self.G.nodes:
                if target != other and random.random() < init_edge_proba:
                    self.G.add_edge(target, other)
                    
        self.history = {}
        self.step = 0
        self.agent_power = agent_power 
        
    def draw(self):
        
        ndict = dict(zip(self.G.nodes, [str(i.name) for i in self.G.nodes]))
        fig = plt.figure(figsize = (10,10))
        pos=nx.spring_layout(self.G)
        nx.draw(self.G, pos = pos)
        nx.draw_networkx_labels(self.G, pos=pos, labels=ndict)
        plt.show()
        
    def edge_evaluation(self, pred_a, pred_b):
        return (pred_a+pred_b)/2*random.uniform(0,2) > 0.5
        
    def selection(self):
        pass
        
    def distribute_power(self, dictionary):
        distributed = [i*self.agent_power/sum(dictionary.values()) for i in dictionary.values()]
        return(dict(zip(dictionary.keys(), distributed)))
    
    def update_history(self, step, **kwargs):
        self.history[step] = self.G.copy()
        
    def iterate(self, iterations, verbose=0, draw=False):
        for iteration in range(iterations):
            self.update_history(step=self.step)
            for agent in self.G.nodes:
                print(agent.name)
                agent.learn(self)
                agent.predict(self)
                agent.prediction_dict[self.step] = self.distribute_power(agent.predict(self))
                        
            for pair in itertools.combinations(self.G.nodes, 2):   
                try:
                    
                    if self.edge_evaluation(pair[0].prediction_dict[self.step][pair[1]],
                                            pair[1].prediction_dict[self.step][pair[0]]):

                        if pair not in self.G.edges:
                            self.G.add_edge(*pair)

                    elif pair in self.G.edges:
                        self.G.remove_edge(*pair)
                except KeyError:
                    print(pair)
                    print(pair[0].prediction_dict[self.step])
                    print(pair[1].prediction_dict[self.step])
                    return pair
            if draw:
                self.draw()
            
            self.selection()
            self.step += 1


def get_name(world, name):
    for i in world.G.nodes:
        if i.name == name:
            return i


class selectiveWorld(World):
    def __init__(self,  agents, init_edge_proba=0, agent_power=3, min_degs=1, selection_proba=0.2):
        World.__init__(self,  agents=agents, init_edge_proba=init_edge_proba, agent_power=agent_power)
        self.min_degs = min_degs
        self.selection_proba = selection_proba
    def selection(self):
        for node in list(self.G.nodes):
            if self.G.degree(node) < self.min_degs and random.random() < self.selection_proba:
                self.G.remove_node(node)
                

class AgentSkeleton:
    def __init__(self, name=None):
        self.prediction_dict = {}
         
        if name is None:
            self.name = '{}-{}'.format(''.join([i for i in self.__class__.__name__ if i.isupper() or i.isdigit()]), 
                                       str(uuid.uuid4())[:8])
        else: 
            self.name = name
            
    def learn(self, world):
        pass
    
    def predict(self, world):
        return dict(zip(list(world.G.nodes), [random.random() for i in world.G.nodes]))
    
    def calculate_embeddedness(self, world):
        return world.centrality_dict[self]       
    

class GetBasedonDegree_V1(AgentSkeleton):
    def predict(self, world):
        centrality_dict = nx.degree_centrality(world.G)
        if sum(centrality_dict.values()) > 0:
            return centrality_dict
        else:
            return dict(zip(list(world.G.nodes), [random.random() for i in world.G.nodes]))


class GetBasedonDegree_V2(AgentSkeleton):
    def predict(self, world):
        centrality_dict = nx.degree_centrality(world.G)
        if sum(centrality_dict.values()) > 0:
            return centrality_dict
        else:
            chosen = random.sample(list(world.G.nodes), world.agent_power)
            return dict(zip(list(world.G.nodes), [1 if i in chosen else 0 for i in list(world.G.nodes)]))


class GetBasedonDegree_V3(AgentSkeleton):
    def predict(self, world):
        centrality_dict = nx.degree_centrality(world.G)
        if sum(centrality_dict.values()) > 0:
            res = centrality_dict.copy()
            for key in res.keys():
                res[key] += 0.1
            return res
        else:
            return dict(zip(list(world.G.nodes), [random.random() for i in world.G.nodes]))


class RegressorAgent(GetBasedonDegree_V1):
    def __init__(self, regressor, name=None):
        GetBasedonDegree_V1.__init__(self, name=name)
        self.regressor = regressor
        self.scaler = StandardScaler()
        self.colnames = ['DEGREE_CENTRALITY', 
                         'CLOSENESS_CENTRALITY', 
                         'SUBJECT_DEGREE_CENTRALITY', 
                         'SUBJECT_CLOSENESS_CENTRALITY'] 
    
    def _get_subject_data(self, snapshot, subject, target=None):
        learn_df = pd.DataFrame()
        learn_df['DEGREE_CENTRALITY'] = pd.Series(nx.degree_centrality(snapshot))
        learn_df['CLOSENESS_CENTRALITY'] = pd.Series(nx.closeness_centrality(snapshot))

        learn_df['SUBJECT_DEGREE_CENTRALITY'] = learn_df['DEGREE_CENTRALITY'][subject]
        learn_df['SUBJECT_CLOSENESS_CENTRALITY'] = learn_df['CLOSENESS_CENTRALITY'][subject]
            
        if target is not None:
            learn_df['TARGET'] = pd.Series(subject.prediction_dict[target])
            learn_df = learn_df.reset_index(drop=True)
                    
        return learn_df
        
    def _get_snapshot_data(self, snapshot, target=None):
        df_result = pd.DataFrame()
        for subject in snapshot.nodes:
            learn_df = self._get_subject_data(snapshot, subject, target)
            df_result = df_result.append(learn_df)
        return df_result
    
    def _data_preprocessing(self, history):
        df_learn_appended = pd.DataFrame()
        for step in list(history.keys())[:-1]:
            snapshot = history[step]
            df_learn_appended = df_learn_appended.append(self._get_snapshot_data(snapshot, target = step))
            df_learn_appended = df_learn_appended.fillna(0)
        pred_cols = [i for i in df_learn_appended.columns if i != 'TARGET']
        X = self.scaler.fit_transform(df_learn_appended[df_learn_appended.columns[1:]])
        y = df_learn_appended['TARGET'].values

        return X, y
            
    def learn(self, world):
        if len(world.history) > 1:
            X, y = self._data_preprocessing(world.history)
            self.regressor.fit(X, y)
        else:
            pass
        
    def predict(self, world):
        if len(world.history) > 1:
            input_df = self._get_subject_data(world.G, subject=self, target=None)
            X = self.scaler.transform(input_df.copy())
            pred = self.regressor.predict(X)
            result = dict(zip(input_df.index, pred))
            return result
        else:
            return GetBasedonDegree_V1.predict(self, world)


class GreedyRegressorAgent(RegressorAgent):
    def predict(self, world):
        edge_proba = RegressorAgent.predict(self, world)
        edge_value = dict(nx.closeness_centrality(world.G))
        result = {}
        for i in edge_proba.keys():
            result[i] = edge_proba[i]*(1+(edge_value[i]))

        return result
        


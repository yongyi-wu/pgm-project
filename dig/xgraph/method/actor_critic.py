import sys
import argparse
import numpy as np

import torch
from torch.autograd import Variable

from benchmarks.xgraph.gnnNets import GCNNet
from dig.xgraph.method import SubgraphX 


# subgraphX will return nodes as a result, we will try to use the nodes that subgraphX returns to
# guide the training of a DNN (GNN) that takes as input the same x (node + edge) as subgraphX and
# try to get similar results as subgraphX

class Actor_Critic(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, critic: SubgraphX, actor: GCNNet, actor_lr=0.01, actor_epoch=200):
        # critic will be subgraphX net
        # actor will be our custom-defined GNN
        self.actor = actor
        self.actor_lr = actor_lr
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.actor_epoch = actor_epoch
        self.critic = critic
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def critic_step(self, x, edge_index):
        _, explanation_results, related_preds = self.critic(x, edge_index)
        num_nodes = x.shape()[0]
        one_hot_encoding = [1 if i in explanation_results else 0 for i in range(num_nodes)]
        return one_hot_encoding

    def actor_step(self, x, edge_index, one_hot_encoding):
        self.actor_optim.zero_grad()
        # probability of selecting the nodes
        prediction_results = self.actor.forward(x=x, edge_index=edge_index)
        loss = self.criterion(prediction_results.squeeze(), one_hot_encoding.float())
        loss.backward()
        self.actor_optim.step()
        return loss

    def train(self, x, edge_index):
        one_hot_encoding = self.critic_step(x, edge_index)
        loss = self.actor_step(x, edge_index, one_hot_encoding)
        return loss
    
    def test(self, x, edge_index):
        with torch.no_grad():
            prediction_results = self.actor.forward(x=x, edge_index=edge_index)
            return prediction_results





import torch
from torch import nn

from benchmarks.xgraph.gnnNets import GCNNet
from dig.xgraph.method import SubgraphX 


# subgraphX will return nodes as a result, we will try to use the nodes that subgraphX returns to
# guide the training of a DNN (GNN) that takes as input the same x (node + edge) as subgraphX and
# try to get similar results as subgraphX

class Actor_Critic(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, critic: SubgraphX, actor: GCNNet, actor_lr=0.01, actor_epoch=200, lamda=1):
        # critic will be subgraphX net
        # actor will be our custom-defined GNN
        self.actor = actor
        self.actor_lr = actor_lr
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.actor_epoch = actor_epoch
        self.critic = critic
        self.bce = nn.BCEWithLogitsLoss()
        self.lamda = lamda

    def criterion(self, ac_ex, mcts_ex, ac_class, gnn_class):
        return self.bce(ac_ex, mcts_ex) - self.lamda * (torch.softmax(ac_class, -1) * gnn_class).sum()

    def critic_step(self, x, edge_index):
        _, explanation_results, related_preds = self.critic(x, edge_index)
        num_nodes = x.shape()[0]
        one_hot_encoding = [1 if i in explanation_results else 0 for i in range(num_nodes)]
        return one_hot_encoding

    def actor_step(self, x, edge_index, one_hot_encoding, prediction_dist, node_idx=None):
        self.actor_optim.zero_grad()
        # probability of selecting the nodes
        explanation, classification = self.actor(x=x, edge_index=edge_index)
        if node_idx is not None:
            classification = classification[node_idx].unsqueeze(0)
            prediction_dist = prediction_dist[node_idx].unsqueeze(0)
        loss = self.criterion(
            explanation.squeeze(),
            one_hot_encoding.float(),
            classification,
            prediction_dist.detach().to(classification.device)
        )
        loss.backward()
        self.actor_optim.step()
        return explanation

    def train(self, x, edge_index):
        one_hot_encoding = self.critic_step(x, edge_index)
        loss = self.actor_step(x, edge_index, one_hot_encoding)
        return loss
    
    def test(self, x, edge_index):
        with torch.no_grad():
            prediction_results = self.actor.forward(x=x, edge_index=edge_index)
            return prediction_results





import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.xgraph.gnnNets import GCNNet
from dig.xgraph.method import SubgraphX 


# subgraphX will return nodes as a result, we will try to use the nodes that subgraphX returns to
# guide the training of a DNN (GNN) that takes as input the same x (node + edge) as subgraphX and
# try to get similar results as subgraphX

class Actor_Critic(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, critic: SubgraphX, actor: GCNNet, batch_size, actor_lr=0.005, actor_epoch=200, lamda=3):
        # critic will be subgraphX net
        # actor will be our custom-defined GNN
        self.actor = actor
        self.actor_lr = actor_lr
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.actor_epoch = actor_epoch
        self.critic = critic
        self.bce = nn.BCEWithLogitsLoss()
        self.nll = nn.NLLLoss()
        self.lamda = lamda
        self.loss = 0
        self.counter = 0
        self.batch_size = batch_size

    def criterion(self, ac_ex, mcts_ex, ac_class, true_class):
        return self.nll(ac_class, true_class) + self.lamda * self.bce(ac_ex, mcts_ex)

    def critic_step(self, x, edge_index):
        _, explanation_results, related_preds = self.critic(x, edge_index)
        num_nodes = x.shape()[0]
        one_hot_encoding = [1 if i in explanation_results else 0 for i in range(num_nodes)]
        return one_hot_encoding

    def actor_step(self, x, edge_index, one_hot_encoding, label, node_idx=None):
        if self.counter == 0:
            self.actor_optim.zero_grad()
        self.counter += 1
        explanation, _ = self.actor(x=x, edge_index=edge_index)

        row, col = edge_index.to(edge_index.device)
        edge_mask = (one_hot_encoding[row] == 1) & (one_hot_encoding[col] == 1)
        _, classification = self.actor(x=x, edge_index=edge_index[:, edge_mask])

        if node_idx is not None:
            classification = classification[node_idx].unsqueeze(0)

        loss = self.criterion(
            explanation.squeeze(),
            one_hot_encoding.float(),
            classification,
            label
        )
        self.loss += 1/self.batch_size * loss
        if self.counter % self.batch_size == 0:
            print(f"loss: {loss} at counter {self.counter}")
            self.loss.backward()
            self.loss = 0
            self.actor_optim.step()
            self.actor_optim.zero_grad()
        return explanation

    # def train(self, x, edge_index):
    #     one_hot_encoding = self.critic_step(x, edge_index)
    #     loss = self.actor_step(x, edge_index, one_hot_encoding)
    #     return loss
    
    def test(self, x, edge_index):
        with torch.no_grad():
            explanation, classification = self.actor.forward(x=x, edge_index=edge_index)
            return explanation





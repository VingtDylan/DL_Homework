import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionNetWork(nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(AttentionNetWork, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector), dim=1)
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),candidate_vector).squeeze(dim=1)
        return target

from torch.distributions import Categorical
import torch
import torch.nn as nn


def agent_copy(model_copy, model):
    for param1, param2 in zip(model_copy.parameters(), model.parameters()):
        param1.data = param2.data


class Agent(nn.Module):
    def __init__(self, agent_model):
        super(Agent, self).__init__()
        self.model = agent_model
        self.onpolicy_reset()
        # self.model.train()

    def onpolicy_reset(self):
        self.probability = None
        self.log_probs = None
        self.rewards = None
        self.action = None

    def forward(self, state):
        logits = self.model(state)
        return logits

    def logits_norm(self, logits):
        normalized_1 = torch.sigmoid(logits)
        normalized_0 = torch.ones_like(normalized_1) - normalized_1
        normalized = torch.stack([normalized_0, normalized_1], dim=4)
        return normalized

    def act(self, state):
        logits = self.forward(state)
        pdparam = self.logits_norm(logits)
        self.probability = pdparam
        pd = Categorical(probs=pdparam)
        action = pd.sample()
        self.action = action
        log_prob = pd.log_prob(action)
        self.log_probs = log_prob


if __name__ == "__main__":
    a = torch.rand((1, 1, 456, 789))
    normalized_1 = torch.sigmoid(a)
    normalized_0 = torch.ones_like(normalized_1) - normalized_1
    normalized = torch.stack([normalized_0, normalized_1], dim=4)
    pd = Categorical(probs=normalized)
    action = pd.sample()
    b = pd.probs
    log = pd.log_prob(action)
    print()

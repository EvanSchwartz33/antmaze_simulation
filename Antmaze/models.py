import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.critic_value = nn.Linear(64, 1)

      
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        x = self.shared(obs)
        return self.actor_mean(x), self.critic_value(x)

    def act(self, obs):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze()

    def evaluate(self, obs, action):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze()

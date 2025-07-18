import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, size, obs_dim, action_dim, gamma=0.99, lam=0.95, device='cpu'):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.size, "Buffer overflow!"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_value=0.0):
        
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)

        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = np.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * gae
            adv[t] = gae
        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = adv + self.val_buf[path_slice]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return all data as torch tensors and normalize advantages.
        """
        assert self.ptr == self.size, "Buffer not full yet!"
        self.ptr, self.path_start_idx = 0, 0
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return {
            'obs': torch.tensor(self.obs_buf, dtype=torch.float32).to(self.device),
            'act': torch.tensor(self.act_buf, dtype=torch.float32).to(self.device),
            'ret': torch.tensor(self.ret_buf, dtype=torch.float32).to(self.device),
            'adv': torch.tensor(self.adv_buf, dtype=torch.float32).to(self.device),
            'logp': torch.tensor(self.logp_buf, dtype=torch.float32).to(self.device),
        }

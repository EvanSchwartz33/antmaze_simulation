import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class AntMazeDataset(Dataset):
    def __init__(self, path="antmaze_dataset.pt"):
        self.data = torch.load(path, weights_only= False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.tensor(sample['obs'], dtype=torch.float32),
            torch.tensor(sample['action'], dtype=torch.float32),
            torch.tensor(sample['reward'], dtype=torch.float32),
            torch.tensor(sample['next_obs'], dtype=torch.float32),
            torch.tensor(sample['done'], dtype=torch.float32)
        )


dataset = AntMazeDataset("antmaze_dataset.pt")
loader = DataLoader(dataset, batch_size=1, shuffle=False)


x_vals, y_vals, rewards = [], [], []


for obs, action, reward, next_obs, done in loader:
    x_vals.append(obs[0][0].item())  
    y_vals.append(obs[0][1].item())  
    rewards.append(reward.item())


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1)
plt.title("Agent Trajectory (x vs y)")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.grid(True)
plt.axis("equal")


plt.subplot(1, 2, 2)
plt.plot(rewards, color='green')
plt.title("Reward per Step")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.grid(True)

plt.tight_layout()
plt.show()

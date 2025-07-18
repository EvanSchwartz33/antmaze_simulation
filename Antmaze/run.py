import torch
import gymnasium as gym
import gymnasium_robotics
from models import ActorCritic
from trainer import train_ppo
import numpy as np

def evaluate_policy(env_id, model_path, episodes=3, render=True, device="cpu"):
    env = gym.make(env_id, render_mode="human" if render else None)
    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        obs = obs_dict["observation"]
        achieved_pos = obs_dict["achieved_goal"]
        done = False
        total_reward = 0
        steps = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = model.act(obs_tensor)
        action_np = action.cpu().numpy().flatten()

        next_obs_dict, _, terminated, truncated, _ = env.step(action_np)
        next_obs = next_obs_dict["observation"]
        next_pos = next_obs_dict["achieved_goal"]

        
        reward_forward = obs[13]
        reward_alive = 1.0 if obs[0] > 0.5 else -1.0
        reward_ctrl = -0.001 * np.sum(np.square(action_np))
        ang_vel = obs[16:19]
        reward_stability = -0.01 * np.linalg.norm(ang_vel)

        

        custom_rew = next_pos[0] - achieved_pos[0] + (1.0 * reward_forward + 2 * reward_alive + 0.001 * reward_ctrl + 0.01 * reward_stability)
        total_reward += custom_rew

        obs = next_obs
        achieved_pos = next_pos
        done = terminated or truncated
        steps += 1

    print(f"Episode {ep+1}: reward = {total_reward:.2f}, steps = {steps}")


    env.close()

if __name__ == "__main__":
    ENV_ID = "AntMaze_UMaze-v5"
    MODEL_PATH = "ppo_antmaze.pt"
    DEVICE = "cpu"  # or "cuda" if using GPU

    train_ppo(env_id=ENV_ID, total_steps=15000000, device=DEVICE)
    evaluate_policy(ENV_ID, MODEL_PATH, episodes=3, render=True, device=DEVICE)

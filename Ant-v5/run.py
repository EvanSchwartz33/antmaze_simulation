import torch
import gymnasium as gym
from models import ActorCritic
from trainer import train_ppo
import numpy as np
from custom_reward import compute_custom_reward

def evaluate_policy(env_id, model_path, episodes=3, render=True, device="cpu"):
    env = gym.make(env_id, render_mode="human" if render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        prev_x = env.unwrapped.data.qpos[0]

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = model.act(obs_tensor)
            action_np = action.cpu().numpy().flatten()

            next_obs, _, terminated, truncated, _ = env.step(action_np)
            current_x = env.unwrapped.data.qpos[0]
            current_y = env.unwrapped.data.qpos[1]
            delta_x = current_x - prev_x

            custom_rew = compute_custom_reward(obs, next_obs, action_np, prev_x, current_x, prev_y, current_y)

            total_reward += custom_rew
            obs = next_obs
            prev_x = current_x
            prev_y = current_y
            done = terminated or truncated
            steps += 1

        print(f"Episode {ep+1}: reward = {total_reward:.2f}, steps = {steps}")
    env.close()

if __name__ == "__main__":
    ENV_ID = "Ant-v5"
    MODEL_PATH = "ppo_ant.pt"
    DEVICE = "cpu"

    train_ppo(env_id=ENV_ID, total_steps=10000000, device=DEVICE)
    evaluate_policy(ENV_ID, MODEL_PATH, episodes=3, render=True, device=DEVICE)

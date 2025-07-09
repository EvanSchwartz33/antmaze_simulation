import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import pickle
from models import ActorCritic

def run_and_save():
    env = gym.make("AntMaze_UMaze-v5", render_mode="human")
    dataset = []

    obs_dict, _ = env.reset()
    obs = obs_dict["observation"]
    goal = obs_dict["desired_goal"]
    achieved_pos = obs_dict["achieved_goal"]

    obs_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]

    policy_net = ActorCritic(obs_dim, action_dim)
    policy_net.load_state_dict(torch.load("ppo_antmaze.pt"))
    policy_net.eval()

    def policy(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy_net.act(obs_tensor)
        return action.squeeze(0).numpy()

    def reward_fn(achieved_pos, next_pos, obs):
        forward_reward = next_pos[0] - achieved_pos[0]
        torso_z_reward = 1.0 if obs[0] > 0.4 else -1.0
        return forward_reward + torso_z_reward

    for episode in range(10):
        obs_dict, _ = env.reset()
        obs = obs_dict["observation"]
        achieved_pos = obs_dict["achieved_goal"]
        done = False
        steps = 0

        while not done and steps < 1000:
            action = policy(obs)
            next_obs_dict, _, terminated, truncated, _ = env.step(action)
            next_obs = next_obs_dict["observation"]
            next_pos = next_obs_dict["achieved_goal"]

            custom_reward = reward_fn(achieved_pos, next_pos, obs)
            done_flag = terminated or truncated

            dataset.append({
                'obs': obs,
                'action': action,
                'reward': custom_reward,
                'next_obs': next_obs,
                'done': done_flag,
                'position': achieved_pos,
                'next_position': next_pos,
                'goal': goal
            })

            obs = next_obs
            achieved_pos = next_pos
            done = done_flag
            steps += 1

    env.close()
    torch.save(dataset, "antmaze_dataset.pt")
    print(f"✅ Saved {len(dataset)} transitions to antmaze_dataset.pt")

    with open("ant_policy.pkl", "wb") as f:
        pickle.dump(policy_net.state_dict(), f)
    print("✅ Saved policy weights to ant_policy.pkl")


if __name__ == "__main__":
    run_and_save()

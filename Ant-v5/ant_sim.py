
import gymnasium as gym
import numpy as np
import torch
import pickle
from models import ActorCritic

def run_and_save():
    env = gym.make("Ant-v5", render_mode="human")
    dataset = []

    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]

    policy_net = ActorCritic(obs_dim, action_dim)
    policy_net.load_state_dict(torch.load("ppo_ant.pt"))
    policy_net.eval()

    def policy(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy_net.act(obs_tensor)
        return action.squeeze(0).numpy()

    def reward_fn(obs, next_obs, action):
        z_height = obs[0]
        next_z_height = next_obs[0]
        pitch = obs[2]
        roll = obs[3]
        ang_vel = obs[16:19]

        delta_x = next_obs[13] - obs[13]  # crude proxy for forward delta (since we don't store qpos directly)
        reward_forward = delta_x
        reward_ctrl = -0.001 * np.sum(np.square(action))
        reward_stability = -0.05 * np.linalg.norm(ang_vel)
        tilt_penalty = -0.1 * (pitch ** 2 + roll ** 2)
        upright_bonus = np.clip((z_height - 0.2) * 5.0, 0.0, 1.0)
        reward_alive = upright_bonus * 1.0 if next_z_height > 0.2 else -2.0
        upright_next = np.clip((next_z_height - 0.2) * 5.0, 0.0, 1.0)
        upright_progress = upright_next - upright_bonus
        reward_recovery = 2.0 * upright_progress if upright_progress > 0 else 0.0

        return (reward_forward + reward_alive + reward_recovery +
                0.05 * reward_ctrl + 0.05 * reward_stability + tilt_penalty)

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 10000:
            action = policy(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done_flag = terminated or truncated

            custom_reward = reward_fn(obs, next_obs, action)

            dataset.append({
                'obs': obs,
                'action': action,
                'reward': custom_reward,
                'next_obs': next_obs,
                'done': done_flag
            })

            obs = next_obs
            done = done_flag
            steps += 1

    env.close()
    torch.save(dataset, "ant_dataset.pt")
    print(f"✅ Saved {len(dataset)} transitions to ant_dataset.pt")

    with open("ant_policy.pkl", "wb") as f:
        pickle.dump(policy_net.state_dict(), f)
    print("✅ Saved policy weights to ant_policy.pkl")

if __name__ == "__main__":
    run_and_save()

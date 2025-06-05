import gymnasium as gym
import numpy as np
import torch

# Set up Ant-v4 using gymnasium's Mujoco backend
def run_and_save():
    env = gym.make("Ant-v5", render_mode="human")
    dataset = []
    goal = np.array([1.5, 1.5])

    def reward_fn(pos):
        return -np.linalg.norm(pos - goal)

    def policy(obs):
        return env.action_space.sample()

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 10000:
            action = policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            xpos, ypos = info["x_position"], info["y_position"] if "x_position" in info else (0, 0)
            custom_reward = reward_fn(np.array([xpos, ypos]))
            done_flag = terminated or truncated

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
    torch.save(dataset, "antmaze_dataset.pt")
    print("Saved", len(dataset), "steps to antmaze_dataset.pt")


if __name__ == "__main__":
    run_and_save()

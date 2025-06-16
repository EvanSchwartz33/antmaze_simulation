import gymnasium as gym
import numpy as np
import torch

# Set up Ant-v5 using gymnasium's Mujoco backend
def run_and_save():
    env = gym.make("Ant-v5", render_mode="human")
    dataset = []
    goal = np.array([2.5, 2.5])
    print(env.spec.entry_point)



    def reward_fn(obs, next_obs, info):
        xpos = env.unwrapped.data.qpos[0]
        ypos = env.unwrapped.data.qpos[1]
        dist_from_goal = np.linalg.norm(np.array([xpos, ypos]) - goal)

        reward = -dist_from_goal

        torso_z = obs[0]

        reward += 0.5 * torso_z

        return reward


    def policy(obs):
        return env.action_space.sample()

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 10000:
            action = policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            xpos = env.unwrapped.data.qpos[0]
            ypos = env.unwrapped.data.qpos[1]

            custom_reward = reward_fn(obs, next_obs, info)
            done_flag = terminated or truncated

            dataset.append({
                'obs': obs,
                'action': action,
                'reward': custom_reward,
                'next_obs': next_obs,
                'done': done_flag,
                'position': (xpos, ypos)
            })

            obs = next_obs
            done = done_flag
            steps += 1
    
    env.close()
    torch.save(dataset, "antmaze_dataset.pt")
    print("Saved", len(dataset), "steps to antmaze_dataset.pt")


if __name__ == "__main__":
    run_and_save()

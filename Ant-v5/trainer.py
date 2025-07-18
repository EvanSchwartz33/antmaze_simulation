import torch
import gymnasium as gym
import numpy as np
from models import ActorCritic
from buffer import RolloutBuffer
import matplotlib.pyplot as plt
from custom_reward import compute_custom_reward

def train_ppo(env_id="Ant-v5", total_steps=5000000, steps_per_epoch=8192, epochs=15,
              minibatch_size=64, gamma=0.99, lam=0.95, clip_ratio=0.2,
              pi_lr=3e-4, vf_lr=1e-3, device='cpu'):

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)

    try:
        model.load_state_dict(torch.load("ppo_ant.pt"))
        print("Loaded existing policy from ppo_ant.pt")
    except FileNotFoundError:
        print("No existing policy found. Starting from scratch.")

    buffer = RolloutBuffer(steps_per_epoch, obs_dim, act_dim, gamma, lam, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pi_lr)

    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0
    prev_x = env.unwrapped.data.qpos[0]
    prev_y = env.unwrapped.data.qpos[1]
    episode_rewards = []

    for t in range(total_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            act, logp, val = model.act(obs_tensor)
        act_np = act.cpu().numpy().flatten()
        next_obs, _, done, trunc, _ = env.step(act_np)
        current_x = env.unwrapped.data.qpos[0]
        current_y = env.unwrapped.data.qpos[1]
        

    

        custom_rew = compute_custom_reward(obs, next_obs, act_np, prev_x, current_x, prev_y, current_y)

        buffer.store(obs, act_np, custom_rew, val.item(), logp.item())

        obs = next_obs
        prev_x = current_x
        ep_ret += custom_rew
        ep_len += 1

        terminal = done or trunc or (t + 1) % steps_per_epoch == 0

        if terminal:
            if not (done or trunc):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, val = model.act(obs_tensor)
                last_val = val.item()
            else:
                last_val = 0
            buffer.finish_path(last_val)

            if (t + 1) % steps_per_epoch == 0:
                data = buffer.get()
                for _ in range(epochs):
                    idxs = np.random.permutation(steps_per_epoch)
                    for start in range(0, steps_per_epoch, minibatch_size):
                        end = start + minibatch_size
                        mb_idx = idxs[start:end]

                        obs_b = data['obs'][mb_idx]
                        act_b = data['act'][mb_idx]
                        ret_b = data['ret'][mb_idx]
                        adv_b = data['adv'][mb_idx]
                        logp_old_b = data['logp'][mb_idx]

                        logp, entropy, value = model.evaluate(obs_b, act_b)
                        ratio = torch.exp(logp - logp_old_b)

                        clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_b
                        actor_loss = -torch.min(ratio * adv_b, clipped).mean()
                        critic_loss = ((ret_b - value) ** 2).mean()
                        entropy_bonus = entropy.mean()

                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                torch.save(model.state_dict(), "ppo_ant.pt")
                print(f"Saved model at step {t + 1}")

            episode_rewards.append(ep_ret)
            obs, _ = env.reset()


            #forces robot to start flipped half the time to increase training difficulty
            if np.random.rand() < 0.5:
                qpos = env.unwrapped.data.qpos.copy()
                qpos[2] = 0.15  # low height
                qpos[3:7] = np.random.uniform(-1, 1, size=4)
                qpos[3:7] /= np.linalg.norm(qpos[3:7])  
                env.unwrapped.set_state(qpos, env.unwrapped.data.qvel)

            prev_x = env.unwrapped.data.qpos[0]
            ep_ret, ep_len = 0, 0

    env.close()
    plt.plot(episode_rewards)
    plt.title("Episode Reward over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("reward_curve.png")
    plt.show()

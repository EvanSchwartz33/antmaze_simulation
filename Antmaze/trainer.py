import torch
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from models import ActorCritic
from buffer import RolloutBuffer

def train_ppo(env_id="AntMaze_UMaze-v5", total_steps=1000, steps_per_epoch=8192, epochs=15,
              minibatch_size=64, gamma=0.99, lam=0.95, clip_ratio=0.2,
              pi_lr=3e-4, vf_lr=1e-3, device='cpu'):

    env = gym.make(env_id)
    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)

    try:
        model.load_state_dict(torch.load("ppo_antmaze.pt"))
        print("Loaded existing policy from ppo_antmaze.pt")
    except FileNotFoundError:
        print("No existing policy found. Starting from scratch.")

    buffer = RolloutBuffer(steps_per_epoch, obs_dim, act_dim, gamma, lam, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pi_lr)

    obs_dict, _ = env.reset()
    obs = obs_dict["observation"]
    ep_ret, ep_len = 0, 0

    for t in range(total_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            act, logp, val = model.act(obs_tensor)
        act_np = act.cpu().numpy().flatten()
        next_obs_dict, _, done, trunc, _ = env.step(act_np)
        next_obs = next_obs_dict["observation"]

        achieved_pos = obs_dict["achieved_goal"]
        next_pos = next_obs_dict["achieved_goal"]
        """custom_rew = 0

        torso_z = obs[0]
        if obs[0] < 0.3:
            z_reward = -1
        else:
            z_reward = 1

        upright_quat_w = obs[4]

        if upright_quat_w > 0.8:
            custom_rew += 10
        elif upright_quat_w < 0.3:
            custom_rew -= 10
        else:
            custom_rew += 0"""
        
        
        #energy_penalty = 0.001 * np.square(act_np).sum()


        z_height = obs[0]
        next_z_height = next_obs[0]
        pitch = obs[2]
        roll = obs[3]
        ang_vel = obs[16:19]
        
        
        tilt_penalty = -0.1 * (pitch ** 2 + roll ** 2)
        reward_forward = obs[13]
        reward_ctrl = -0.001 * np.sum(np.square(act_np))
        reward_stability = -0.05 * np.linalg.norm(ang_vel)
        upright_bonus = np.clip((z_height - 0.2) * 5.0, 0.0, 1.0)
        reward_alive = upright_bonus * 1.0 if next_z_height > 0.2 else -2.0
        upright_next = np.clip((next_z_height - 0.2) * 5.0, 0.0, 1.0)
        upright_progress = upright_next - upright_bonus
        reward_recovery = 2.0 * upright_progress if upright_progress > 0 else 0.0

        custom_rew = (abs(next_pos[0] - achieved_pos[0]) + reward_forward + reward_alive + reward_recovery + 0.05 * reward_ctrl + 0.05 * reward_stability + tilt_penalty)
        
        
        


        rew = custom_rew  # override the environment reward
        print(f"Stored reward: {rew:.3f}")
        buffer.store(obs, act_np, rew, val.item(), logp.item())
        

        obs = next_obs
        ep_ret += rew
        ep_len += 1

        terminal = done or trunc or (t+1) % steps_per_epoch == 0

        if terminal:
            if not (done or trunc):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, val = model.act(obs_tensor)
                last_val = val.item()
            else:
                last_val = 0
            buffer.finish_path(last_val)

            if (t+1) % steps_per_epoch == 0:
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

            obs_dict, _ = env.reset()
            obs = obs_dict["observation"]
            ep_ret, ep_len = 0, 0

            torch.save(model.state_dict(), "ppo_antmaze.pt")

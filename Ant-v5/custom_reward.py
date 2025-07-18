import numpy as np

def compute_custom_reward(obs, next_obs, action, prev_x, current_x, prev_y, current_y):
    # --- Inputs ---
    z = obs[0]                  # torso height
    z_next = next_obs[0]
    pitch = obs[2]
    roll = obs[3]
    ang_vel = obs[16:19]

    # --- 1. Standing Up ---
    # max ~1.0 when torso is upright and high
    upright = np.clip((z - 0.2) * 2.5, 0.0, 1.0)
    upright_next = np.clip((z_next - 0.2) * 2.5, 0.0, 1.0)
    standing_bonus = 2 if z > 0.5 else 0

    # --- 2. Recovery Bonus ---
    # bonus if torso height increases toward upright
    recovery_progress = upright_next - upright
    recovery_bonus = 5 * recovery_progress

    # --- 3. Forward Progress ---
    delta_x = current_x - prev_x   
    delta_y = current_y - prev_y
    forward_bonus = 2.0 * (abs(delta_x + abs(delta_y))) if z_next > 0 else -3 * (abs(delta_x + abs(delta_y)))

    # --- 4. Penalties ---
    ctrl_penalty = -0.01 * np.sum(np.square(action))
    instability_penalty = -0.05 * np.linalg.norm(ang_vel)
    tilt_penalty = -0.1 * (pitch ** 2 + roll ** 2)

    # --- Combine ---
    total_reward = (
        standing_bonus +
        recovery_bonus +
        forward_bonus +
        ctrl_penalty +
        instability_penalty +
        tilt_penalty
    )
    return total_reward

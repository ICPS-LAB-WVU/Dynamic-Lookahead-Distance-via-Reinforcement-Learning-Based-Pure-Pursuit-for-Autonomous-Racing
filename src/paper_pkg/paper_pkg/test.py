#!/usr/bin/env python3
# Filename: test_ppo_model.py

import os
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from paper_pkg.pure_env import F1TenthEnv

def make_env():
    """Factory to create monitored environment."""
    env = F1TenthEnv()
    return Monitor(env)

def extract_obs(obs):
    """Handle obs shape for both VecEnv and raw env."""
    if hasattr(obs, "shape") and obs.ndim == 2:
        return float(obs[0][0]), float(obs[0][1])
    elif hasattr(obs, "__len__") and len(obs) == 2:
        return float(obs[0]), float(obs[1])
    else:
        raise ValueError(f"Unexpected obs format: {obs}")

def extract_action(action):
    """Safely extract scalar from PPO action output."""
    if hasattr(action, "__len__") and len(action) > 0:
        return float(action[0])
    return float(action)

def main():
    # === Start ROS2 only if not already initialized ===
    if not rclpy.ok():
        rclpy.init()

    # === Create vectorized env ===
    env = DummyVecEnv([make_env])

    # === Load VecNormalize stats if present ===
    vec_norm_path = os.path.join("best_model", "vecnormalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ Loaded VecNormalize stats.")
    else:
        print("⚠ No VecNormalize stats found, running without normalization.")

    # === Load PPO model ===
    model_path = "ppo_lookahead_model"
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"❌ Model file '{model_path}.zip' not found!")
    model = PPO.load(model_path, env=env)

    # === Run test episodes ===
    num_episodes = 5
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        print(f"\n=== Starting Episode {ep+1} ===")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Extract safe values
            speed, curvature = extract_obs(obs)
            action_value = extract_action(action)
            reward_value = float(reward[0]) if hasattr(reward, "__len__") else float(reward)

            print(f"[Step {steps}] Action={action_value:.2f}, "
                  f"Speed={speed:.2f}, Curvature={curvature:.3f}, Reward={reward_value:.2f}")

            total_reward += reward_value
            steps += 1

        # === End-of-episode reason logging (if env provides it) ===
        reason = "Unknown"
        if isinstance(info, list) and info:
            info = info[0]
        if isinstance(info, dict):
            if info.get("collision"):
                reason = "Collision"
            elif info.get("stalled"):
                reason = "Vehicle stalled"
            elif info.get("lap_completed"):
                reason = "Lap completed"
        elif steps >= getattr(env.envs[0].unwrapped, "max_steps", steps):
            reason = "Max steps reached"

        print(f"✅ Episode {ep+1} finished. Total Reward={total_reward:.2f}, "
              f"Steps={steps}, End reason={reason}")

    # === Proper cleanup ===
    env.close()  # Calls F1TenthEnv.close(), which handles ROS shutdown
    if rclpy.ok():  # Avoid double shutdown
        rclpy.shutdown()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from paper_pkg.pure_env import F1TenthEnv

from stable_baselines3.common.monitor import Monitor as OriginalMonitor
import gym

class Monitor(OriginalMonitor):
    def __init__(self, env, *args, **kwargs):
        # override the gymnasium.core.Env check with classic gym.Env
        if not isinstance(env, gym.Env):
            raise ValueError("Expected gym.Env, got something else")
        super().__init__(env, *args, **kwargs)

def main():
    # === 1. Create training and evaluation environments ===
    train_env = DummyVecEnv([lambda: Monitor(ControllerArbiterEnv())])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([lambda: Monitor(ControllerArbiterEnv())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    # === 2. Set up evaluation callback ===
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # === 3. Initialize PPO model ===
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=lambda f: 3e-4 * f,  # f decreases from 1 → 0  #1e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.92,
        clip_range=0.15,
        ent_coef=0.001,               # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_logs"
    )

    # === 4. Train the model ===
    model.learn(total_timesteps=500_000, callback=eval_callback)
    model.save("ppo_lookahead_model")

    print("✅ Training completed and final model saved!")

    # === 5. Evaluate final model in a rollout ===
    obs = train_env.reset()
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _ = train_env.step(action)
        if done:
            obs = train_env.reset()

    train_env.close()
    eval_env.close()

if __name__ == '__main__':
    main()

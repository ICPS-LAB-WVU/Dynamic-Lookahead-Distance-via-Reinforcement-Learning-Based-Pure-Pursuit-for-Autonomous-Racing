#!/usr/bin/env python3

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from paper_pkg.pure_env import F1TenthEnv

def evaluate_model(model, env, n_episodes=3):
    total_reward = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / n_episodes

def objective(trial):
    # === 1. PPO hyperparameter tuning ===
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    n_epochs = trial.suggest_int("n_epochs", 5, 20)

    # === 2. Create training environment ===
    train_env = DummyVecEnv([lambda: F1TenthEnv()])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # === 3. Create evaluation environment (freeze normalization stats) ===
    eval_env = DummyVecEnv([lambda: F1TenthEnv()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # === 4. Initialize PPO ===
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=4096,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_logs"
    )

    # === 5. Train model ===
    model.learn(total_timesteps=500_000, callback=eval_callback)
    
    # === 6. Save model and normalization stats ===
    model.save("ppo_lookahead_model")
    train_env.save("vec_normalize.pkl")
    print("✅ Training completed and model saved!")

    # === 7. Evaluate ===
    mean_reward = evaluate_model(model, eval_env)

    train_env.close()
    eval_env.close()

    return mean_reward

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\n✅ Best trial:")
    print(f"  Value (avg reward): {study.best_trial.value:.2f}")
    print("  Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"    '{key}': {value:.4f},")

if __name__ == "__main__":
    main()


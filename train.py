import os
import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


class EpisodeLoggerCallback(BaseCallback):
    """
    Saves episode reward/length during training to a CSV file.
    Works best when env is wrapped with Monitor/VecMonitor.
    """
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self.rows.append({
                    "timesteps": self.num_timesteps,
                    "episode_reward": ep["r"],
                    "episode_length": ep["l"],
                    "time": round(time.time(), 2),
                })
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timesteps", "episode_reward", "episode_length", "time"]
            )
            writer.writeheader()
            writer.writerows(self.rows)


def linear_schedule(start_value: float, end_value: float, end_fraction: float):
    """
    SB3 DQN accepts exploration_fraction, exploration_initial_eps, exploration_final_eps.
    This helper is only for recording config meaning clearly in metadata.
    """
    return {
        "epsilon_start": start_value,
        "epsilon_end": end_value,
        "epsilon_decay_fraction": end_fraction
    }


def make_cnn_env(env_id: str, n_envs: int, seed: int, render_mode=None):
    """
    For CNNPolicy: image-based Atari env + frame stacking.
    SB3 Atari helper handles preprocessing.
    """
    env = make_atari_env(env_id, n_envs=n_envs, seed=seed, env_kwargs={"render_mode": render_mode})
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def make_mlp_env(env_id: str, seed: int, render_mode=None):
    """
    For MLPPolicy: use RAM observations.
    We build ALE/Tennis-ram-v5 from ALE/Tennis-v5 automatically.
    """
    ram_env_id = env_id.replace("-v5", "-ram-v5")
    env = gym.make(ram_env_id, render_mode=render_mode)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="General DQN training script for Atari Tennis experiments")
    parser.add_argument("--member", type=str, default="Best", help="Member name")
    parser.add_argument("--experiment", type=str, default="exp_01", help="Experiment id")
    parser.add_argument("--env-id", type=str, default="ALE/Tennis-v5", help="Gymnasium Atari env id")
    parser.add_argument("--policy", type=str, choices=["CnnPolicy", "MlpPolicy"], default="CnnPolicy")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=50000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4, help="Used for CNN Atari env")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path("results") / args.member / args.experiment
    model_dir = base_dir / "models"
    log_dir = base_dir / "logs"
    best_model_dir = base_dir / "best_model"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "member": args.member,
        "experiment": args.experiment,
        "env_id": args.env_id,
        "policy": args.policy,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "target_update_interval": args.target_update_interval,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "epsilon": linear_schedule(
            args.exploration_initial_eps,
            args.exploration_final_eps,
            args.exploration_fraction
        )
    }

    with open(base_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.policy == "CnnPolicy":
        train_env = make_cnn_env(args.env_id, n_envs=args.n_envs, seed=args.seed)
        eval_env = make_cnn_env(args.env_id, n_envs=1, seed=args.seed + 100)
    else:
        train_env = make_mlp_env(args.env_id, seed=args.seed)
        eval_env = make_mlp_env(args.env_id, seed=args.seed + 100)

    episode_logger = EpisodeLoggerCallback(str(log_dir / "training_log.csv"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = DQN(
        policy=args.policy,
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        verbose=1,
        tensorboard_log=str(log_dir / "tb"),
        seed=args.seed,
        device=args.device,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[episode_logger, eval_callback],
        progress_bar=True
    )

    final_model_path = model_dir / "dqn_model"
    model.save(str(final_model_path))

    print(f"\nTraining finished.")
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"Best model folder: {best_model_dir}")
    print(f"Logs saved in: {log_dir}")


if __name__ == "__main__":
    main()
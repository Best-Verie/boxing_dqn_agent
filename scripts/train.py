#!/usr/bin/env python
# coding: utf-8

"""
Reusable DQN training script for Atari experiments.
Train ONE experiment at a time and save:
- final model
- best checkpoint during training
- training CSV
- config JSON
- evaluation JSON
"""

import os
import csv
import json
import time
import argparse
from pathlib import Path

import ale_py  # noqa: F401
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor


class EpisodeCSVLogger(BaseCallback):
    def __init__(self, csv_path: str, progress_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.progress_every = max(1, int(progress_every))
        self.rows = []
        self._last_progress_t = 0
        self._last_episode_reward = None
        self._last_episode_length = None

    def _on_training_start(self) -> None:
        self._last_progress_t = 0

    def _on_step(self) -> bool:
        saw_episode = False
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self._last_episode_length = ep["l"]
                self._last_episode_reward = ep["r"]
                saw_episode = True
                self.rows.append(
                    {
                        "timestep": self.num_timesteps,
                        "ep_length": ep["l"],
                        "ep_reward": ep["r"],
                        "time": round(time.time(), 2),
                        "row_type": "episode_end",
                    }
                )

        # Add heartbeat points for long episodes so plotting never looks empty.
        if (not saw_episode) and (self.num_timesteps - self._last_progress_t) >= self.progress_every:
            self._last_progress_t = self.num_timesteps
            self.rows.append(
                {
                    "timestep": self.num_timesteps,
                    "ep_length": self._last_episode_length,
                    "ep_reward": self._last_episode_reward,
                    "time": round(time.time(), 2),
                    "row_type": "progress",
                }
            )
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            # Keep at least one row so downstream plotting logic has a file to read.
            self.rows.append(
                {
                    "timestep": self.num_timesteps,
                    "ep_length": self._last_episode_length,
                    "ep_reward": self._last_episode_reward,
                    "time": round(time.time(), 2),
                    "row_type": "summary",
                }
            )
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestep", "ep_length", "ep_reward", "time", "row_type"],
            )
            writer.writeheader()
            writer.writerows(self.rows)


def append_eval_summary_row(csv_path: Path, timestep: int, mean_reward: float) -> None:
    default_fields = ["timestep", "ep_length", "ep_reward", "time", "row_type"]
    row = {
        "timestep": int(timestep),
        "ep_length": "",
        "ep_reward": float(mean_reward),
        "time": round(time.time(), 2),
        "row_type": "eval_summary",
    }

    fieldnames = list(default_fields)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
                if existing_header:
                    fieldnames = existing_header
        except Exception:
            fieldnames = list(default_fields)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def make_cnn_env(env_id: str, seed: int, render_mode=None):
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": render_mode},
    )
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def make_mlp_env(env_id: str, seed: int, render_mode=None):
    candidate_ids = list(
        dict.fromkeys(
            [
                env_id,
                env_id.replace("-v5", "-ram-v5"),
                "ALE/Tennis-ram-v5",
                "Tennis-ram-v5",
            ]
        )
    )

    last_err = None
    for ram_env_id in candidate_ids:
        try:
            env = DummyVecEnv(
                [
                    lambda ram_env_id=ram_env_id, render_mode=render_mode: Monitor(
                        gym.make(
                            ram_env_id,
                            obs_type="ram",
                            frameskip=4,
                            repeat_action_probability=0.0,
                            render_mode=render_mode,
                        )
                    )
                ]
            )
            env.seed(seed)
            env = VecMonitor(env)
            return env
        except Exception as err:
            last_err = err

    raise RuntimeError(
        "Could not create a RAM Atari env for MlpPolicy. "
        f"Tried ids={candidate_ids}. Last error: {last_err}"
    ) from last_err


def parse_args():
    parser = argparse.ArgumentParser(description="Train one DQN Atari experiment")
    parser.add_argument("--member", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="ALE/Boxing-v5")
    parser.add_argument("--policy", type=str, choices=["CnnPolicy", "MlpPolicy"], default="CnnPolicy")
    parser.add_argument("--total-timesteps", type=int, default=100_000) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=2000) 
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-progress-every", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path("results") / args.member
    model_dir = base_dir / "models"
    log_dir = base_dir / "logs"
    table_dir = base_dir / "tables"
    experiment_dir = base_dir / args.experiment

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / f"{args.experiment}_config.json"
    training_csv_path = log_dir / f"{args.experiment}_training_metrics.csv"
    eval_json_path = experiment_dir / f"{args.experiment}_eval.json"
    final_model_path = model_dir / f"{args.experiment}.zip"
    best_model_dir = model_dir / f"{args.experiment}_best"
    eval_log_dir = experiment_dir / "eval_logs"

    config = {
        "member": args.member,
        "experiment": args.experiment,
        "env_id": args.env_id,
        "policy": args.policy,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "target_update_interval": args.target_update_interval,
        "exploration_initial_eps": args.exploration_initial_eps,
        "exploration_final_eps": args.exploration_final_eps,
        "exploration_fraction": args.exploration_fraction,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.policy == "CnnPolicy":
        train_env = make_cnn_env(args.env_id, seed=args.seed)
        eval_env = make_cnn_env(args.env_id, seed=args.seed + 100)
        final_eval_env = make_cnn_env(args.env_id, seed=args.seed + 200)
    else:
        train_env = make_mlp_env(args.env_id, seed=args.seed)
        eval_env = make_mlp_env(args.env_id, seed=args.seed + 100)
        final_eval_env = make_mlp_env(args.env_id, seed=args.seed + 200)

    episode_logger = EpisodeCSVLogger(
        str(training_csv_path),
        progress_every=args.log_progress_every,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = DQN(
        policy=args.policy,
        env=train_env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        tensorboard_log=str(log_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    print("\n[START TRAINING]")
    print(json.dumps(config, indent=2))

    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[episode_logger, eval_callback],
        progress_bar=False,
    )
    train_minutes = (time.time() - start_time) / 60.0

    model.save(str(final_model_path.with_suffix("")))

    mean_reward, std_reward = evaluate_policy(
        model,
        final_eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    # Ensure each experiment has at least one numeric reward point for plotting.
    append_eval_summary_row(training_csv_path, args.total_timesteps, float(mean_reward))

    eval_summary = {
        "member": args.member,
        "experiment": args.experiment,
        "policy": args.policy,
        "env_id": args.env_id,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "train_minutes": float(train_minutes),
        "model_path": str(final_model_path),
        "best_model_path": str(best_model_dir / "best_model.zip"),
        "training_csv_path": str(training_csv_path),
        "config_path": str(config_path),
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "target_update_interval": args.target_update_interval,
        "exploration_initial_eps": args.exploration_initial_eps,
        "exploration_final_eps": args.exploration_final_eps,
        "exploration_fraction": args.exploration_fraction,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
    }

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n[END TRAINING]")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best checkpoint folder: {best_model_dir}")
    print(f"Training CSV: {training_csv_path}")
    print(f"Eval summary JSON: {eval_json_path}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Train minutes: {train_minutes:.2f}")

    train_env.close()
    eval_env.close()
    final_eval_env.close()


if __name__ == "__main__":
    main()

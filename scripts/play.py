
#!/usr/bin/env python
# coding: utf-8

"""
Play the known best trained Atari DQN model.

Usage example:
python play.py \
  --model-path results/BestVerie/models/boxing_exp01_baseline_cnn.zip \
  --env-id ALE/Boxing-v5 \
  --policy CnnPolicy \
  --episodes 3 \
  --render-mode rgb_array \
  --record

For MLP/RAM example:
python play.py \
  --model-path results/BestVerie/models/boxing_exp10_baseline_mlp.zip \
  --env-id ALE/Boxing-v5 \
  --policy MlpPolicy \
  --episodes 3 \
  --render-mode rgb_array \
  --record
"""

import argparse
from pathlib import Path

import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

gym.register_envs(ale_py)


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained Atari DQN model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .zip model")
    parser.add_argument("--env-id", type=str, required=True, help="Environment ID, e.g. ALE/Boxing-v5")
    parser.add_argument("--policy", type=str, choices=["CnnPolicy", "MlpPolicy"], required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array", "None"])
    parser.add_argument("--record", action="store_true", help="Record gameplay video")
    return parser.parse_args()


def normalize_render_mode(render_mode: str):
    return None if render_mode == "None" else render_mode


def make_cnn_env(env_id: str, seed: int, render_mode=None, video_folder=None):
    def _make():
        env = gym.make(env_id, render_mode=render_mode)
        if video_folder:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda episode_id: True,
                disable_logger=True,
            )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([_make])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def make_mlp_env(env_id: str, seed: int, render_mode=None, video_folder=None):
    def _make():
        env = gym.make(env_id, obs_type="ram", render_mode=render_mode)
        if video_folder:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda episode_id: True,
                disable_logger=True,
            )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([_make])
    env = VecMonitor(env)
    return env


def main():
    args = parse_args()
    render_mode = normalize_render_mode(args.render_mode)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    video_folder = None
    if args.record:
        video_folder = Path("play_videos")
        video_folder.mkdir(parents=True, exist_ok=True)

    if args.policy == "CnnPolicy":
        env = make_cnn_env(
            env_id=args.env_id,
            seed=args.seed,
            render_mode=render_mode,
            video_folder=str(video_folder) if video_folder else None,
        )
    else:
        env = make_mlp_env(
            env_id=args.env_id,
            seed=args.seed,
            render_mode=render_mode,
            video_folder=str(video_folder) if video_folder else None,
        )

    model = DQN.load(str(model_path), env=env)

    print("[PLAYING MODEL]")
    print(f"Model path: {model_path}")
    print(f"Environment: {args.env_id}")
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Render mode: {render_mode}")

    obs = env.reset()
    episode_rewards = []
    current_reward = 0.0
    finished = 0

    while finished < args.episodes:
        # deterministic=True = greedy action selection
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        current_reward += float(rewards[0])

        if bool(dones[0]):
            episode_rewards.append(current_reward)
            print(f"Episode {finished + 1} reward: {current_reward:.2f}")
            current_reward = 0.0
            finished += 1

    env.close()

    print("\n[PLAY COMPLETE]")
    print("Episode rewards:", episode_rewards)
    if episode_rewards:
        print(f"Average reward: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
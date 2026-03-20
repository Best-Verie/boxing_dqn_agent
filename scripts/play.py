#!/usr/bin/env python
# coding: utf-8

"""
Load a trained DQN model and run evaluation episodes in the same Atari environment.
Supports greedy play, optional GUI rendering, and optional video recording.
"""

import argparse
import time
from pathlib import Path

import ale_py  # noqa: F401
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder


def make_cnn_env(env_id: str, seed: int, render_mode=None):
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": render_mode},
    )
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
            return env
        except Exception as err:
            last_err = err

    raise RuntimeError(
        "Could not create a RAM Atari env for MlpPolicy. "
        f"Tried ids={candidate_ids}. Last error: {last_err}"
    ) from last_err


def parse_args():
    parser = argparse.ArgumentParser(description="Play with a trained DQN Atari model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to dqn_model.zip or another trained model zip")
    parser.add_argument("--env-id", type=str, default="ALE/Tennis-v5")
    parser.add_argument("--policy", type=str, choices=["CnnPolicy", "MlpPolicy"], default="CnnPolicy")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true", help="Use human rendering when available")
    parser.add_argument("--record-video", action="store_true", help="Record a video of one episode")
    parser.add_argument("--video-dir", type=str, default="results/play_videos")
    parser.add_argument("--video-prefix", type=str, default="dqn_play")
    parser.add_argument("--fps-sleep", type=float, default=0.0, help="Optional delay between steps for human render")
    return parser.parse_args()


def make_env(policy: str, env_id: str, seed: int, render_mode=None):
    if policy == "CnnPolicy":
        return make_cnn_env(env_id=env_id, seed=seed, render_mode=render_mode)
    return make_mlp_env(env_id=env_id, seed=seed, render_mode=render_mode)


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    render_mode = "human" if args.render else ("rgb_array" if args.record_video else None)
    env = make_env(policy=args.policy, env_id=args.env_id, seed=args.seed, render_mode=render_mode)

    video_dir = Path(args.video_dir)
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        env = VecVideoRecorder(
            env,
            video_folder=str(video_dir),
            record_video_trigger=lambda step: step == 0,
            video_length=args.max_steps,
            name_prefix=args.video_prefix,
        )

    model = DQN.load(str(model_path), device=args.device)

    rewards = []
    for ep in range(args.episodes):
        env.seed(args.seed + ep)
        obs = env.reset()
        done = [False]
        step_count = 0
        ep_reward = 0.0

        while not done[0] and step_count < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += float(reward[0])
            step_count += 1
            if args.render and args.fps_sleep > 0:
                time.sleep(args.fps_sleep)

        rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{args.episodes} | steps={step_count} | reward={ep_reward:.2f}")

    env.close()

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Average reward over {len(rewards)} episodes: {mean_reward:.2f}")

    if args.record_video and video_dir.exists():
        mp4_files = sorted(video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if mp4_files:
            print(f"Video saved: {mp4_files[-1]}")


if __name__ == "__main__":
    main()

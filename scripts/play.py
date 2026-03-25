#!/usr/bin/env python
"""
play.py  -  DQN Agent Evaluation on ALE/Boxing-v5
==================================================
Loads a trained DQN model and plays using a Greedy Policy
(exploration_rate = 0.0 -> always picks argmax Q(s,a)).

Usage
-----
  # Local machine - watch live
  python play.py --model dqn_model.zip

  # Headless - no display
  python play.py --model dqn_model.zip --no-render

  # Record gameplay video (works on any machine)
  python play.py --model dqn_model.zip --no-render --record --output videos/boxing_gameplay.avi

Author  : Group (shared script)
Course  : Deep Learning - Formative 3
Env     : ALE/Boxing-v5
"""

import utils.http_session
import utils.transport
import utils.session
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import ale_py  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecTransposeImage


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on ALE/Boxing-v5 using greedy policy."
    )
    parser.add_argument("--model",       type=str,   default="dqn_model.zip")
    parser.add_argument("--episodes",    type=int,   default=5)
    parser.add_argument("--no-render",   action="store_true")
    parser.add_argument("--record",      action="store_true")
    parser.add_argument("--output",      type=str,   default="videos/boxing_gameplay.avi")
    parser.add_argument("--fps",         type=int,   default=30)
    parser.add_argument("--frame-delay", type=float, default=0.03)
    parser.add_argument("--seed",        type=int,   default=0)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────
def make_env(render_mode=None, seed=0):
    kwargs = {"render_mode": render_mode} if render_mode else {}
    env = make_atari_env("ALE/Boxing-v5", n_envs=1, seed=seed, env_kwargs=kwargs)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> DQN:
    path = Path(model_path)

    if not path.exists():
        fallbacks = [
            Path("dqn_model.zip"),
            Path("best_model.zip"),
            *list(Path(".").rglob("best_model.zip")),
            *list(Path(".").rglob("dqn_model.zip")),
        ]
        for fb in fallbacks:
            if fb.exists():
                print(f"[INFO] '{model_path}' not found. Using: {fb}")
                path = fb
                break
        else:
            print(f"\n[ERROR] Could not find model at '{model_path}'.")
            sys.exit(1)

    print(f"Loading model : {path}")
    model = DQN.load(str(path), device="auto")

    # GreedyQPolicy: set epsilon=0 so agent always picks argmax Q(s,a)
    model.exploration_rate = 0.0
    print(f"Policy        : GreedyQPolicy (exploration_rate=0.0 -> argmax Q)")
    print(f"Device        : {model.device}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Save collected frames to video
# ──────────────────────────────────────────────────────────────────────────────
def save_video(frames: list, output_path: str, fps: int):
    if len(frames) == 0:
        print("[WARNING] No frames collected, video not saved.")
        return

    try:
        import cv2
    except ImportError:
        print("[WARNING] opencv-python not installed.")
        print("          Install with: pip install opencv-python")
        return

    # Get real dimensions from actual frame
    h, w = frames[0].shape[:2]
    print(f"Frame size    : {w}x{h}")

    # Make sure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Use XVID codec for .avi - works on all Windows machines
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"[ERROR] Could not open video writer for '{output_path}'.")
        return

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"Video saved   : {output_path}  ({len(frames)} frames)")


# ──────────────────────────────────────────────────────────────────────────────
# Main play loop
# ──────────────────────────────────────────────────────────────────────────────
def play(args):
    model = load_model(args.model)

    # Determine render mode
    if args.record:
        render_mode = "rgb_array"
    elif not args.no_render:
        render_mode = "human"
    else:
        render_mode = None

    # Build environment
    try:
        env = make_env(render_mode=render_mode, seed=args.seed)
    except Exception as e:
        print(f"[WARNING] render_mode='{render_mode}' failed: {e}")
        print("          Falling back to no rendering.")
        env = make_env(render_mode=None, seed=args.seed)
        render_mode = None

    print("\n" + "=" * 55)
    print(f"  Environment : ALE/Boxing-v5")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Render      : {render_mode or 'OFF'}")
    print(f"  Recording   : {'ON -> ' + args.output if args.record else 'OFF'}")
    print("=" * 55)

    all_frames      = []   # collect all frames across episodes
    episode_rewards = []
    episode_lengths = []

    for ep in range(1, args.episodes + 1):
        obs     = env.reset()
        total_r = 0.0
        steps   = 0
        done    = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_r += float(reward[0])
            steps   += 1

            # Collect frame for video
            if args.record and render_mode == "rgb_array":
                frame = env.venv.envs[0].render()
                if frame is not None:
                    all_frames.append(np.array(frame).copy())

            if render_mode == "human":
                time.sleep(args.frame_delay)

        episode_rewards.append(total_r)
        episode_lengths.append(steps)

        label = "WIN" if total_r > 0 else ("DRAW" if total_r == 0 else "LOSS")
        print(
            f"  Episode {ep:>2}/{args.episodes}"
            f"  |  Reward: {total_r:>7.1f}"
            f"  |  Steps: {steps:>6,}"
            f"  |  {label}"
        )

    env.close()

    # Save video after all episodes
    if args.record and len(all_frames) > 0:
        save_video(all_frames, args.output, args.fps)

    # Final summary
    wins   = sum(1 for r in episode_rewards if r > 0)
    draws  = sum(1 for r in episode_rewards if r == 0)
    losses = sum(1 for r in episode_rewards if r < 0)

    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    print(f"  Mean reward   : {np.mean(episode_rewards):>7.2f}")
    print(f"  Std  reward   : {np.std(episode_rewards):>7.2f}")
    print(f"  Best episode  : {np.max(episode_rewards):>7.2f}")
    print(f"  Worst episode : {np.min(episode_rewards):>7.2f}")
    print(f"  Record        : {wins}W / {draws}D / {losses}L")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    play(args)

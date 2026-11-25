"""Watch, score and record a trained model.

Always runs on the plain game with no reward wrapper, so custom bonuses
can't flatter the score. An average of 200+ counts as solved.

By default the network picks its most likely action on every frame. With
--explore it picks actions by chance using its own odds, the way it acts
during training. A reward hack can hide in that difference.

    python watch.py models/ppo_default_s0.zip                 # watch 3 attempts
    python watch.py models/ppo_default_s0.zip --record        # save MP4s to videos/
    python watch.py models/ppo_default_s0.zip --no-render --episodes 20   # score only
    python watch.py models/ppo_hover_bonus_s0.zip --explore --record      # see the cheat
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("model", help="path to a saved .zip model")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--record", action="store_true", help="save each attempt as an MP4 in videos/")
parser.add_argument("--no-render", action="store_true", help="no window, just scores (fast)")
parser.add_argument("--explore", action="store_true", help="pick actions by chance, like in training")
parser.add_argument("--seed", type=int, default=1000, help="seed for the test attempts")
args = parser.parse_args()

name = Path(args.model).stem
if args.explore:
    name = name + "_explore"
if args.record:
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos/{name}", name_prefix=name,
                                   episode_trigger=lambda e: True)
else:
    env = gym.make("LunarLander-v3", render_mode=None if args.no_render else "human")

model = PPO.load(args.model, device="cpu")

scores = []
for episode in range(args.episodes):
    obs, info = env.reset(seed=args.seed + episode)
    score, frames, done = 0.0, 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=not args.explore)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        frames += 1
        done = terminated or truncated
    ending = "landed" if terminated and reward == 100 else "crashed / flew off" if terminated else "ran out of time"
    print(f"Attempt {episode + 1:2d}: score {score:7.1f}  {frames:4d} frames  {ending}")
    scores.append(score)

env.close()
mean = np.mean(scores)
print(f"\nAverage over {len(scores)} attempts: {mean:.1f} (+/- {np.std(scores):.1f})")
print("SOLVED (average >= 200)" if mean >= 200 else "Not solved yet (need an average of 200+)")
if args.record:
    print(f"Videos saved in videos/{name}/")

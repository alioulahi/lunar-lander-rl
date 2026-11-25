"""Score every trained model in both modes and write the results table.

For each model: 20 attempts on the plain game (no reward wrapper), the same
starting positions as watch.py, once picking the network's top action
("test") and once picking actions by chance with its odds ("explore", the
way it acts during training). Results go to results/results.csv and are
printed as a Markdown table.

    python evaluate.py                                   # every model in models/
    python evaluate.py models/ppo_hover_fix_s0.zip       # just one
"""
import argparse
import csv
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="*", help="model .zip files (default: all of models/)")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=1000, help="seed of the first test attempt (same as watch.py)")
args = parser.parse_args()

ORDER = ["default", "time_penalty", "hover_bonus", "hover_fix"]


def sort_key(path):
    reward, seed = re.match(r"ppo_(.+)_s(\d+)$", path.stem).groups()
    return (ORDER.index(reward) if reward in ORDER else len(ORDER), reward, int(seed))


def run(path, explore):
    # Loading resets PyTorch's random seed to the training seed, so explore
    # mode rolls the same dice as `watch.py --explore` and is reproducible.
    model = PPO.load(path, device="cpu")
    env = gym.make("LunarLander-v3")
    scores, frames, endings = [], [], []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        score, n, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=not explore)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            n += 1
            done = terminated or truncated
        scores.append(score)
        frames.append(n)
        endings.append("landed" if terminated and reward == 100 else "crashed" if terminated else "timeout")
    env.close()
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "landed": endings.count("landed"),
        "crashed": endings.count("crashed"),
        "timeout": endings.count("timeout"),
        "frames": float(np.mean(frames)),
    }


paths = sorted([Path(p) for p in args.models] or Path("models").glob("ppo_*_s*.zip"), key=sort_key)
rows = []
for path in paths:
    reward, seed = re.match(r"ppo_(.+)_s(\d+)$", path.stem).groups()
    for mode in ("test", "explore"):
        print(f"{path.stem:24s} {mode:8s}", end=" ", flush=True)
        r = run(path, explore=(mode == "explore"))
        print(f"{r['mean']:6.1f} +/- {r['std']:5.1f}   landed {r['landed']:2d}  crashed {r['crashed']:2d}  timeout {r['timeout']:2d}")
        rows.append({"model": path.stem, "reward": reward, "seed": int(seed), "mode": mode, **r})

out = Path("results/results.csv")
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})

n = args.episodes
print(f"\nSaved {out}\n")
print("| Reward | Seed | Test average | Test landed | Explore: landed / crashed / time ran out |")
print("|---|---|---|---|---|")
for test, explore in zip(rows[0::2], rows[1::2]):
    print(f"| {test['reward']} | {test['seed']} | {test['mean']:.1f} ± {test['std']:.1f} | {test['landed']} / {n} "
          f"| {explore['landed']} / {explore['crashed']} / {explore['timeout']} |")

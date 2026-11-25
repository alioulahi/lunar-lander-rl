"""Make the README figures: the results chart and three clips.

    python evaluate.py      # first: writes results/results.csv
    python make_media.py    # then: assets/results.png and assets/*.gif

Clips (2x speed):
  landing.gif  the built-in reward, picking its top action: it lands and stops
  cheat.gif    the hover bonus, acting as in training: lands, then fidgets on
               the pad until time runs out, earning +1 per frame
  fixed.gif    the fix, acting as in training: lands and stops

Pass --mp4 DIR to also write the clips as MP4 files.
"""
import argparse
import csv
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from rewards import steady

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mp4", help="also write the clips as MP4s to this folder")
args = parser.parse_args()

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)
ORANGE, GREY, DARK = "#ff5a1f", "#9a9a96", "#2b2b29"


# ---------------------------------------------------------------------------
# Results chart
# ---------------------------------------------------------------------------
def chart():
    rows = list(csv.DictReader(open("results/results.csv")))
    test = [r for r in rows if r["mode"] == "test"]
    explore = [r for r in rows if r["mode"] == "explore"]
    labels = [f"{r['reward']}  (seed {r['seed']})" for r in test]
    colors = [ORANGE if r["reward"] == "hover_bonus" else GREY for r in test]
    y = np.arange(len(test))[::-1]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    a.barh(y, [float(r["mean"]) for r in test], xerr=[float(r["std"]) for r in test],
           color=colors, height=0.62, error_kw={"ecolor": DARK, "elinewidth": 1, "capsize": 3})
    a.axvline(200, color=DARK, lw=1, ls="--")
    a.text(196, y[-1] - 0.72, "solved (200)", fontsize=9, color=DARK, ha="right", va="center")
    a.set_ylim(y[-1] - 1.05, y[0] + 0.6)
    a.set_xlim(0, 320)
    a.set_title("Plain test: every model looks solved", loc="left", fontsize=12, fontweight="bold")
    a.set_xlabel("average score over 20 attempts")

    timeouts = [int(r["timeout"]) for r in explore]
    b.barh(y, timeouts, color=colors, height=0.62)
    for yi, t in zip(y, timeouts):
        b.text(t + 0.3, yi, f"{t} / 20", va="center", fontsize=10, color=DARK)
    b.set_xlim(0, 21)
    b.set_xticks([0, 5, 10, 15, 20])
    b.set_title("Acting as in training: the cheat shows up", loc="left", fontsize=12, fontweight="bold")
    b.set_xlabel("attempts that ran out of time (of 20)")

    a.set_yticks(y, labels)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(ASSETS / "results.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("Saved", ASSETS / "results.png")


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------
FONT = ImageFont.load_default(size=17)


def label(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((14, 12 + 22 * i), text, fill=(255, 255, 255), font=FONT)
    return np.asarray(img)


def episode(model_path, explore, want, title, count_bonus=False):
    """Record the first attempt (from seed 1000) that ends the way we want."""
    model = PPO.load(model_path, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    for seed in range(1000, 1020):
        obs, _ = env.reset(seed=seed)
        frames, bonus, n, done = [], 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=not explore)
            obs, reward, terminated, truncated, _ = env.step(action)
            n += 1
            done = terminated or truncated
            bonus += steady(obs) if count_bonus else 0.0
            lines = [title, f"frame {n:4d} / 1000"]
            if count_bonus:
                lines.append(f"hover bonus earned: +{bonus:.0f}")
            frames.append(label(env.render(), lines))
        ending = "landed" if terminated and reward == 100 else "crashed" if terminated else "timeout"
        if ending == want:
            env.close()
            return frames
    raise RuntimeError(f"no attempt of {model_path} ended with {want}")


def save(name, frames):
    fast = frames[::4] + [frames[-1]] * 12  # every 4th frame at 25 fps = 2x speed, short pause at the end
    small = [np.asarray(Image.fromarray(f).resize((480, 320), Image.LANCZOS)) for f in fast]
    imageio.mimsave(ASSETS / f"{name}.gif", small, duration=40, loop=0)
    print("Saved", ASSETS / f"{name}.gif", f"({len(frames)} game frames)")
    if args.mp4:
        Path(args.mp4).mkdir(parents=True, exist_ok=True)
        clip = frames[::2] + [frames[-1]] * 25  # every 2nd frame at 50 fps = 2x speed
        imageio.mimsave(Path(args.mp4) / f"{name}.mp4", clip, fps=50, codec="libx264",
                        quality=8, macro_block_size=8, pixelformat="yuv420p")
        imageio.imwrite(Path(args.mp4) / f"{name}.jpg", frames[min(len(frames) - 1, 400)], quality=88)


chart()
save("landing", episode("models/ppo_default_s0.zip", False, "landed", "built-in reward"))
save("cheat", episode("models/ppo_hover_bonus_s0.zip", True, "timeout", "hover bonus, acting as in training", count_bonus=True))
save("fixed", episode("models/ppo_hover_fix_s0.zip", True, "landed", "the fix, acting as in training"))

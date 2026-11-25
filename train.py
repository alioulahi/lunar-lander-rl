"""Train a PPO agent to land the rocket.

PPO with the standard small network (MlpPolicy: two layers of 64 units).
Watch `ep_rew_mean` in the printout: the average score over the last 100
attempts. It starts well below zero, wobbles, and should climb past 200.

`ep_rew_mean` always shows the game's own score, even when training with a
custom reward rule, so you can see whether the rocket really lands. Each
finished attempt is also logged to logs/<run>/ for plotting learning curves.

    python train.py                                # built-in reward, 1M steps
    python train.py --reward hover_fix             # with a rule from rewards.py
    python train.py --reward hover_fix --seed 1    # same rule, different seed
"""
import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rewards import REWARDS

parser = argparse.ArgumentParser()
parser.add_argument("--reward", default="default", choices=sorted(REWARDS))
parser.add_argument("--steps", type=int, default=1_000_000, help="one step = one frame")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", help="where to save the model (default: models/ppo_<reward>_s<seed>.zip)")
args = parser.parse_args()

out = Path(args.out or f"models/ppo_{args.reward}_s{args.seed}.zip")
out.parent.mkdir(parents=True, exist_ok=True)

# 16 copies of the game run side by side so training collects frames faster.
# make_vec_env puts its score logger (Monitor) *under* our wrapper, so the
# logged score is the game's own reward, not the shaped one.
env = make_vec_env("LunarLander-v3", n_envs=16, seed=args.seed,
                   wrapper_class=REWARDS[args.reward],
                   monitor_dir=f"logs/{out.stem}")

# Settings from the Hugging Face Deep RL Course, Unit 1 (known to land in ~1M steps).
model = PPO(
    "MlpPolicy", env,
    n_steps=1024, batch_size=64, n_epochs=4,
    gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
    seed=args.seed, device="cpu", verbose=1,
)

print(f"Training '{args.reward}' reward for {args.steps:,} steps (seed {args.seed})...")
start = time.time()
model.learn(total_timesteps=args.steps, progress_bar=True)
minutes = (time.time() - start) / 60

model.save(out)
print(f"\nDone in {minutes:.1f} min. Saved to {out}")
print(f"Next: python watch.py {out}")

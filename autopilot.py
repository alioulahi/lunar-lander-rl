"""A hand-written autopilot, before any learning.

A pilot (in RL: a "policy") is just a function: 8 numbers in, 1 action out.
This one is written by hand with if/else: level the rocket first, then brake
the fall. It scores about -38 on average, against about -200 for a random
pilot. train.py has a neural network learn a pilot instead.

The 5 attempts always start the same way (fixed seeds), so changing the
pilot and running again is a fair comparison.

    python autopilot.py              # watch it fly
    python autopilot.py --no-render  # just the scores, fast
"""
import argparse

import gymnasium as gym


def pilot(obs):
    x, y, vx, vy, tilt, spin, left_leg, right_leg = obs

    if tilt > 0.1:      # leaning left: the right engine turns it back
        return 3
    if tilt < -0.1:     # leaning right: the left engine turns it back
        return 1
    if vy < -0.5:       # falling too fast: main engine
        return 2
    return 0


# ---------------------------------------------------------------------------
# The game loop (same as random_agent.py).
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()

env = gym.make("LunarLander-v3", render_mode=None if args.no_render else "human")
scores = []

for attempt in range(5):
    obs, info = env.reset(seed=attempt)
    score, frame, done = 0.0, 0, False
    while not done:
        last_obs = obs
        action = pilot(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        frame += 1
        done = terminated or truncated

    if terminated and reward == 100:
        ending = "LANDED"
    elif terminated:
        ending = "crashed / flew off"
    else:
        ending = "ran out of time (1000 frames)"
    x, y, vx, vy, tilt, spin, _, _ = last_obs
    print(f"Attempt {attempt + 1}: score {score:7.1f}  {frame:4d} frames  {ending}")
    print(f"           last numbers: x={x:+.2f}  y={y:+.2f}  vx={vx:+.2f}  vy={vy:+.2f}  tilt={tilt:+.2f}")
    scores.append(score)

env.close()
print(f"\nAverage score: {sum(scores) / len(scores):.1f}   (random pilot is about -200, solved is 200+)")

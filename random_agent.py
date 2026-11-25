"""A random pilot: watch it fail.

A rocket that picks a random action every frame. It tumbles, crashes and
restarts (about -200 per attempt). Every 10th frame it prints the 8 numbers
the AI sees.

    python random_agent.py              # opens the game window
    python random_agent.py --no-render  # just the numbers, no window
"""
import argparse

import gymnasium as gym

LABELS = ["x (left/right)", "y (height)", "vx (side speed)", "vy (up/down speed)",
          "tilt (rad)", "spin", "left leg", "right leg"]
ACTIONS = ["nothing", "left engine", "main engine", "right engine"]

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()

env = gym.make("LunarLander-v3", render_mode=None if args.no_render else "human")

for episode in range(args.episodes):
    obs, info = env.reset()
    score, frame, done = 0.0, 0, False
    while not done:
        action = env.action_space.sample()  # a random action: 0, 1, 2 or 3
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        if frame % 10 == 0:
            numbers = "  ".join(f"{name}={value:+.2f}" for name, value in zip(LABELS, obs))
            print(f"frame {frame:4d} | {ACTIONS[action]:12s} | reward {reward:+7.2f} | {numbers}")
        frame += 1
    ending = "landed" if terminated and reward == 100 else "crashed / flew off" if terminated else "ran out of time"
    print(f"\nEpisode {episode + 1}: {frame} frames, score {score:.1f} ({ending})\n")

env.close()

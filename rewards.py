"""Reward rules: extra points added on top of the game's own reward.

A wrapper sits between the game and PPO. On every frame it takes the built-in
reward, adds or subtracts its own terms, and passes the total on. The game's
code is never touched.

To add a rule:
  1. Subclass ShapedReward and fill in `extra()`.
  2. Register it in REWARDS below with a short name.
  3. Train with it:  python train.py --reward <name>

What `obs` holds (run random_agent.py to watch the values live):
  obs[0] x        obs[1] y        obs[2] vx       obs[3] vy
  obs[4] tilt     obs[5] spin     obs[6] left leg obs[7] right leg
Actions: 0 nothing, 1 left engine, 2 main engine, 3 right engine.
"""
import gymnasium as gym


class ShapedReward(gym.Wrapper):
    """Adds `extra(...)` to the game's own reward on every step.

    Inside `extra()`, `self.prev_obs` holds the previous frame's 8 numbers,
    so a rule can pay for a change (this frame vs the last) instead of a state.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs  # a new attempt: there is no earlier frame yet
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self.extra(obs, action, reward, terminated)
        self.prev_obs = obs  # remember this frame for the next one
        info["builtin_reward"] = reward  # keep the real score around
        return obs, reward + bonus, terminated, truncated, info

    def extra(self, obs, action, reward, terminated):
        """Return the number to add to this frame's reward (negative = penalty)."""
        return 0.0


def steady(obs):
    """1.0 if the rocket is upright and slow (tilt, vx and vy all within 0.1 of zero), else 0.0."""
    x, y, vx, vy, tilt, spin, left_leg, right_leg = obs
    if abs(tilt) < 0.1 and abs(vx) < 0.1 and abs(vy) < 0.1:
        return 1.0
    return 0.0


class TimePenalty(ShapedReward):
    """-0.05 on every frame, so dragging an attempt out costs points."""

    def extra(self, obs, action, reward, terminated):
        return -0.05


class HoverBonus(ShapedReward):
    """+1 on every frame the rocket is upright and slow.

    Pays for a *state*, so staying in that state forever is the best plan:
    the agent learns to touch down and keep firing its side engines, so the
    game never counts the landing and the bonus keeps paying.
    """

    def extra(self, obs, action, reward, terminated):
        return steady(obs)


class HoverFix(ShapedReward):
    """The fix: pay for *becoming* steady, not for being steady, plus a small fee.

    steady(now) - steady(before) sums to steady(last) - steady(first) over an
    attempt, so the bonus is worth at most +1 however long the agent stalls
    (potential-based reward shaping, Ng, Harada & Russell 1999).
    """

    def extra(self, obs, action, reward, terminated):
        now = steady(obs)
        before = steady(self.prev_obs)
        return now - before - 0.05


# name -> wrapper class. None means "the game's built-in reward only".
REWARDS = {
    "default": None,
    "time_penalty": TimePenalty,
    "hover_bonus": HoverBonus,
    "hover_fix": HoverFix,
}

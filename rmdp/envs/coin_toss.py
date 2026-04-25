r"""Coin-toss robust MDP (Section 4.1).

State :math:`X_{t}\in\{0,\dots,n\}` — number of heads in a block of
:math:`n` Bernoulli trials. Actions :math:`a\in\{-1,0,+1\}` (bet-lower,
abstain, bet-higher). Reference transition
:math:`X_{t+1}\sim\mathrm{Binomial}(n,p_{0})`, independent of
:math:`(x,a)`. Running reward

.. math::

    f(t,x,a,x') = a\,\mathbf 1_{x'>x} - a\,\mathbf 1_{x'<x}
                  - |a|\,\mathbf 1_{x'=x},\qquad g(x)=0.

Ground cost :math:`c(x,y)=|x-y|` (Wasserstein-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


ACTIONS = np.array([-1, 0, 1], dtype=np.int64)


@dataclass
class CoinTossSpec:
    n:  int   = 10
    T:  int   = 10
    p0: float = 0.5


def build(spec: CoinTossSpec):
    """Return ``(P0, reward, terminal, cost, mu0)``."""
    n      = spec.n
    states = np.arange(n + 1)
    n_x, n_a = n + 1, ACTIONS.size

    pmf = binom.pmf(states, n, spec.p0)
    P0  = np.broadcast_to(pmf, (n_x, n_a, n_x)).copy()

    xp_gt = states[None, None, :] >  states[:, None, None]
    xp_lt = states[None, None, :] <  states[:, None, None]
    xp_eq = states[None, None, :] == states[:, None, None]
    a     = ACTIONS[None, :, None]
    reward = (a * xp_gt - a * xp_lt - np.abs(a) * xp_eq).astype(float)

    terminal = np.zeros(n_x)
    cost = np.abs(states[:, None] - states[None, :]).astype(float)
    mu0  = np.full(n_x, 1.0 / n_x)
    return P0, reward, terminal, cost, mu0


def simulate_profit(
    actions: np.ndarray,            # (T, |X|) greedy action index at (t, x)
    p_true:  float,
    spec:    CoinTossSpec,
    n_games: int = 100_000,
    seed:    int = 0,
) -> np.ndarray:
    r"""Monte-Carlo cumulative profit under the true bias :math:`p_{\mathrm{true}}`."""
    rng    = np.random.default_rng(seed)
    x      = np.full(n_games, spec.n // 2, dtype=np.int64)
    profit = np.zeros(n_games)
    for t in range(spec.T):
        a      = ACTIONS[actions[t, x]]
        x_next = rng.binomial(spec.n, p_true, size=n_games)
        r      = np.where(x_next > x, a, np.where(x_next < x, -a, -np.abs(a)))
        profit += r
        x       = x_next
    return profit

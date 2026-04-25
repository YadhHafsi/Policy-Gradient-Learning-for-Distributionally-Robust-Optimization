r"""Supply-chain (inventory) robust MDP — Section 4.2.

State :math:`X_{t}\in\{0,\dots,n\}` is on-hand inventory, action
:math:`a\in\{0,\dots,n\}` is the order quantity. With post-order
inventory :math:`\bar x=\min(n,x+a)` and reference demand
:math:`D_{t}\sim\mathrm{Uniform}\{0,\dots,n\}`, the next inventory is
:math:`X_{t+1}=\max(0,\bar x-D_{t})`, and the one-step cost is

.. math::

    \ell(x,a,d) = h\,(\bar x - d)_{+}
                + p\,(d - \bar x)_{+}
                + k\,\mathbf 1_{a>0}.

Since the actor--critic interface uses :math:`(x,a,x')`, we work with
the conditional expected reward given :math:`X_{t+1}=x'`:

.. math::

    f(t,x,a,x') = -\,\mathbb E_{D}\!\bigl[\ell(x,a,D)\,\big|\,X_{t+1}=x'\bigr].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SupplyChainSpec:
    n: int   = 10
    T: int   = 1
    h: float = 1.0       # holding cost
    p: float = 3.0       # shortage cost
    k: float = 2.0       # fixed ordering cost


def build(spec: SupplyChainSpec):
    """Return ``(P0, reward, terminal, cost, mu0)``."""
    n = spec.n
    states = np.arange(n + 1)
    n_x = n_a = n + 1
    pmf = np.full(n + 1, 1.0 / (n + 1))                          # D ~ Uniform

    x  = states[:, None, None]                                   # (X, 1, 1)
    a  = states[None, :, None]                                   # (1, A, 1)
    d  = states[None, None, :]                                   # (1, 1, D)
    bar_x  = np.minimum(n, x + a)
    x_next = np.maximum(0, bar_x - d)                            # (X, A, D)
    loss   = (spec.h * np.maximum(bar_x - d, 0)
              + spec.p * np.maximum(d - bar_x, 0)
              + spec.k * (a > 0))                                # (X, A, D)

    # Aggregate over demand atoms into (X, A, X') tables.
    P0     = np.zeros((n_x, n_a, n_x))
    reward = np.zeros((n_x, n_a, n_x))
    for di in range(n + 1):
        np.add.at(P0,     (np.arange(n_x)[:, None], np.arange(n_a)[None, :],
                           x_next[:, :, di]),
                  pmf[di])
        np.add.at(reward, (np.arange(n_x)[:, None], np.arange(n_a)[None, :],
                           x_next[:, :, di]),
                  -pmf[di] * loss[:, :, di])
    with np.errstate(invalid="ignore", divide="ignore"):
        reward = np.where(P0 > 0, reward / np.maximum(P0, 1e-12), 0.0)

    terminal = np.zeros(n_x)
    cost = np.abs(states[:, None] - states[None, :]).astype(float)
    mu0  = np.full(n_x, 1.0 / n_x)
    return P0, reward, terminal, cost, mu0

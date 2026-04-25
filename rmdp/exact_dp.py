r"""Wasserstein-robust backward induction (tabular benchmark).

Solves the dual robust Bellman recursion

.. math::

    V_{T}(x) \;=\; g(x), \qquad
    V_{t}(x) \;=\; \max_{a\in A}\widehat G_{t}(x,a),

where :math:`\widehat G_{t}` is the Blanchet--Murthy dual of
:mod:`rmdp.duality`.  In the finite tabular setting the robust optimum
over randomized policies is attained by a deterministic selector at every
:math:`(t,x)`, hence replacing :math:`\mathbb E_{a\sim\pi_{t}}` by
:math:`\max_a` yields the exact robust value.  Used throughout the paper
as the tabular benchmark against which Algorithm 1 is evaluated.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .duality import robust_continuation


def solve_robust_dp(
    reward:   np.ndarray,        # (T, |X|, |A|, |X|) or (|X|, |A|, |X|)
    terminal: np.ndarray,        # (|X|,)
    P0:       np.ndarray,
    cost:     np.ndarray,        # (|X|, |X|)
    eps:      float,
    q:        int = 1,
    T:        int | None = None,
    lam_max:  float = 50.0,
    n_lam:    int   = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return :math:`(V, G, \pi^{\star})` with shapes
    ``(T+1,|X|)``, ``(T,|X|,|A|)``, ``(T,|X|)``."""
    reward = np.asarray(reward);  P0 = np.asarray(P0)
    if reward.ndim == 3:
        assert T is not None, "pass T= for time-homogeneous inputs"
        reward = np.broadcast_to(reward, (T, *reward.shape))
        P0     = np.broadcast_to(P0,     (T, *P0.shape))
    T, n_x, n_a = reward.shape[0], reward.shape[1], reward.shape[2]

    V = np.zeros((T + 1, n_x));  V[T] = terminal
    G = np.zeros((T, n_x, n_a))
    pi_star = np.zeros((T, n_x), dtype=np.int64)
    for t in reversed(range(T)):
        G[t], _ = robust_continuation(
            reward[t], V[t + 1], P0[t], cost,
            eps=eps, q=q, lam_max=lam_max, n_lam=n_lam,
        )
        pi_star[t] = G[t].argmax(axis=1)
        V[t]       = G[t].max(axis=1)
    return V, G, pi_star

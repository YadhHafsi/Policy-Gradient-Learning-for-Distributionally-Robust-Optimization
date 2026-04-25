r"""KL-robust backward induction (ambiguity-set comparison).

Donsker--Varadhan dual of the KL ball

.. math::

    \inf_{\mathbb P\,:\,\mathrm{KL}(\mathbb P\Vert\mathbb P^{0})\le\eta}
         \mathbb E^{\mathbb P}[Z]
    \;=\;\sup_{\beta>0}\Bigl\{
        -\beta\,\log\mathbb E^{\mathbb P^{0}}\!\bigl[e^{-Z/\beta}\bigr]
        -\beta\,\eta\Bigr\}.

Used in the supply-chain experiment (Section 4.2) to contrast the
Wasserstein-robust value function against the KL-robust one.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _kl_inner(Z: np.ndarray, P0: np.ndarray, eta: float,
              beta_grid: np.ndarray) -> np.ndarray:
    r"""Row-wise KL infimum along the last axis via Donsker--Varadhan.

    Evaluates, for each row of ``Z``,
    :math:`\sup_{\beta}\{-\beta\log\mathbb E^{\mathbb P^{0}}[e^{-Z/\beta}]-\beta\eta\}`.
    """
    zb = -Z[None] / beta_grid[:, None, None]
    m  = zb.max(axis=-1, keepdims=True)
    log_mgf = m[..., 0] + np.log(
        np.maximum((P0[None] * np.exp(zb - m)).sum(axis=-1), 1e-300))
    return (-beta_grid[:, None] * (log_mgf + eta)).max(axis=0)


def solve_kl_robust_dp(
    reward:   np.ndarray,      # (|X|, |A|, |X|) -- time-homogeneous
    terminal: np.ndarray,      # (|X|,)
    P0:       np.ndarray,      # (|X|, |A|, |X|)
    eta:      float,
    T:        int,
    beta_grid: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """KL-robust backward induction. Returns ``(V, pi)``."""
    if beta_grid is None:
        beta_grid = np.geomspace(1e-2, 20.0, 80)

    n_x, n_a, _ = reward.shape
    V  = np.zeros((T + 1, n_x));  V[T] = terminal
    pi = np.zeros((T, n_x), dtype=np.int64)
    for t in reversed(range(T)):
        Z = (reward + V[t + 1][None, None, :]).reshape(n_x * n_a, n_x)
        G = _kl_inner(Z, P0.reshape(n_x * n_a, n_x), eta, beta_grid
                      ).reshape(n_x, n_a)
        pi[t] = G.argmax(axis=1)
        V[t]  = G.max(axis=1)
    return V, pi

r"""Discrete-adversary benchmark for the robust LQ problem
(Section 4.3 of the paper, eqs. (4.2)--(4.3)).

At every :math:`(t,x)` and every candidate control :math:`u`, the robust
Bellman operator

.. math::
    Q_{t}(x,u) = \ell(x,u)
        + \sup_{\gamma\in\mathcal P(\{\hat w^{(i)}\})\,:\,\mathcal W_{q}(\gamma,\nu)\le\varepsilon}
          \mathbb E_{w\sim\gamma}\!\bigl[V_{t+1}(Ax+Bu+\Xi w)\bigr]

is evaluated through the finite-support Wasserstein dual

.. math::
    \sup_{\lambda\ge 0}
        \Bigl\{\tfrac{1}{N}\sum_{j=1}^{N}\min_{i=1,\dots,N}
            \bigl[V_{t+1}(Ax+Bu+\Xi\hat w^{(i)})+\lambda\,c_{ij}\bigr]
            -\varepsilon^{q}\lambda\Bigr\},\quad
    c_{ij}=\|\hat w^{(i)}-\hat w^{(j)}\|^{q}.

The successor :math:`Ax+Bu+\Xi\hat w^{(i)}` is linearly interpolated onto
the display grid before being read from :math:`V_{t+1}`.  We use the
reward convention (``V`` is the negated cost-to-go), matching Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LQGridSpec:
    """Grid resolution of the discrete-adversary solver."""
    L:      float = 3.0
    n_x:    int   = 61
    n_u:    int   = 41
    u_max:  float = 5.0
    n_lam:  int   = 15


def discrete_adversary_value(instance, eps: float, q: float = 2.0,
                             spec: LQGridSpec = LQGridSpec()) -> dict:
    r"""Exact backward induction for the discrete-adversary LQ problem.

    Returns a ``dict`` with keys ``x_grid``, ``u_grid``,
    ``V`` :math:`(T+1, n_x)`, ``pi`` :math:`(T, n_x)`,
    ``V0``/``u0`` (first-row slices), and ``J`` (uniform-grid mean of
    :math:`V_{0}`).
    """
    assert instance.A.shape == (1, 1), "scalar LQ only"
    a  = float(instance.A[0, 0]);   b  = float(instance.B[0, 0])
    sd = float(instance.Xi[0, 0])
    Q  = float(instance.Q[0, 0]);   R  = float(instance.R[0, 0])
    P_T = float(instance.P_T[0, 0]); T = instance.T
    w = instance.samples.reshape(-1);  N = w.size
    p0 = np.full(N, 1.0 / N)
    C  = np.abs(sd * (w[:, None] - w[None, :])) ** q
    eps_q = eps ** q

    x_grid = np.linspace(-spec.L, spec.L, spec.n_x)
    u_grid = np.linspace(-spec.u_max, spec.u_max, spec.n_u)

    stage = -(Q * x_grid[:, None] ** 2 + R * u_grid[None, :] ** 2)   # (n_x, n_u)
    V  = np.empty((T + 1, spec.n_x))
    V[T] = -P_T * x_grid ** 2
    pi = np.zeros((T, spec.n_x))

    for t in reversed(range(T)):
        y = (a * x_grid[:, None, None]
             + b * u_grid[None, :, None]
             + sd * w[None, None, :])                             # (n_x, n_u, N)
        V_at = np.interp(y.reshape(-1), x_grid, V[t + 1]).reshape(
            spec.n_x, spec.n_u, N)                                # V_{t+1}(y_i)

        if eps_q <= 0:
            best = (p0[None, None, :] * V_at).sum(axis=-1)
        else:
            nominal = (p0[None, None, :] * V_at).sum(axis=-1)
            zero_dual = V_at.min(axis=-1)
            lam_ub = float(np.max(np.maximum(nominal - zero_dual, 0.0) / eps_q))
            lam_ub = max(1e-3, 1.05 * lam_ub + 1e-6)
            if spec.n_lam <= 1:
                lams = np.array([0.0])
            else:
                lams = np.concatenate([
                    [0.0],
                    np.logspace(-4, np.log10(max(lam_ub, 1e-3)), spec.n_lam - 1),
                ])

            best = np.full((spec.n_x, spec.n_u), -np.inf)
            for lam in lams:
                inner = (V_at[:, :, None, :] + lam * C[None, None, :, :]).min(axis=-1)
                dual  = (p0[None, None, :] * inner).sum(-1) - eps_q * lam
                np.maximum(best, dual, out=best)

        Q_t   = stage + best
        idx   = Q_t.argmax(axis=1)
        V[t]  = Q_t[np.arange(spec.n_x), idx]
        pi[t] = u_grid[idx]

    return {"x_grid": x_grid, "u_grid": u_grid, "V": V, "pi": pi,
            "V0": V[0], "u0": pi[0], "J": float(V[0].mean())}

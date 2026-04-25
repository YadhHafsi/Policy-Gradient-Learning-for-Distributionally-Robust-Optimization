r"""Kim--Yang robust Riccati recursion (Proposition 4.1 of the paper).

For the Wasserstein-penalty LQ problem of Section 4.3,

.. math::

    \inf_{\pi}\sup_\gamma\;\mathbb E^{\pi,\gamma}\!\Bigl[
        X_T^{\top}P_T X_T
        +\sum_{s=t}^{T-1}\bigl(X_s^{\top}QX_s+u_s^{\top}Ru_s
                                -\lambda\,\mathcal W_q(\gamma_s,\nu)^{q}\bigr)
        \,\Big|\,X_t=x\Bigr],

let :math:`\Phi:=BR^{-1}B^{\top}-\lambda^{-1}\Xi\Xi^{\top}`.  Under
:math:`\lambda>\bar\lambda_t:=\lambda_{\max}(\Xi^{\top}\Pi_{t+1}\Xi)`
(equivalently :math:`\Phi\succeq 0`), the robust value is quadratic in
:math:`x` and satisfies :math:`V_t(x)=x^{\top}\Pi_t x+2 r_t^{\top}x+z_t`
with :math:`u_t^{\star}(x)=K_t x+L_t` given by the recursion below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustLQ:
    A: np.ndarray           # (d, d)
    B: np.ndarray           # (d, m)
    Xi: np.ndarray          # (d, k)
    Q: np.ndarray           # (d, d),   Q >= 0
    R: np.ndarray           # (m, m),   R >  0
    P_T: np.ndarray         # (d, d),   P_T >= 0
    T: int
    samples: np.ndarray     # empirical reference noise  (N, k)

    @property
    def mean_w(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    @property
    def cov_w(self) -> np.ndarray:
        return self.samples.T @ self.samples / self.samples.shape[0]


def robust_riccati(lq: RobustLQ, lam: float) -> dict:
    r"""Backward recursion of Proposition 4.1.

    Returns ``dict(Pi, r, z, K, L)`` with, for :math:`t=T,\dots,0`,
    :math:`\Pi_t=Q+A^{\top}(I+\Pi_{t+1}\Phi)^{-1}\Pi_{t+1}A`,
    :math:`r_t=A^{\top}(I+\Pi_{t+1}\Phi)^{-1}(\Pi_{t+1}\Xi\bar w+r_{t+1})`,
    and :math:`K_t,L_t` as in the paper.
    """
    d, T = lq.A.shape[0], lq.T
    k = lq.Xi.shape[1]
    I = np.eye(d);  Ik = np.eye(k)
    Phi     = lq.B @ np.linalg.solve(lq.R, lq.B.T) - (lq.Xi @ lq.Xi.T) / lam
    Rinv_Bt = np.linalg.solve(lq.R, lq.B.T)
    w_bar, Sigma = lq.mean_w, lq.cov_w

    Pi = [None] * (T + 1);  r = [None] * (T + 1);  z = [None] * (T + 1)
    K  = [None] * T;        L = [None] * T
    Pi[T], r[T], z[T] = lq.P_T.copy(), np.zeros(d), 0.0

    for t in reversed(range(T)):
        Pn, rn = Pi[t + 1], r[t + 1]
        M       = np.linalg.solve(I + Pn @ Phi, Pn)           # (I + Pi Phi)^{-1} Pi
        Pi[t]   = lq.Q + lq.A.T @ M @ lq.A
        r[t]    = lq.A.T @ np.linalg.solve(
            I + Pn @ Phi, Pn @ lq.Xi @ w_bar + rn)
        K[t]    = -Rinv_Bt @ np.linalg.solve(I + Pn @ Phi, Pn @ lq.A)
        L[t]    = -Rinv_Bt @ np.linalg.solve(
            I + Pn @ Phi, Pn @ lq.Xi @ w_bar + rn)

        ZPX = lq.Xi.T @ Pn @ lq.Xi                            # Xi^T Pi_{t+1} Xi
        z[t] = (
            z[t + 1]
            + float(np.trace(np.linalg.solve(Ik - ZPX / lam, ZPX @ Sigma)))
            + float(w_bar @ lq.Xi.T @ (
                  np.linalg.inv(I + Pn @ Phi)
                - np.linalg.inv(I - Pn @ lq.Xi @ lq.Xi.T / lam)
              ) @ Pn @ lq.Xi @ w_bar)
            + float((2 * w_bar @ lq.Xi.T - rn @ Phi)
                    @ np.linalg.solve(I + Pn @ Phi, rn))
        )
    return {"Pi": Pi, "r": r, "z": z, "K": K, "L": L}


def value_function(lq: RobustLQ, lam: float, x: np.ndarray) -> np.ndarray:
    r""":math:`V_{0}(x)=x^{\top}\Pi_{0}x+2r_{0}^{\top}x+z_{0}`."""
    sol = robust_riccati(lq, lam)
    Pi0, r0, z0 = sol["Pi"][0], sol["r"][0], sol["z"][0]
    x = np.atleast_2d(x)
    return np.einsum("ni,ij,nj->n", x, Pi0, x) + 2.0 * x @ r0 + z0


def policy(lq: RobustLQ, lam: float, t: int, x: np.ndarray) -> np.ndarray:
    r""":math:`u_{t}^{\star}(x)=K_{t}x+L_{t}`."""
    sol = robust_riccati(lq, lam)
    return x @ sol["K"][t].T + sol["L"][t]

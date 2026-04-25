r"""Robust Linear-Quadratic control environment (Section 4.3 of the paper).

Dynamics :math:`X_{t+1}=AX_{t}+Bu_{t}+\Xi w_{t}` with an empirical
reference noise measure :math:`\nu=\tfrac{1}{N}\sum_{i}\delta_{\hat w^{(i)}}`.
Running cost :math:`x^{\top}Qx+u^{\top}Ru`, terminal cost
:math:`x^{\top}P_{T}x`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lq_riccati import RobustLQ


@dataclass
class LQInstance:
    A:       np.ndarray
    B:       np.ndarray
    Xi:      np.ndarray
    Q:       np.ndarray
    R:       np.ndarray
    P_T:     np.ndarray
    T:       int
    samples: np.ndarray                 # (N, k) empirical noise atoms

    def to_riccati(self) -> RobustLQ:
        return RobustLQ(self.A, self.B, self.Xi, self.Q, self.R,
                        self.P_T, self.T, self.samples)


def toy_1d_instance(seed: int = 0) -> LQInstance:
    r"""Scalar LQ benchmark of Section 4.3 (:math:`d=m=k=1`).

    Dynamics :math:`X_{t+1}=0.9X_{t}+u_{t}+w_{t}`, stage cost
    :math:`x^{2}+u^{2}`, terminal cost :math:`2x^{2}`, horizon
    :math:`T=10`, empirical reference noise with :math:`N=30` atoms.
    With :math:`\Xi=1,\,P_{T}=2` the Kim--Yang Riccati recursion is
    PSD-valid provided :math:`\lambda\ge 3`.
    """
    rng = np.random.default_rng(seed)
    A   = np.array([[0.9]]);  B   = np.array([[1.0]])
    Xi  = np.array([[1.0]]);  Q   = np.array([[1.0]])
    R   = np.array([[1.0]]);  P_T = np.array([[2.0]])
    samples = rng.standard_normal(size=(30, 1)) * 0.3
    return LQInstance(A, B, Xi, Q, R, P_T, T=10, samples=samples)

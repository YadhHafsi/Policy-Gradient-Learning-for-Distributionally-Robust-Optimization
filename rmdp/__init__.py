r"""Robust Actor-Critic Learning under Distributional Uncertainty.

The package is organised one paper-object per file:

* :mod:`rmdp.duality`         Blanchet--Murthy operator :math:`\mathcal F^\lambda`
                              and the scalar Wasserstein dual (Prop. 3.4).
* :mod:`rmdp.actor_critic`    Algorithm 1 in the tabular regime
                              (closed-form critics; Section 3.4, Remark 3.11).
* :mod:`rmdp.lq_actor_critic` Algorithm 1 on the scalar LQ problem
                              (Section 4.3, Gaussian-MLP policy).
* :mod:`rmdp.exact_dp`        Wasserstein-robust backward induction
                              (tabular benchmark, Thm. 2.7 + Rmk. 3.5).
* :mod:`rmdp.kl_dp`           KL-robust backward induction
                              (Donsker--Varadhan, supply-chain comparison).
* :mod:`rmdp.lq_riccati`      Kim--Yang robust Riccati (Prop. 4.1).
* :mod:`rmdp.lq_grid`         Discrete-adversary LQ value iteration
                              (eq. (4.2)).
* :mod:`rmdp.plotting`        Publication-quality matplotlib defaults.
* :mod:`rmdp.envs`            Coin toss (Section 4.1), supply chain
                              (Section 4.2), self-exciting bandits
                              (Appendix C.1), linear--quadratic (Section 4.3).

Tabular experiments are pure NumPy; the LQ actor--critic uses ``torch.float32``.
"""

from __future__ import annotations

from . import actor_critic, duality, exact_dp, kl_dp, lq_grid, lq_riccati, plotting

__all__ = [
    "actor_critic", "duality", "exact_dp", "kl_dp",
    "lq_grid", "lq_riccati", "plotting",
]

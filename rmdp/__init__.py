r"""Policy gradient learning for distributionally robust MDPs under Wasserstein
ambiguity (arXiv:2606.27610).

The package is organised one paper-object per file:

* :mod:`rmdp.exact_ascent`    Algorithm 1, exact robust policy ascent (Section 3.5).
* :mod:`rmdp.actor_critic`    Algorithm 2 in the tabular regime
                              (tabulated critics, Remark 3.26).
* :mod:`rmdp.lq_actor_critic` Algorithm 2 on the scalar LQ problem
                              (Section 4.1.3, Gaussian-MLP policy).
* :mod:`rmdp.duality`         Operator :math:`\mathcal F^\lambda` and the scalar
                              Wasserstein dual (Proposition 3.4).
* :mod:`rmdp.exact_dp`        Wasserstein-robust backward induction
                              (tabular benchmark, Theorem 2.7 and Remark 3.5).
* :mod:`rmdp.kl_dp`           KL-robust backward induction (Section 4.1.2).
* :mod:`rmdp.lq_riccati`      Robust Riccati recursion (Proposition 4.1).
* :mod:`rmdp.lq_grid`         Discrete-adversary LQ value iteration (Section 4.1.3).
* :mod:`rmdp.plotting`        Matplotlib defaults.
* :mod:`rmdp.envs`            Coin toss (Section 4.1.1), supply chain
                              (Section 4.1.2), linear--quadratic (Section 4.1.3),
                              self-exciting bandits (Appendix D.1).

Tabular experiments are pure NumPy; the LQ actor--critic uses ``torch.float32``.
"""

from __future__ import annotations

from . import (actor_critic, duality, exact_ascent, exact_dp, kl_dp, lq_grid,
               lq_riccati, plotting)

__all__ = [
    "actor_critic", "duality", "exact_ascent", "exact_dp", "kl_dp",
    "lq_grid", "lq_riccati", "plotting",
]

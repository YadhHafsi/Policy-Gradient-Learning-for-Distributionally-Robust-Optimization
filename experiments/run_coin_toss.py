"""Coin-toss experiment (Section 4.1).

Runs Algorithm 1 on the finite tabular robust MDP described in Section 4.1
and compares the learned greedy policy with the exact Wasserstein-robust DP
benchmark. Results are saved as ``.npz`` files for use by the notebook.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmdp import actor_critic, exact_dp
from rmdp.envs import coin_toss


def run(eps: float, outer: int, seed: int, outdir: str) -> None:
    spec = coin_toss.CoinTossSpec(n=10, T=10, p0=0.5)
    P0, rew, term, cost, mu0 = coin_toss.build(spec)

    V_ex, _, pi_ex = exact_dp.solve_robust_dp(
        rew, term, P0, cost, eps=eps, q=1, T=spec.T)
    cfg = actor_critic.ACConfig(eps=eps, q=1, lr=0.15, n_outer=outer, seed=seed)
    theta, hist = actor_critic.run_actor_critic(
        rew, term, P0, cost, mu0, cfg, T=spec.T)
    pi = actor_critic.greedy_actions(theta)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"coin_toss_eps{eps:.1f}.npz")
    np.savez(path, V_exact=V_ex, pi_exact=pi_ex, pi_learned=pi, theta=theta,
             history_J=np.array([h["J"] for h in hist]),
             history_iter=np.array([h["iter"] for h in hist]),
             eps=eps, p0=spec.p0, n=spec.n, T=spec.T)
    print(f"[coin-toss  ε={eps}]  J={hist[-1]['J']:+.3f}  →  {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    p.add_argument("--outer", type=int, default=800)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--outdir", default="figures")
    a = p.parse_args()
    for e in a.epsilon:
        run(e, a.outer, a.seed, a.outdir)


if __name__ == "__main__":
    main()

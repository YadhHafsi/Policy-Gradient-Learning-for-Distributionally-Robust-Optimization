"""Algorithm 1 (exact robust policy ascent, Section 3.5) on a small random model.

Algorithm 1 enumerates the vertices of every ball and every near-active branch,
so it is run on a small seeded finite MDP. The script reports whether J(theta_k)
is nondecreasing (Theorem 3.25), the final zeta, and the exact robust DP value.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmdp.exact_ascent import ball_vertices, exact_ascent


def random_model(n, m, seed):
    """Seeded finite MDP with rewards in [-1, 1] and cost c(x, y) = |x - y|."""
    rng = np.random.default_rng(seed)
    states = np.arange(n)
    return (rng.dirichlet(np.ones(n), size=(n, m)), rng.uniform(-1, 1, (n, m, n)),
            rng.uniform(-1, 1, n), np.abs(states[:, None] - states[None, :]).astype(float),
            np.full(n, 1.0 / n))


def robust_dp_value(P0, reward, terminal, cost, mu0, eps, T):
    """Optimal robust value (Theorem 2.7), with exact minima over the ball vertices."""
    n, m = reward.shape[:2]
    verts = [[ball_vertices(P0[x, a], cost, eps) for a in range(m)] for x in range(n)]   # q = 1
    V = terminal
    for _ in range(T):
        V = np.array([max((verts[x][a] @ (reward[x, a] + V)).min() for a in range(m)) for x in range(n)])
    return float(mu0 @ V)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--states",  type=int,   default=2)
    p.add_argument("--actions", type=int,   default=2)
    p.add_argument("--T",       type=int,   default=3)
    p.add_argument("--epsilon", type=float, default=0.4)
    p.add_argument("--iters",   type=int,   default=5000)
    p.add_argument("--seed",    type=int,   default=0)
    p.add_argument("--outdir", default="figures")
    a = p.parse_args()

    P0, rew, term, cost, mu0 = random_model(a.states, a.actions, a.seed)
    theta, hist = exact_ascent(rew, term, P0, cost, mu0, eps=a.epsilon, q=1, T=a.T,
                               n_iter=a.iters, seed=a.seed)
    J, zeta = np.array([h["J"] for h in hist]), np.array([h["zeta"] for h in hist])
    J_dp = robust_dp_value(P0, rew, term, cost, mu0, a.epsilon, a.T)

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"exact_ascent_eps{a.epsilon:.1f}.npz")
    np.savez(path, theta=theta, history_J=J, history_zeta=zeta, J_dp=J_dp)
    print(f"[exact ascent  ε={a.epsilon}]  J: {J[0]:+.4f} -> {J[-1]:+.4f}  "
          f"(robust DP {J_dp:+.4f})  monotone={bool(np.all(np.diff(J) >= -1e-12))}  "
          f"final ζ={zeta[-1]:.1e}  →  {path}")


if __name__ == "__main__":
    main()

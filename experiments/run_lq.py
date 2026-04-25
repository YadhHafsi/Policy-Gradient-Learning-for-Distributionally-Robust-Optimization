"""Robust Linear-Quadratic control (Section 4.3 of the paper).

Two modes:

* ``--mode riccati`` — Kim--Yang closed form (Proposition 4.1) on the 2-D
  toy instance for a single fixed :math:`\\lambda`.  Reference for Figure 3.
* ``--mode grid``    — Discrete-adversary grid solver on the scalar toy
  instance, paired with the learned Algorithm 1 policy.  Reference for
  Figure 4.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmdp import lq_grid, lq_riccati
from rmdp.envs import lq as lq_env
from rmdp.lq_actor_critic import LQACConfig, LQRobustActorCritic


def run_riccati(lam: float, outdir: str) -> None:
    # Scalar 1-D instance of Section 4.3: Xi = 1, P_T = 2 -> require lam >= 3.
    inst = lq_env.toy_1d_instance(seed=0)
    sol = lq_riccati.robust_riccati(inst.to_riccati(), lam=lam)
    xs = np.linspace(-3.0, 3.0, 121).reshape(-1, 1)
    u  = (xs @ sol["K"][0].T + sol["L"][0]).reshape(-1)
    V  = lq_riccati.value_function(inst.to_riccati(), lam, xs)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"lq_riccati_lam{lam:.1f}.npz")
    np.savez(path, x=xs.reshape(-1), u=u, V=V, lam=lam, T=inst.T)
    print(f"[LQ/riccati  lambda={lam}]  saved -> {path}")


def run_grid(eps: float, outer: int, outdir: str) -> None:
    torch.set_num_threads(1)
    inst = lq_env.toy_1d_instance(seed=0)
    ref = lq_grid.discrete_adversary_value(inst, eps=eps, q=2.0)

    cfg = LQACConfig(eps=eps, q=2.0, L=3.0, n_outer=outer, seed=0)
    agent = LQRobustActorCritic(inst, cfg)
    logs = agent.train(verbose=False)
    ev = agent.evaluate(ref["x_grid"])

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"lq_grid_eps{eps:.2f}.npz")
    np.savez(path, x=ref["x_grid"], V_bench=ref["V0"], u_bench=ref["u0"],
             V_learned=ev["V"], u_learned=ev["u"],
             history_J=np.array(logs["J_net"]),
             history_sigma=np.array(logs["sigma"]),
             eps=eps, T=inst.T)
    print(f"[LQ/grid     ε={eps}]  J_learned={logs['J_net'][-1]:+.3f}  "
          f"J_bench={ref['J']:+.3f}  →  {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["riccati", "grid"], default="grid")
    p.add_argument("--epsilon", nargs="+", type=float, default=[0.05, 0.2, 0.4, 0.7])
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()
    if args.mode == "riccati":
        run_riccati(args.lam, args.outdir)
    else:
        for e in args.epsilon:
            run_grid(e, args.outer, args.outdir)


if __name__ == "__main__":
    main()

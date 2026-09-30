# Policy Gradient Learning for Distributionally Robust Optimization

Code for [*Policy Gradient Learning for Distributionally Robust Markov Decision
Processes under Wasserstein Ambiguity*](https://arxiv.org/abs/2606.27610)
(arXiv:2606.27610). It implements both algorithms of the paper and the
examples of Section 4.1 and Appendix D.1.

| Algorithm | Code | Run |
|---|---|---|
| Algorithm 1 — exact robust policy ascent (Section 3.5) | `rmdp/exact_ascent.py` | `experiments/run_exact_ascent.py` |
| Algorithm 2 — robust actor–critic, tabular (Section 3.5.1) | `rmdp/actor_critic.py` | `experiments/run_coin_toss.py`, `run_supply_chain.py`, `run_bandits.py` |
| Algorithm 2 — robust actor–critic, LQ (Section 4.1.3) | `rmdp/lq_actor_critic.py` | `experiments/run_lq.py` |

## Quick start

```bash
pip install -r requirements.txt

python3 experiments/run_exact_ascent.py                          # Algorithm 1
python3 experiments/run_coin_toss.py    --epsilon 0.5 1.0 2.0    # Section 4.1.1
python3 experiments/run_supply_chain.py --epsilon 1.0            # Section 4.1.2
python3 experiments/run_lq.py --mode grid --epsilon 0.05 0.2 0.4 0.7   # Section 4.1.3
python3 experiments/run_bandits.py      --epsilon 0.3            # Appendix D.1
```

[`experiments.ipynb`](experiments.ipynb) runs these experiments end-to-end
(≈ 20 min on one CPU thread, mostly the LQ actor–critic) and writes the figures to
`figures/`: Table 2 and Figures 2–4 and 6–9. The paper does not list the bandit
parameters behind Figures 5 and 7; `rmdp/envs/bandits.py` states the ones used here.

## Layout

```
rmdp/
  exact_ascent.py      Algorithm 1: ball vertices, near-active LP, safeguarded step
  actor_critic.py      Algorithm 2, tabular regime (tabulated critics, Remark 3.26)
  lq_actor_critic.py   Algorithm 2, scalar LQ, Gaussian-MLP policy
  duality.py           Scalar Wasserstein dual                      Proposition 3.4
  exact_dp.py          Robust backward induction (benchmark)        Theorem 2.7, Remark 3.5
  kl_dp.py             KL-robust backward induction                 Section 4.1.2
  lq_riccati.py        Robust Riccati recursion                     Proposition 4.1
  lq_grid.py           Discrete-adversary LQ value iteration        Section 4.1.3
  plotting.py          Matplotlib defaults
  envs/                coin_toss (4.1.1), supply_chain (4.1.2), lq (4.1.3), bandits (D.1)
experiments/           one script per experiment
```

## Implementation notes

* **Algorithm 1** enumerates the transport plans of every Wasserstein ball
  (|X|^|supp P⁰| per ball) and every near-active branch in S_ζ(θ), a product over
  all (t, x, a), so the script runs it on a small random model. The step uses
  global bounds C₁ = 4TM and C₂ = TM(4T+1)/2, with M = T‖f‖∞ + ‖g‖∞, valid for
  the tabular softmax (Theorem 3.22(iv)).
* **Algorithm 2** computes the supremum over λ ∈ [0, Λ] on a log-spaced grid
  anchored at λ = 0 (`n_lam`, `lam_max`). The tabular actor step is
  preconditioned by Adam by default (`ACConfig.use_adam=False` gives the plain
  step of line 11); `ACConfig.B` projects the iterates onto [−B, B]^{d_θ}
  (Remark 3.26).
* **Algorithm 2, LQ**: line 11 uses the Monte-Carlo target of line 9 at t = 0,
  runs `n_inner_sweeps` backward sweeps per actor step, and clips the ascent
  direction to norm `grad_clip` instead of projecting.
* The Riccati recursion of Proposition 4.1 requires λ > λ̄ₜ for every t ≥ 1; for
  the scalar instance (Ξ = 1, P_T = 2) this means λ > 2.

## Citation

```bibtex
@article{hafsi2026policy,
  title   = {Policy Gradient Learning for Distributionally Robust Markov Decision
             Processes under Wasserstein Ambiguity},
  author  = {Hafsi, Yadh and Mekkaoui, Samy and Pham, Huy{\^e}n and Yan, Kaixin},
  journal = {arXiv preprint arXiv:2606.27610},
  year    = {2026}
}
```

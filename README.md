# Policy Gradient Learning for Distributionally Robust Optimization

Reference implementation of **Algorithm 1** of https://arxiv.org/pdf/2606.27610

Every figure and table of Sections 4.1–4.3 and Appendix C.1 is
reproduced by [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb).
All plots are saved to `figures/` as Type-42 TrueType `.pdf` and `.png`.

## Layout

```
rmdp/
  duality.py           Blanchet–Murthy dual F^λ                        Prop. 3.4 (Strong duality)
  actor_critic.py      Algorithm 1 — tabular regime, closed-form critics
  lq_actor_critic.py   Algorithm 1 — scalar LQ, Gaussian-MLP policy    §4.3
  exact_dp.py          Wasserstein-robust backward induction           Thm. 2.7 (DPP) + Rmk. 3.5 (dual DPP)
  kl_dp.py             KL-robust backward induction (Sec. 4.2 cmp.)
  lq_riccati.py        Kim–Yang robust Riccati                         Prop. 4.1
  lq_grid.py           Discrete-adversary LQ value iteration           eq. (4.2)
  plotting.py          Publication-quality matplotlib defaults + Wong palette
  envs/
    coin_toss.py         §4.1
    supply_chain.py      §4.2
    bandits.py           App. C.1
    lq.py                §4.3

experiments/
  run_coin_toss.py  run_supply_chain.py  run_bandits.py  run_lq.py

notebooks/experiments.ipynb     reproduces every figure end-to-end
figures/*.pdf, *.png            all paper figures
```

### Algorithm 1 — code map

| Paper line                                  | Function                                                |
|---------------------------------------------|---------------------------------------------------------|
| Line 2 (init `V_T = g`, `U_T = 0`)          | `actor_critic.backward_pass`                            |
| Step 1 — robust continuation Ĝ_t            | `duality.robust_continuation`                           |
| Step 2 — V-critic regression                | `actor_critic.backward_pass` (tabular collapse: `V[t]=E_a[Ĝ_t]`) |
| Step 3 — worst-case transport y⋆            | `duality.transport_indices`                             |
| Step 4 — gradient of continuation ∇Ĝ_t     | `actor_critic.backward_pass` (`einsum` over y⋆)         |
| Step 5 — U-critic regression                | `actor_critic.backward_pass`                            |
| Actor update `θ ← θ + η μ₀ᵀU₀`              | `actor_critic.run_actor_critic`                         |

## Quick start

```bash
pip install -r requirements.txt

# Option A — reproduce every figure end-to-end via the notebook.
jupyter lab notebooks/experiments.ipynb                         # interactive
jupyter nbconvert --to notebook --execute notebooks/experiments.ipynb \
    --inplace --ExecutePreprocessor.timeout=3600

# Option B — run each experiment individually.
python3 experiments/run_coin_toss.py    --epsilon 0.5 1.0 2.0
python3 experiments/run_supply_chain.py --epsilon 1.0
python3 experiments/run_bandits.py      --epsilon 0.3
python3 experiments/run_lq.py --mode riccati --lam 5.0
python3 experiments/run_lq.py --mode grid    --epsilon 0.05 0.2 0.4 0.7
```

End-to-end wall-time on a 2024 MacBook (1 thread): **≈ 90 min**, dominated
by the LQ actor–critic (≈ 35 min for four ε values). The tabular sections
finish in under 10 min and the LQ Riccati / discrete-adversary panels in
under 10 min more — pass `--ExecutePreprocessor.timeout=600` and skip the
LQ-training cell if you only need the tabular figures.

## Algorithm 1 — the dual at a glance

For a finite-horizon robust MDP under a Wasserstein ambiguity set
`B^{ε,q}(P^0) = { P : W_q(P, P^0) ≤ ε }`, the paper proceeds via the
Blanchet–Murthy scalar dual

```
F^λ(V)(x)                         = sup_y { V(y) − λ c(x, y) },
inf_{P ∈ B^{ε,q}(P^0)} E^P[V(X)]  = sup_{λ ≥ 0} { E^{P^0}[ −F^λ(−V)(X) ] − ε^q λ }.
```

This yields a closed-form robust Bellman recursion and an explicit
policy-gradient DPP **without** differentiating the worst-case kernel P*.
In the tabular regime the critics `V_{ψ,t}, U_{ξ,t}` collapse to direct
bookkeeping (remark after Algorithm 1); in the LQ regime they are small
MLPs.

## Experiments reproduced

| Experiment       | §         | State                    | Action             | T  | Paper ε             |
|------------------|-----------|--------------------------|--------------------|----|----------------------|
| Coin toss        | §4.1      | `{0,…,10}` heads         | `{−1, 0, +1}`      | 10 | `0.5, 1, 2`          |
| Supply chain     | §4.2      | `{0,…,10}` inventory     | `{0,…,10}` order   | 5  | `1`                  |
| Bandits          | App. C.1  | `(m, b)` signed/arm      | `(k, j)` stake/arm | 5  | `0.3`                |
| Robust LQ        | §4.3      | `x ∈ ℝ`                  | `u ∈ ℝ`            | 10 | `0.05, 0.2, 0.4, 0.7`|

All tabular experiments: `c(x, y) = |x − y|`, q = 1.
LQ: empirical reference noise `ν = 1/N ∑ δ_{ŵ^{(i)}}`, q = 2.

Tabular learned policies match the exact Wasserstein-robust DP benchmark
exactly for every ε reported in the paper.

## Figures produced

| Paper figure / table                       | Section in notebook   |
|--------------------------------------------|-----------------------|
| Table 1 (greedy coin-toss policy)          | §1                    |
| Fig. 1 (coin misspecification, `p₀=0.5,0.6`)| §1                   |
| `coin_training_eps{0.5,1.0,2.0}.pdf`       | §1                    |
| Fig. 2 left (supply-chain learned policy)  | §2                    |
| Fig. 2 right (value across ambiguity sets) | §2                    |
| `supply_training_eps1.0.pdf`               | §2                    |
| Fig. 5 (self-exciting bandits policy)      | §3                    |
| `bandits_training_eps0.3.pdf`              | §3                    |
| Fig. 3 (Kim–Yang Riccati)                  | §4                    |
| Fig. 4 top (discrete-adversary benchmark)  | §4                    |
| Fig. 4 bottom (Algorithm 1 on LQ)          | §4                    |
| `lq_training.pdf`                          | §4                    |

## Implementation notes

* The dual-variable grid in `rmdp/duality.py` is log-spaced on `[0, Λ]`
  and anchored at `λ = 0`, so the non-robust case is attained exactly.
  Per Lemma G.1 of the paper a safe default for `Λ̄` is
  `2(‖f‖_∞ + M)/ε^q` with `M = T‖f‖_∞ + ‖g‖_∞`; the tabular default
  `lam_max = 50` dominates this for all reported instances.
* Kim–Yang Riccati (`rmdp/lq_riccati.py`) is PSD-valid only when
  `λ > λ_max(Ξᵀ Π_{t+1} Ξ)` for every `t`; for the scalar toy instance
  (`Ξ = 1, P_T = 2`) this requires `λ ≳ 3`.
* The LQ actor–critic exploits the Gaussian score factorisation
  `∇_θ log π(a|x,t) = ((a − μ)/σ²) ∇_θ μ(x, t)` and assembles the
  mean-network Jacobian in closed form through the two ReLU layers
  (`GaussianPolicy.mu_jacobian`), avoiding any `vmap`/`functional_call`
  overhead.
* Outer actor update is `θ ← θ + η · (1/M) ∑ⱼ U_{ξ,0}(X₀^(j))` as in the
  paper; an Adam preconditioner is optionally applied to the same
  ascent direction for numerical stability (`ACConfig.use_adam`).

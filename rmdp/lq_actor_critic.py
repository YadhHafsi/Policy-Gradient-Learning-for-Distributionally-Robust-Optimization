r"""Algorithm 1 -- scalar robust Linear-Quadratic control (Section 4.3).

Dynamics :math:`X_{t+1} = A X_t + B u_t + \Xi w_t` with reference noise
:math:`\nu = \tfrac1N \sum_{i=1}^N \delta_{\hat w^{(i)}}` (empirical). The
Wasserstein adversary redistributes mass among the :math:`N` empirical
atoms, so the robust Bellman step of Algorithm 1 specialises to a finite
dual over :math:`(\hat w^{(i)})_{i=1}^N`:

with :math:`y_i = A x + B a + \Xi \hat w^{(i)}`,
:math:`u_i = f(t, x, a, y_i) + V_{t+1}(y_i)` and
:math:`c_{ij} = \|\Xi(\hat w^{(i)} - \hat w^{(j)})\|^q`,

.. math::

    \widehat G_t(x, a) \;=\;
    \sup_{\lambda \in [0, \Lambda]}\!\Bigl\{
        \tfrac{1}{N}\sum_{j=1}^N \min_{i \in \{1,\dots,N\}}
        \bigl[\,u_i + \lambda\,c_{ij}\,\bigr]
        \,-\, \varepsilon^q\,\lambda\Bigr\},
    \qquad i^\star(j) \in \arg\min_i\{u_i + \lambda^\star c_{ij}\}.

Parametrisation
---------------
:math:`\pi^\theta_t(x,\cdot) = \mathcal N(\mu_\theta(t,x), \sigma_\theta^2)`
with :math:`\mu_\theta` a two-layer ReLU MLP and :math:`\log \sigma_\theta`
a learnable scalar. Per-stage critics :math:`V_{\psi,t}` and
:math:`U_{\xi,t}` are small MLPs with hard-wired boundary values
:math:`V_{\psi,T}(x) = -P_T x^2`, :math:`U_{\xi,T}(x) = 0`.

Everything runs in ``torch.float32``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .envs.lq import LQInstance

torch.set_num_threads(1)
_DTYPE = torch.float32


# -------------------------------------------------------------------- #
# Algorithm 1, Step 1 -- discrete Wasserstein dual over N atoms.       #
# -------------------------------------------------------------------- #
def wasserstein_dual(
    u_batch: torch.Tensor,      # (M, N) -- u_i per (x, a) row
    p0:      torch.Tensor,      # (N,)   -- 1 / N
    C:       torch.Tensor,      # (N, N) -- c_{ij}
    eps_q:   float,
    n_lam:   int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return :math:`(\widehat G, i^\star)` for each row of ``u_batch``."""
    M, N   = u_batch.shape
    if eps_q <= 0:
        value = (p0[None] * u_batch).sum(dim=-1)
        return value, torch.arange(N).expand(M, N)

    # For any lambda, the dual objective is bounded above by
    # E_{P0}[u] - eps^q lambda, while lambda=0 gives min_i u_i. Therefore no
    # maximizer can lie beyond (E_{P0}[u] - min_i u_i) / eps^q. This bound has
    # the right small-radius behaviour: as eps -> 0, the search interval grows.
    nominal = (p0[None] * u_batch).sum(dim=-1)
    zero_dual = u_batch.min(dim=-1).values
    lam_ub = float(((nominal - zero_dual).clamp_min(0.0) / eps_q).max().item())
    lam_ub = max(1e-3, 1.05 * lam_ub + 1e-6)
    if n_lam <= 1:
        lams = torch.zeros(1, dtype=_DTYPE)
    else:
        lams = torch.cat([
            torch.zeros(1, dtype=_DTYPE),
            torch.logspace(-4, math.log10(max(lam_ub, 1e-3)), n_lam - 1, dtype=_DTYPE),
        ])
    # shifted[l, m, j, i] = u_{m, i} + λ_l · c_{ji}
    shifted       = u_batch[None, :, None, :] + lams[:, None, None, None] * C[None, None]
    mins, argmins = shifted.min(dim=-1)                               # (L, M, N)
    dual          = (p0[None, None] * mins).sum(-1) - eps_q * lams[:, None]
    best          = dual.argmax(dim=0)                                # (M,)
    value         = dual.gather(0, best[None, :]).squeeze(0)
    i_star        = argmins.gather(0, best[None, :, None].expand(1, M, N)).squeeze(0)
    return value, i_star


# -------------------------------------------------------------------- #
# Diagonal Gaussian policy  π^θ_t(x, ·) = N(μ_θ(t, x), σ_θ^2).         #
# -------------------------------------------------------------------- #
class GaussianPolicy(nn.Module):
    """Two-layer ReLU mean network with a learnable scalar log-std."""

    def __init__(self, hidden: int = 128, log_sigma_init: float = 0.0):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(hidden, 2, dtype=_DTYPE))
        self.b1 = nn.Parameter(torch.zeros(hidden, dtype=_DTYPE))
        self.W2 = nn.Parameter(torch.empty(1, hidden, dtype=_DTYPE))
        self.b2 = nn.Parameter(torch.zeros(1, dtype=_DTYPE))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.log_sigma = nn.Parameter(torch.tensor(log_sigma_init, dtype=_DTYPE))
        self.hidden    = hidden

    def _forward(self, x, t):
        inp = torch.stack([x, t], dim=-1)                             # (B, 2)
        z   = inp @ self.W1.T + self.b1                               # (B, H)
        h   = torch.relu(z)
        mu  = (h @ self.W2.T + self.b2).squeeze(-1)
        return mu, z, h, inp

    def mu(self, x, t):  return self._forward(x, t)[0]
    def sigma(self):     return torch.exp(self.log_sigma)

    def sample(self, x, t, S):
        return self.mu(x, t).unsqueeze(-1) + self.sigma() * torch.randn(
            x.shape[0], S, dtype=_DTYPE)

    @torch.no_grad()
    def mu_jacobian(self, x, t) -> torch.Tensor:
        r"""Closed-form :math:`\nabla_{(W_1, b_1, W_2, b_2)} \mu_\theta(x, t)`."""
        _, z, h, inp = self._forward(x, t)
        B     = h.shape[0]
        delta = (z > 0).to(_DTYPE) * self.W2.squeeze(0)                # (B, H)
        return torch.cat([
            (delta.unsqueeze(-1) * inp.unsqueeze(1)).reshape(B, -1),   # dW1
            delta,                                                     # db1
            h.reshape(B, -1),                                          # dW2
            torch.ones(B, 1, dtype=_DTYPE),                            # db2
        ], dim=-1)


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, out_dim))


# -------------------------------------------------------------------- #
# Hyper-parameters.                                                    #
# -------------------------------------------------------------------- #
@dataclass
class LQACConfig:
    eps:            float = 0.2       # Wasserstein radius ε
    q:              float = 2.0       # ground-cost exponent
    L:              float = 3.0       # evaluation domain [-L, L]
    n_outer:        int   = 100
    log_every:      int   = 10
    n_inner_sweeps: int   = 5         # backward sweeps per actor step
    n_critic_steps: int   = 15        # SGD steps for V_ψ and U_ξ
    batch_size:     int   = 64
    n_samples:      int   = 8         # action samples per state
    actor_hidden:   int   = 128
    critic_hidden:  int   = 64
    actor_lr:       float = 0.05
    critic_lr:      float = 1e-2
    log_sigma_init: float = 0.0
    grad_clip:      float = 1.0
    n_lam:          int   = 50
    seed:           int   = 0


# -------------------------------------------------------------------- #
# Robust actor--critic for the scalar LQ problem (d = m = k = 1).      #
# -------------------------------------------------------------------- #
class LQRobustActorCritic:
    r"""Algorithm 1 specialised to the scalar robust LQ problem."""

    def __init__(self, inst: LQInstance, cfg: LQACConfig = LQACConfig()):
        assert inst.A.shape == (1, 1), "scalar LQ only"
        self.inst, self.cfg, self.K = inst, cfg, inst.T
        torch.manual_seed(cfg.seed);  np.random.seed(cfg.seed)

        a, b, sd = float(inst.A[0, 0]), float(inst.B[0, 0]), float(inst.Xi[0, 0])
        self._a  = torch.tensor(a,  dtype=_DTYPE)
        self._b  = torch.tensor(b,  dtype=_DTYPE)
        self._sd = torch.tensor(sd, dtype=_DTYPE)
        self._Q, self._R, self._P_T = (float(inst.Q[0, 0]), float(inst.R[0, 0]),
                                       float(inst.P_T[0, 0]))

        # Empirical reference noise and pairwise cost matrix c_{ij}.
        w           = inst.samples.reshape(-1).astype(np.float32)
        self.w_hat  = torch.as_tensor(w, dtype=_DTYPE)
        self.N      = w.size
        self.p0     = torch.full((self.N,), 1.0 / self.N, dtype=_DTYPE)
        self.C      = torch.abs(self._sd *
                                (self.w_hat[None, :] - self.w_hat[:, None])) ** cfg.q
        self.eps_q  = cfg.eps ** cfg.q

        # Policy and per-stage critics (V_ψ, U_ξ).
        self.policy  = GaussianPolicy(cfg.actor_hidden, cfg.log_sigma_init)
        self.d_theta = sum(p.numel() for p in self.policy.parameters())
        self.d_mu    = self.d_theta - 1                               # every param except log_sigma
        self.V_nets  = nn.ModuleList([_mlp(1, 1,            cfg.critic_hidden) for _ in range(self.K)])
        self.U_nets  = nn.ModuleList([_mlp(1, self.d_theta, cfg.critic_hidden) for _ in range(self.K)])
        self.V_opts  = [optim.Adam(n.parameters(), lr=cfg.critic_lr) for n in self.V_nets]
        self.U_opts  = [optim.Adam(n.parameters(), lr=cfg.critic_lr) for n in self.U_nets]

    # Critics with hard-wired boundaries  V_T(x) = -P_T x^2,  U_T(x) = 0.
    def _V(self, t, x):
        if t == self.K:
            return -self._P_T * x.squeeze(-1) ** 2
        return self.V_nets[t](x).squeeze(-1)

    def _U(self, t, x):
        if t == self.K:
            return torch.zeros(x.shape[0], self.d_theta, dtype=_DTYPE)
        return self.U_nets[t](x)

    # Gaussian score ∇_θ log π^θ_t(x, a).
    @torch.no_grad()
    def _score(self, x_state, t_state, x_s, t_s, a_s):
        B = x_state.shape[0];  S = x_s.shape[0] // B
        sigma = float(self.policy.sigma().item())
        J_mu  = self.policy.mu_jacobian(x_state, t_state)             # (B, d_mu)
        J_mu  = J_mu.unsqueeze(1).expand(B, S, self.d_mu).reshape(B * S, self.d_mu)
        mu_bs = self.policy.mu(x_s, t_s)
        res   = (a_s - mu_bs) / (sigma * sigma)
        score_sigma = ((a_s - mu_bs) / sigma) ** 2 - 1.0
        return torch.cat([res.unsqueeze(-1) * J_mu,
                          score_sigma.unsqueeze(-1)], dim=-1)

    # On-policy rollout under the reference dynamics.
    @torch.no_grad()
    def _rollout(self, B: int):
        xs = [(2 * torch.rand(B, dtype=_DTYPE) - 1) * self.cfg.L]
        for t in range(self.K):
            tn = torch.full((B,), float(t) / self.K, dtype=_DTYPE)
            a  = self.policy.sample(xs[-1], tn, 1).squeeze(-1)
            j  = torch.multinomial(self.p0.expand(B, -1), 1).squeeze(-1)
            xs.append(self._a * xs[-1] + self._b * a + self._sd * self.w_hat[j])
        return xs

    def _backward_sweep(self):
        cfg, K, N = self.cfg, self.K, self.N
        B, S = cfg.batch_size, cfg.n_samples;  BS = B * S
        xs   = self._rollout(B);  U0 = None

        for t in reversed(range(K)):
            x  = xs[t]
            tn = torch.full((B,), float(t) / K, dtype=_DTYPE)
            with torch.no_grad():
                a_s = self.policy.sample(x, tn, S)                    # (B, S)

            x_r = x .unsqueeze(1).expand(B, S).reshape(BS)
            t_r = tn.unsqueeze(1).expand(B, S).reshape(BS)
            a_r = a_s.reshape(BS)

            # Successor atoms  y_i = A x + B a + Ξ ŵ^(i)  and  u_i = f + V_{t+1}(y_i).
            y      = (self._a * x_r + self._b * a_r)[:, None] + self._sd * self.w_hat[None, :]
            with torch.no_grad():
                V_next = self._V(t + 1, y.reshape(-1).unsqueeze(-1)).reshape(BS, N)
            rwd    = -(self._Q * x_r ** 2 + self._R * a_r ** 2)       # f(x, a, ·)
            u_vec  = rwd[:, None] + V_next

            # Step 1 -- discrete Wasserstein dual over the N atoms.
            with torch.no_grad():
                G_flat, i_star = wasserstein_dual(
                    u_vec, self.p0, self.C, self.eps_q, n_lam=cfg.n_lam)

            # Steps 3--4 -- continuation gradient ∇_θ Ĝ_t(x, a).
            with torch.no_grad():
                y_star = y.gather(1, i_star)                          # (BS, N)
                U_next = self._U(t + 1, y_star.reshape(-1, 1)).reshape(BS, N, self.d_theta)
                grad_G = (self.p0[None, :, None] * U_next).sum(dim=1)

            # Step 5 target  z_t = Ĝ_t · ∇log π^θ_t(x, a) + ∇_θ Ĝ_t(x, a).
            score = self._score(x, tn, x_r, t_r, a_r).detach()
            z     = G_flat[:, None] * score + grad_G

            # Average S samples per state to get (B, ·) regression targets.
            V_tgt = G_flat.reshape(B, S).mean(dim=1)                  # Step 2
            U_tgt = z     .reshape(B, S, self.d_theta).mean(dim=1)    # Step 5

            # Steps 2 & 5 -- regress V_ψ and U_ξ.
            x_in = x.unsqueeze(-1).detach()
            for _ in range(cfg.n_critic_steps):
                lv = ((self.V_nets[t](x_in).squeeze(-1) - V_tgt) ** 2).mean()
                self.V_opts[t].zero_grad();  lv.backward();  self.V_opts[t].step()
                lu = ((self.U_nets[t](x_in) - U_tgt) ** 2).mean()
                self.U_opts[t].zero_grad();  lu.backward();  self.U_opts[t].step()

            if t == 0:
                U0 = U_tgt.detach()
        return U0

    def _actor_update(self, U0: torch.Tensor) -> float:
        g  = U0.mean(dim=0)
        gn = float(g.norm().item())
        if gn > self.cfg.grad_clip:
            g = g * (self.cfg.grad_clip / gn)
        offset = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.add_(self.cfg.actor_lr * g[offset:offset + n].view_as(p))
            offset += n
        return gn

    @torch.no_grad()
    def estimate_J_net(self, n: int = 1000) -> float:
        r"""Plug-in estimate :math:`\widehat J = \mathbb E_{X_0}[V_{\psi,0}(X_0)]`."""
        x = ((2 * torch.rand(n, dtype=_DTYPE) - 1) * self.cfg.L).unsqueeze(-1)
        return float(self._V(0, x).mean().item())

    @torch.no_grad()
    def estimate_J_mc(self, n: int = 200) -> float:
        r"""Monte-Carlo value under the learned worst-case kernel."""
        cfg = self.cfg
        x = (2 * torch.rand(n, dtype=_DTYPE) - 1) * cfg.L
        cost = torch.zeros(n, dtype=_DTYPE)
        for t in range(self.K):
            tn = torch.full((n,), float(t) / self.K, dtype=_DTYPE)
            a  = self.policy.sample(x, tn, 1).squeeze(-1)
            cost += self._Q * x ** 2 + self._R * a ** 2
            y = (self._a * x + self._b * a)[:, None] + self._sd * self.w_hat[None, :]
            V_next = self._V(t + 1, y.reshape(-1, 1)).reshape(n, self.N)
            _, i_star = wasserstein_dual(
                -(self._Q * x ** 2 + self._R * a ** 2)[:, None] + V_next,
                self.p0, self.C, self.eps_q, n_lam=cfg.n_lam)
            j = torch.multinomial(self.p0.expand(n, -1), 1).squeeze(1)
            x = y[torch.arange(n), i_star[torch.arange(n), j]]
        cost += self._P_T * x ** 2
        return float(-cost.mean().item())

    def train(self, verbose: bool = False) -> dict:
        cfg = self.cfg
        logs = {"J_net": [], "J_mc": [], "grad_norm": [], "sigma": [], "iter": []}
        for it in range(1, cfg.n_outer + 1):
            for _ in range(cfg.n_inner_sweeps):
                U0 = self._backward_sweep()
            gn = self._actor_update(U0)

            if it % cfg.log_every == 0 or it == cfg.n_outer:
                logs["J_net"].append(self.estimate_J_net())
                logs["J_mc"].append(self.estimate_J_mc())
                logs["grad_norm"].append(gn)
                logs["sigma"].append(float(self.policy.sigma().item()))
                logs["iter"].append(it)
                if verbose:
                    print(f"  [{it:>3}/{cfg.n_outer}]  J_net={logs['J_net'][-1]:+.3f}  "
                          f"J_mc={logs['J_mc'][-1]:+.3f}  "
                          f"sigma={logs['sigma'][-1]:.3f}  |g|={gn:.2f}")
        return logs

    @torch.no_grad()
    def evaluate(self, x_grid: np.ndarray) -> dict:
        xt = torch.as_tensor(x_grid, dtype=_DTYPE);  zt = torch.zeros_like(xt)
        mu = self.policy.mu(xt, zt).cpu().numpy()
        V  = self._V(0, xt.unsqueeze(-1)).cpu().numpy()
        return {"x": x_grid, "u": mu, "V": V,
                "sigma": float(self.policy.sigma().item())}

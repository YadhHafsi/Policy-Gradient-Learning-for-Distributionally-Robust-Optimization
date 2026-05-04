r"""Robust Actor--Critic Gradient Algorithm (Algorithm 1) -- tabular regime.

On finite :math:`\mathcal X, \mathcal A` we use the time-indexed tabular
softmax policy

.. math::

    \pi^{\theta}_t(x, a) \;=\;
    \frac{\exp(\theta_{t,x,a})}{\sum_{a'}\exp(\theta_{t,x,a'})}.

By the remark following Algorithm 1, the parametric critics
:math:`V_{\psi,t}, U_{\xi,t}` collapse to direct tables in this regime
and the regressions of Steps 2 and 5 become exact assignments. The score
is block-diagonal in :math:`(t, x)`:

.. math::

    \nabla_{\theta_{t',x',b}} \log \pi^{\theta}_t(x, a)
    \;=\; \mathbf 1_{\{t'=t,\,x'=x\}}\,
          \bigl(\mathbf 1_{\{a=b\}} - \pi^{\theta}_t(x, b)\bigr).

Algorithm 1 -- line by line in :func:`backward_pass`:

============= =========================================================
Paper line    Code
============= =========================================================
Init.         ``V[T] = g``,  ``U[T] = 0``
Step 1        :func:`rmdp.duality.robust_continuation`
Step 2        ``V[t] = E_{a~pi_t}[ G_hat ]``                  (tabular)
Step 3        :func:`rmdp.duality.transport_indices`
Step 4        ``grad_G[t] = E_{X~P0}[ U[t+1, y_star(X)] ]``   (tabular)
Step 5        ``U[t] = E_{a~pi_t}[ G_hat * grad log pi + grad_G ]``
Actor update  ``theta <- theta + eta * mu0^T U[0]``
============= =========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .duality import robust_continuation, transport_indices


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def greedy_actions(theta: np.ndarray) -> np.ndarray:
    r"""Greedy selector :math:`a^{\star}_t(x)=\arg\max_a \theta_{t,x,a}`."""
    return theta.argmax(axis=-1)


@dataclass
class ACConfig:
    """Hyper-parameters of Algorithm 1 (tabular regime)."""
    eps:       float = 1.0      # Wasserstein radius eps
    q:         int   = 1        # ground-cost exponent: c(x,y) = |x-y|^q
    lam_max:   float = 50.0     # dual-multiplier upper bound Lambda
    n_lam:     int   = 101      # dual-grid resolution
    lr:        float = 0.1      # actor step size eta_theta
    n_outer:   int   = 1500     # outer actor iterations
    seed:      int   = 0
    log_every: int   = 50
    use_adam:  bool  = True     # Adam preconditioner on the actor ascent direction


class _Adam:
    """Minimal Adam preconditioner applied to the actor ascent direction."""
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape);  self.v = np.zeros(shape);  self.t = 0

    def step(self, g: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat  = self.m / (1 - self.b1 ** self.t)
        v_hat  = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def backward_pass(
    theta:    np.ndarray,    # (T, |X|, |A|)
    reward:   np.ndarray,    # (T, |X|, |A|, |X|)  -- f(t, x, a, x')
    terminal: np.ndarray,    # (|X|,)              -- g(x)
    P0:       np.ndarray,    # (T, |X|, |A|, |X|)  -- reference kernel
    cost:     np.ndarray,    # (|X|, |X|)          -- c(x, y) = |x-y|^q
    cfg:      ACConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Single backward sweep of Algorithm 1 (Steps 1--5).

    Returns ``(V, U)`` with ``V[t, x] = V_t^{theta}(x)`` and
    ``U[t, x] = grad_theta V_t^{theta}(x)`` flattened along
    :math:`(T, |\mathcal X|, |\mathcal A|)`.
    """
    T, n_x, n_a, _ = reward.shape
    pi      = softmax(theta, axis=-1)
    d_theta = T * n_x * n_a

    # Initialisation -- V_T(x) = g(x), U_T(x) = 0  (line 2 of Algorithm 1).
    V = np.zeros((T + 1, n_x));            V[T] = terminal
    U = np.zeros((T + 1, n_x, d_theta))

    for t in reversed(range(T)):
        # Step 1 -- robust continuation G_hat_t(x, a) and dual maximiser lambda*.
        G_hat, lam_star = robust_continuation(
            reward[t], V[t + 1], P0[t], cost,
            eps=cfg.eps, q=cfg.q, lam_max=cfg.lam_max, n_lam=cfg.n_lam,
        )

        # Step 2 -- V-critic regression. Tabular collapse: V[t] = E_a[ G_hat ].
        V[t] = (pi[t] * G_hat).sum(axis=1)

        # Step 3 -- worst-case transport index
        # y*(X) in argmin_y { f(t, x, a, y) + V_{t+1}(y) + lambda* c(X, y) }.
        inner  = (reward[t] + V[t + 1][None, None, :]).reshape(n_x * n_a, n_x)
        y_star = transport_indices(inner, cost, lam_star.reshape(-1)
                                   ).reshape(n_x, n_a, n_x)

        # Step 4 -- gradient of robust continuation
        # grad_G(x, a) = E_{X ~ P^0}[ U_{t+1}( y*(X) ) ].
        grad_G = np.einsum("xay,xayd->xad", P0[t], U[t + 1][y_star])

        # Step 5 -- U-critic regression. Target
        # z_t(x, a) = G_hat(x, a) * grad log pi_t(x, a) + grad_G(x, a),
        # then U[t, x] = E_{a ~ pi_t}[ z_t(x, a) ]. For the softmax score
        # E_a[ G_hat * grad log pi(x, ·) ] populates only the (t, x)-block
        # with entry (t, x, b) = pi_t(x, b) * ( G_hat(x, b) - V_t(x) ).
        U_t       = np.einsum("xa,xad->xd", pi[t], grad_G)
        reinforce = pi[t] * (G_hat - V[t][:, None])
        block     = t * n_x * n_a
        for x in range(n_x):
            U_t[x, block + x * n_a : block + (x + 1) * n_a] += reinforce[x]
        U[t] = U_t
    return V, U


def run_actor_critic(
    reward:   np.ndarray,
    terminal: np.ndarray,
    P0:       np.ndarray,
    cost:     np.ndarray,
    mu0:      np.ndarray,
    cfg:      ACConfig,
    T:        Optional[int] = None,
) -> Tuple[np.ndarray, List[dict]]:
    r"""Outer loop of Algorithm 1.

    The actor update
    :math:`\theta \leftarrow \theta + \eta_{\theta}\,\mathbb E_{X_0\sim\mu_0}
    [\,\nabla_{\theta} V_0^{\theta}(X_0)\,]`
    is computed exactly against :math:`\mu_0` (the :math:`M\to\infty`
    limit of the Monte-Carlo mean in the paper, exact in the tabular regime).
    """
    reward = np.asarray(reward, dtype=np.float64)
    P0     = np.asarray(P0,     dtype=np.float64)
    if reward.ndim == 3:                                    # time-homogeneous inputs
        assert T is not None, "pass T= for time-homogeneous inputs"
        reward = np.broadcast_to(reward, (T, *reward.shape)).copy()
        P0     = np.broadcast_to(P0,     (T, *P0.shape)).copy()
    T_ = reward.shape[0]
    n_x, n_a = reward.shape[1], reward.shape[2]

    rng   = np.random.default_rng(cfg.seed)
    theta = 0.01 * rng.standard_normal((T_, n_x, n_a))
    adam  = _Adam(theta.shape, cfg.lr) if cfg.use_adam else None

    history: List[dict] = []
    for it in range(cfg.n_outer):
        V, U   = backward_pass(theta, reward, terminal, P0, cost, cfg)
        grad_J = (mu0 @ U[0]).reshape(T_, n_x, n_a)
        theta  = theta + (adam.step(grad_J) if adam is not None
                          else cfg.lr * grad_J)

        if it % cfg.log_every == 0 or it == cfg.n_outer - 1:
            history.append({"iter": it, "J": float(mu0 @ V[0]),
                            "V0": V[0].copy()})
    return theta, history

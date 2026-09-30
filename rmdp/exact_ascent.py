r"""Algorithm 1 -- exact robust policy ascent (Section 3.5).

Finite :math:`\mathcal X, A`, time-homogeneous :math:`\mathbb P^0` and
:math:`f`, and the tabular softmax policy of :mod:`rmdp.actor_critic` on the
box :math:`\Theta=[-B,B]^{d_\theta}`.

Each Wasserstein ball is a polytope whose vertices are enumerated once
(:func:`ball_vertices`). A selection :math:`\sigma` picks one of them per
:math:`(t,x,a)`; :math:`J_\sigma` is the value of :math:`\pi^\theta` under that
fixed kernel. Every iteration evaluates the :math:`\zeta`-active selections
:math:`S_\zeta(\theta)`, solves the direction linear program, and either
refines :math:`\zeta` or takes the safeguarded step (Lemma 3.23, Theorem 3.25).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from .actor_critic import softmax


def ball_vertices(p0, cost, eps_q, tol=1e-12):
    r"""Finite set of laws in the ball that contains all of its vertices.

    A vertex of the transport polytope (3.11) sends every source atom to one
    target, except at most one atom split between two targets when the budget
    binds. The second marginals of these plans lie in the ball and include its
    vertices, so minima over them are exact. ``cost`` is :math:`c(x,y)` and
    ``eps_q`` is :math:`\varepsilon^q`.
    """
    n, src = len(p0), list(np.flatnonzero(p0 > tol))
    cands = []
    for split in [None, *src]:
        rest = [i for i in src if i != split]
        for dest in itertools.product(range(n), repeat=len(rest)):
            p, used = np.zeros(n), sum(p0[i] * cost[i, j] for i, j in zip(rest, dest))
            np.add.at(p, list(dest), p0[rest])
            if split is None:
                if used <= eps_q + tol:
                    cands.append(p)
                continue
            for j, k in itertools.combinations(range(n), 2):
                if abs(cost[split, k] - cost[split, j]) < tol:
                    continue
                w = (eps_q - used - p0[split] * cost[split, j]) / (cost[split, k] - cost[split, j])
                if -tol <= w <= p0[split] + tol:
                    q = p.copy()
                    q[j] += p0[split] - w
                    q[k] += w
                    cands.append(q)
    return np.unique(np.round(cands, 12), axis=0)


def _vertex_values(theta, reward, terminal, verts):
    """Exact robust DP: ``H[t][x][a]`` = vertex values of <P, f + V_{t+1}>, and ``V``."""
    T, n, m = theta.shape
    pi = softmax(theta)
    V, H = np.zeros((T + 1, n)), [None] * T
    V[T] = terminal
    for t in reversed(range(T)):
        H[t] = [[verts[x][a] @ (reward[x, a] + V[t + 1]) for a in range(m)] for x in range(n)]
        V[t] = [pi[t, x] @ [h.min() for h in H[t][x]] for x in range(n)]
    return V, H


def _value_and_grad(theta, P, reward, terminal, mu0):
    r""":math:`J_\sigma(\theta)` and its policy gradient for the fixed kernel ``P`` (T, X, A, X)."""
    T, pi = len(P), softmax(theta)
    V, Q = [None] * T + [terminal], [None] * T
    for t in reversed(range(T)):
        Q[t] = (P[t] * (reward + V[t + 1])).sum(-1)
        V[t] = (pi[t] * Q[t]).sum(-1)
    d, grad = mu0, np.zeros_like(theta)
    for t in range(T):
        grad[t] = d[:, None] * pi[t] * (Q[t] - V[t][:, None])
        d = np.einsum("x,xa,xay->y", d, pi[t], P[t])
    return mu0 @ V[0], grad


def _direction(grads, theta, B, zeta):
    r"""Direction LP of Algorithm 1, line 3: :math:`\max_{r \in T_\zeta(\theta)} \min_\sigma \langle\nabla J_\sigma, r\rangle`."""
    g = np.unique(np.round(np.reshape(grads, (len(grads), -1)), 12), axis=0)
    bounds = [(0 if th <= -B + zeta else -1, 0 if th >= B - zeta else 1) for th in theta.ravel()]
    res = linprog(np.r_[np.zeros(g.shape[1]), -1.0], A_ub=np.c_[-g, np.ones(len(g))],
                  b_ub=np.zeros(len(g)), bounds=bounds + [(None, None)])
    return -res.fun, res.x[:-1].reshape(theta.shape)


def exact_ascent(reward, terminal, P0, cost, mu0, eps, q, T, B=5.0, zeta0=0.1, nu=0.5,
                 n_iter=2000, zeta_min=1e-4, max_branches=100_000, seed=0):
    r"""Run Algorithm 1 and return ``(theta, history)``.

    ``reward`` and ``P0`` have shape (X, A, X); ``cost`` is :math:`c(x,y)`, as in
    :mod:`rmdp.duality`.
    The step uses global bounds valid for the tabular softmax (Theorem 3.22(iv)):
    with :math:`M = T\|f\|_\infty + \|g\|_\infty`, :math:`C_1 = 4TM` and
    :math:`C_2 = TM(4T+1)/2`.
    """
    n, m = reward.shape[:2]
    verts = [[ball_vertices(P0[x, a], cost, eps ** q) for a in range(m)] for x in range(n)]
    M = T * np.abs(reward).max() + np.abs(terminal).max()
    C1, C2 = 4 * T * M, T * M * (4 * T + 1) / 2
    theta = 0.01 * np.random.default_rng(seed).standard_normal((T, n, m))
    zeta, history = zeta0, []
    rows = list(itertools.product(range(T), range(n), range(m)))
    for k in range(n_iter):
        V, H = _vertex_values(theta, reward, terminal, verts)
        history.append({"iter": k, "J": float(mu0 @ V[0]), "zeta": zeta})
        if zeta < zeta_min:
            break
        choices = [np.flatnonzero(H[t][x][a] <= H[t][x][a].min() + zeta) for t, x, a in rows]
        if np.prod([len(c) for c in choices], dtype=float) > max_branches:
            raise RuntimeError("S_zeta(theta) is too large to enumerate; lower zeta0 or use a smaller model")
        grads = []
        for sel in itertools.product(*choices):
            P = np.empty((T, n, m, n))
            for (t, x, a), i in zip(rows, sel):
                P[t, x, a] = verts[x][a][i]
            grads.append(_value_and_grad(theta, P, reward, terminal, mu0)[1])
        delta, r = _direction(grads, theta, B, zeta)
        if delta <= zeta:
            zeta *= nu                                             # refinement
        else:
            theta = theta + min(zeta, zeta / C1, delta / (2 * C2)) * r   # safeguarded ascent step
    return theta, history

r"""Blanchet--Murthy scalar Wasserstein dual (Theorem 1 of the paper).

For the Wasserstein ball :math:`\mathbb B^{\varepsilon,q}(\mathbb P^0)`
with ground cost :math:`c(x,y)=\lVert x-y\rVert^q`,

.. math::

    \mathcal F^\lambda(V)(x) \;=\; \sup_{y\in\mathcal X}\bigl\{V(y)-\lambda\,c(x,y)\bigr\},
    \qquad
    \inf_{\mathbb P\in\mathbb B^{\varepsilon,q}(\mathbb P^0)}
        \mathbb E^{\mathbb P}[V(X)]
    \;=\;\sup_{\lambda\ge 0}\!\Bigl\{
        \mathbb E^{\mathbb P^0}\!\bigl[-\mathcal F^\lambda(-V)(X)\bigr]
        -\varepsilon^q\lambda\Bigr\}.

In the tabular regime :math:`(-\mathcal F^\lambda(-V))(x)=\min_{y}\{V(y)+\lambda c(x,y)\}`.
Algorithm 1 invokes:

* :func:`robust_continuation` -- Step 1, returns
  :math:`\widehat G_t(x,a)` and a maximiser
  :math:`\widehat\lambda_t^\star`.
* :func:`transport_indices` -- Step 3, returns the worst-case
  transport indices
  :math:`y^\star(X)\in\arg\min_y\{f+V_{t+1}(y)+\lambda^\star c(X,y)\}`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def lambda_grid(lam_max: float, n_lam: int) -> np.ndarray:
    r"""Log-spaced grid on :math:`[0,\Lambda]` anchored at zero.

    The leading zero recovers the non-robust case exactly; the log spacing
    concentrates the grid where the dual is sensitive (``λ ≈ 0``).
    """
    if n_lam <= 1:
        return np.array([0.0])
    tail = np.logspace(-4, np.log10(max(lam_max, 1e-3)), n_lam - 1)
    return np.concatenate([[0.0], tail])


def robust_continuation(
    reward:  np.ndarray,    # (|X|, |A|, |X|)  -- f(t, x, a, x')
    V_next:  np.ndarray,    # (|X|,)           -- V_{t+1}
    P0:      np.ndarray,    # (|X|, |A|, |X|)  -- reference kernel
    cost:    np.ndarray,    # (|X|, |X|)       -- c(x, y) = |x - y|^q
    eps:     float,
    q:       int   = 1,
    lam_max: float = 50.0,
    n_lam:   int   = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Algorithm 1, Step 1 -- tabular dual evaluation.

    Returns ``(G_hat, lam_star)`` of shape ``(|X|, |A|)`` with

    .. math::

        \widehat G_t(x,a) \;=\;
        \sup_{\lambda\in[0,\Lambda]}\!\Bigl\{
            \mathbb E^{\mathbb P^0_t(x,a,\cdot)}\!\bigl[
                \min_{y\in\mathcal X}
                \{f(t,x,a,y)+V_{t+1}(y)+\lambda\,c(X,y)\}\bigr]
            \,-\,\varepsilon^q\lambda\Bigr\}.
    """
    n_x, n_a, _ = reward.shape
    obj     = (reward + V_next).reshape(n_x * n_a, n_x)             # f + V_{t+1}
    P0_flat = P0.reshape(n_x * n_a, n_x)
    lams    = lambda_grid(lam_max, n_lam)

    # -F^λ(-(f + V_{t+1}))(X) = min_y { f(t,x,a,y) + V_{t+1}(y) + λ c(X, y) }.
    neg_F = (obj[None, :, None, :]
             + lams[:, None, None, None] * cost[None, None, :, :]).min(axis=-1)
    dual  = (P0_flat[None] * neg_F).sum(axis=-1) - (eps ** q) * lams[:, None]

    best     = dual.argmax(axis=0)
    G_hat    = dual[best, np.arange(dual.shape[1])].reshape(n_x, n_a)
    lam_star = lams[best].reshape(n_x, n_a)
    return G_hat, lam_star


def transport_indices(
    inner: np.ndarray,    # (B, |X|)   -- f + V_{t+1}
    cost:  np.ndarray,    # (|X|, |X|) -- c(X, y)
    lam:   np.ndarray,    # (B,)       -- λ*(x, a)
) -> np.ndarray:
    r"""Algorithm 1, Step 3 -- tabular worst-case transport indices.

    Returns ``y_star`` of shape ``(B, |X|)`` with
    :math:`y^\star(X)\in\arg\min_{y\in\mathcal X}\{f+V_{t+1}(y)+\lambda^\star c(X,y)\}`.
    """
    return (inner[:, None, :] + lam[:, None, None] * cost[None, :, :]).argmin(axis=-1)

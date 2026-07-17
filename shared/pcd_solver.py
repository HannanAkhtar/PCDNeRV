"""
PCDSolver — Primary-Constrained Descent solver.

Extracted verbatim from the reference implementation (lines 209-318).
Do not modify the math in this module.

NOTE (PCD-NeRV): the solver is objective-agnostic — it only sees gradient
lists. In the original benchmark the primary objective was cross-entropy,
hence the `grads_ce` / "ce_*" naming below. In PCD-NeRV the primary objective
is the NeRV reconstruction loss (e.g. L2 or Fusion6) and the secondary is the
group-lasso loss. Parameter and diagnostic names are kept verbatim so this
file stays diff-identical to the reference; the training script maps
"ce" -> "primary" when logging.
"""

import math


class PCDSolver:
    """
    CE-Primary Constrained Descent.

    In normalized space:
        d* = argmin ||d - g_CE||^2
             s.t.   g_GL^T d >= tau*||g_GL||^2
    """

    def __init__(
        self,
        tau=0.05,
        beta=0.999,
        eps=1e-8,
    ):
        self.tau = tau
        self.beta = beta
        self.eps = eps
        self.v = [0.0, 0.0]
        self.t = 0

    def step(self, grads_ce, grads_gl):
        self.t += 1

        norm_ce_sq = sum(g.pow(2).sum().item() for g in grads_ce)
        norm_gl_sq = sum(g.pow(2).sum().item() for g in grads_gl)
        dot_cg = sum((a * b).sum().item() for a, b in zip(grads_ce, grads_gl))
        norm_ce = math.sqrt(max(norm_ce_sq, 0.0))

        self.v[0] = self.beta * self.v[0] + (1 - self.beta) * norm_ce_sq
        self.v[1] = self.beta * self.v[1] + (1 - self.beta) * norm_gl_sq
        bc = 1.0 - self.beta ** self.t
        vh_ce = self.v[0] / bc
        vh_gl = self.v[1] / bc
        s_ce = 1.0 / (math.sqrt(vh_ce) + self.eps)
        s_gl = 1.0 / (math.sqrt(vh_gl) + self.eps)

        t_12 = s_ce * s_gl * dot_cg
        t_22 = s_gl * s_gl * norm_gl_sq
        t_11 = s_ce * s_ce * norm_ce_sq
        tau_before = self.tau
        threshold = tau_before * t_22

        d_normed_norm = 0.0
        if t_12 >= threshold:
            mu = 0.0
            conflict = False
            combined = list(grads_ce)
        else:
            if t_22 < 1e-20:
                mu = 0.0
                conflict = False
                combined = list(grads_ce)
            else:
                mu = (threshold - t_12) / t_22
                conflict = True
                d_normed_sq = t_11 + 2 * mu * t_12 + mu * mu * t_22
                d_normed_norm = math.sqrt(max(d_normed_sq, 1e-20))
                rescale = norm_ce / d_normed_norm
                combined = [rescale * (s_ce * gc + mu * s_gl * gg) for gc, gg in zip(grads_ce, grads_gl)]

        cosine = t_12 / (math.sqrt(t_11 * t_22) + 1e-12) if t_11 > 0 and t_22 > 0 else 0.0

        if conflict and d_normed_norm > 1e-12 and t_11 > 1e-20:
            ce_proj = (t_11 + mu * t_12) / (math.sqrt(t_11) * d_normed_norm)
            ce_efficiency = min(ce_proj, 1.0)
        else:
            ce_efficiency = 1.0

        tau_after = tau_before

        diag = {
            "mu": mu,
            "conflict": conflict,
            "cosine_sim": cosine,
            "ce_efficiency": ce_efficiency,
            "g_ce_norm_raw": math.sqrt(max(norm_ce_sq, 0.0)),
            "g_gl_norm_raw": math.sqrt(max(norm_gl_sq, 0.0)),
            "g_ce_norm_normed": math.sqrt(max(t_11, 0.0)),
            "g_gl_norm_normed": math.sqrt(max(t_22, 0.0)),
            "gl_progress": tau_before if conflict else (t_12 / t_22 if t_22 > 1e-20 else 0.0),
            "tau": tau_before,
            "tau_next": tau_after,
        }
        return combined, diag

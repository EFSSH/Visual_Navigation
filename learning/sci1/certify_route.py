"""
SCI-1 v8: route-level certification of corner pass plans.

Groups g = (corner type, pass mode) each have an unknown per-encounter
close-call rate q_g, estimated from n_g fully verified lab encounters with
x_g close calls (independent binomials). A route a assigns a mode to every
corner; its reference annual risk is

    R_ref(a) = sum_k E_k(a_k) q_{type(k), a_k} = sum_g W_g(a) q_g.

Joint confidence region (likelihood ratio against a prior mixture fixed in
advance; valid for any prior, Markov's inequality on the e-value):

    C = { q : L(q; x) >= delta * Lbar(x) },  Lbar = int L(q; x) pi(dq).

The prior is a product over groups; each group gets an atom w0 at zero plus
an equal mixture of Beta components centred on simulator predictions. Then
U(a) = sup_{q in C} W(a).q bounds R_ref for every route simultaneously, so a
route chosen after looking at the data is still covered.

Per-group Clopper-Pearson bounds with Bonferroni over groups (v7c style) are
also simultaneous; they serve as a baseline and as a small safety net.
"""

from math import lgamma, log

import numpy as np

from .release_control_prototype import clopper_pearson_upper


# --------------------------------------------------------------------------
# Prior and likelihoods
# --------------------------------------------------------------------------

def beta_prior(centres, kappas=(10.0, 100.0, 1000.0), w0=0.2, floor=1e-4):
    """Per-group prior: [(w0, [(a, b), ...]), ...] with Beta(kappa m, kappa (1 - m)).

    centres: (G,) predicted rates, or (P, G) for a mixture of P predictions.
    """
    c = np.atleast_2d(np.asarray(centres, float))
    prior = []
    for g in range(c.shape[1]):
        comps = []
        for m in np.clip(c[:, g], floor, 1 - floor):
            for k in kappas:
                comps.append((k * m, k * (1 - m)))
        prior.append((w0, comps))
    return prior


def _log_beta(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def log_marginal(x, n, w0, comps):
    """log int q^x (1-q)^(n-x) pi(dq) for one group (no binomial coefficient)."""
    terms = [log(1 - w0) + _log_beta(x + a, n - x + b) - _log_beta(a, b) - log(len(comps))
             for a, b in comps] if w0 < 1 else []
    if x == 0 and w0 > 0:
        terms.append(log(w0))
    if not terms:
        return -np.inf
    mx = max(terms)
    return mx + log(sum(np.exp(t - mx) for t in terms))


def _loglik(q, x, n):
    q = np.clip(q, 1e-300, 1.0)
    out = np.where(x > 0, x * np.log(q), 0.0)
    out = out + np.where(n - x > 0, (n - x) * np.log(np.clip(1 - q, 1e-300, 1.0)), 0.0)
    return out


# --------------------------------------------------------------------------
# Joint region: sup of a linear function
# --------------------------------------------------------------------------

def _q_of_lambda(c, x, n):
    """Maximiser of W q + lambda * loglik for one group, c = W / lambda (closed form).

    Stationarity x/q - (n-x)/(1-q) = -c  <=>  c q^2 + (n - c) q - x = 0.
    """
    q = np.empty_like(c)
    big = c > 1e-12
    cb, nb, xb = c[big], n[big], x[big]
    q[big] = (-(nb - cb) + np.sqrt((nb - cb) ** 2 + 4 * cb * xb)) / (2 * cb)
    with np.errstate(divide='ignore', invalid='ignore'):
        q[~big] = np.where(n[~big] > 0, x[~big] / np.maximum(n[~big], 1), 1.0)
    return np.clip(q, 0.0, 1.0)


def joint_upper(W, x, n, prior, delta, iters=80):
    """sup_{q in C} W.q and the maximiser. Groups with n = 0 are unconstrained (q = 1)."""
    W, x, n = (np.asarray(v, float) for v in (W, x, n))
    thr = log(delta) + sum(log_marginal(int(xi), int(ni), *prior[g])
                           for g, (xi, ni) in enumerate(zip(x, n)))
    free = (n == 0)
    act = (W > 0) & ~free
    mle = np.where(n > 0, x / np.maximum(n, 1), 1.0)
    q = mle.copy()
    q[free] = 1.0
    ll_rest = float(_loglik(mle[~act & ~free], x[~act & ~free], n[~act & ~free]).sum())
    if not act.any():
        return float(W @ q), q
    xa, na, Wa = x[act], n[act], W[act]

    def ll_at(lam):
        qa = _q_of_lambda(Wa / lam, xa, na)
        return float(_loglik(qa, xa, na).sum()) + ll_rest, qa

    lo, hi = -20.0, 20.0            # log10 lambda
    for _ in range(iters):
        mid = (lo + hi) / 2
        ll, _ = ll_at(10 ** mid)
        if ll >= thr:
            hi = mid                # feasible: allow a smaller lambda (further out)
        else:
            lo = mid
    _, qa = ll_at(10 ** hi)
    q[act] = qa
    return float(W @ q), q


def cp_bounds(x, n, delta):
    """Per-group Clopper-Pearson upper bounds (n = 0 -> 1)."""
    return np.array([clopper_pearson_upper(int(a), int(b), delta) if b > 0 else 1.0
                     for a, b in zip(x, n)])


# --------------------------------------------------------------------------
# Route search
# --------------------------------------------------------------------------

def allocate_general(risk, seconds, budget):
    """Greedy multiple-choice allocation: minimise total seconds s.t. total risk <= budget.

    risk: (K, M) annual risk of corner k in mode m; seconds: (M,) per pass, modes in
    any order. Per corner only Pareto options (more time must buy less risk) are kept.
    Returns (plan (K,) mode indices, total risk) or (None, risk of the safest plan).
    """
    risk = np.asarray(risk, float)
    K, M = risk.shape
    order = np.argsort(seconds)
    options = []
    for k in range(K):
        opts = []
        for m in order:
            if not opts or risk[k, m] < risk[k, opts[-1]] - 1e-15:
                opts.append(int(m))
        options.append(opts)
    level = np.zeros(K, int)
    cur = np.array([risk[k, options[k][0]] for k in range(K)])
    total = float(cur.sum())
    while total > budget:
        best, bk = -1.0, -1
        for k in range(K):
            if level[k] + 1 < len(options[k]):
                a, b = options[k][level[k]], options[k][level[k] + 1]
                gain = risk[k, a] - risk[k, b]
                cost = max(seconds[b] - seconds[a], 1e-9)
                if gain / cost > best:
                    best, bk = gain / cost, k
        if bk < 0:
            return None, total
        level[bk] += 1
        new = risk[bk, options[bk][level[bk]]]
        total += new - cur[bk]
        cur[bk] = new
    return np.array([options[k][level[k]] for k in range(K)]), total


def route_weights(plan, E, ctype, n_types, n_modes):
    """W[g] with g = type * n_modes + mode."""
    W = np.zeros(n_types * n_modes)
    np.add.at(W, ctype * n_modes + plan, E[np.arange(len(plan)), plan])
    return W


def search_joint(E, ctype, seconds, budget, upper, q_start, n_types, iters=15):
    """Iterative linearisation for a convex route bound U(plan) = upper(W) -> (value, q*).

    Every returned plan is re-checked with the exact bound, so certification holds
    whatever the search does; the search itself is a heuristic.
    """
    K, M = E.shape
    q_w = np.asarray(q_start, float)
    b = budget
    best = None
    for _ in range(iters):
        risk = E * q_w.reshape(n_types, M)[ctype]
        plan, _ = allocate_general(risk, seconds, b)
        if plan is None:
            b *= 1.2
            if b > 10 * budget:
                break
            continue
        U, q_star = upper(route_weights(plan, E, ctype, n_types, M))
        sec = float(seconds[plan].sum())
        if U <= budget:
            if best is None or sec < best[1]:
                best = (plan, sec, U)
            q_w, b = q_star, budget
        else:
            q_w = q_star
            b *= 0.98 * budget / U
    return best


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    i = np.arange(0, k + 1)
    logpmf = (np.array([lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1) for j in i])
              + i * np.log(p) + (n - i) * np.log1p(-p))
    return float(np.exp(logpmf).sum())


def single_route_threshold(N, total_W, budget, delta):
    """Largest close-call count c with CP_upper(c, N) * total_W <= budget (else -1)."""
    c = -1
    for k in range(N + 1):
        if clopper_pearson_upper(k, N, delta) * total_W <= budget:
            c = k
        else:
            break
    return c


def route_chernoff_upper(W, x, n, delta, lams=np.geomspace(1e-3, 1e3, 400)):
    """Upper confidence bound on R = W.q for one pre-specified route, any allocation n.

    S = sum_g (W_g / n_g) x_g is unbiased for R. For lambda > 0,
    P(S <= s) <= exp(lambda s) prod_g (1 - q_g (1 - e^{-lambda w_g}))^{n_g}
              <= exp(lambda s - R * (1 - e^{-lambda w_max}) / w_max),
    with w_g = W_g / n_g, since (1 - e^{-lambda w}) / w decreases in w. Inverting at
    level delta gives U = min_lambda (lambda s + log(1/delta)) w_max / (1 - e^{-lambda w_max}).
    With no events this is log(1/delta) * max_g W_g / n_g. Groups with W_g > 0 and
    n_g = 0 make the route uncertifiable (returns inf).
    """
    W, x, n = (np.asarray(v, float) for v in (W, x, n))
    use = W > 0
    if (n[use] == 0).any():
        return np.inf
    w = W[use] / n[use]
    s = float((w * x[use]).sum())
    wm = float(w.max())
    vals = (lams * s + np.log(1 / delta)) * wm / (-np.expm1(-lams * wm))
    return float(vals.min())

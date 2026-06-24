"""The six distances as low-dimensional trainers on the spiral substrate (Fig. 3 / Table 4).

Each trains the shared MLP generator (:mod:`rdm.toy.spiral`) under one distance at a matched
budget and returns ``(samples_2d, recall, medDist)``. The MMD family uses a multi-scale
Gaussian kernel with ``m`` features / landmarks per scale; the Nystrom arm uses the explicit
``K_ZZ^{-1/2}`` feature map (the eigh form of :func:`rdm.compare.nystrom.nystrom_feature_map`)
against a frozen global reference mean, while the exact estimator sees ``B`` reference samples
per step. SW uses 1000 quantile-matched projections; drifting is the faithful coupled field
with a memory bank of 128. These are the controlled study behind "why MMD, why Nystrom": at
small batch only MMD-Nystrom (frozen reference) stays sharp.
"""
import math
import os

import torch

from .spiral import (D, L_SW, M_FEAT, MLP, SCALES, ZDIM, metrics, real_sigma, real_tr)


def run_kernel(bs, method, seed=42, n_iters=8000):
    """Train under one kernel/transport distance ('exact' | 'nystrom' | 'rff' | 'sw')."""
    torch.manual_seed(seed)
    P = torch.linalg.qr(torch.randn(D, 2))[0]
    realD = real_tr @ P.t()
    Nr = len(realD)
    sig = real_sigma(realD)
    sigs = [s * sig for s in SCALES]
    Ws = []
    for s in sigs:
        W = torch.randn(D, M_FEAT) / s
        bb = torch.rand(M_FEAT) * 2 * math.pi
        phi = lambda x, W=W, bb=bb: math.sqrt(2 / M_FEAT) * torch.cos(x @ W + bb)
        Ws.append((phi, phi(realD).mean(0)))
    lm = realD[torch.randperm(Nr)[:M_FEAT]].clone()
    d_lm = torch.cdist(lm, lm) ** 2
    d_rlm = torch.cdist(realD, lm) ** 2
    nys = []
    for s in sigs:
        K = torch.exp(-d_lm / (2 * s * s))
        ev, evec = torch.linalg.eigh(K + 1e-6 * torch.eye(M_FEAT))
        Wn = evec @ torch.diag(ev.clamp(min=1e-6).rsqrt())           # K_ZZ^{-1/2}
        MUn = (torch.exp(-d_rlm / (2 * s * s)) @ Wn).mean(0)         # frozen reference mean
        nys.append((s, Wn, MUn))
    gen = MLP(D)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters, eta_min=1e-4)

    def L_exact(g, real):
        dgg = torch.cdist(g, g) ** 2
        dgr = torch.cdist(g, real) ** 2
        out = 0
        for s in sigs:
            out = out + torch.exp(-dgg / (2 * s * s)).mean() - 2 * torch.exp(-dgr / (2 * s * s)).mean()
        return out

    def L_nys(g):                                                    # k_gg exact + k_gr via Nystrom mean
        dgg = torch.cdist(g, g) ** 2
        dglm = torch.cdist(g, lm) ** 2
        out = 0
        for s, Wn, MUn in nys:
            out = out + torch.exp(-dgg / (2 * s * s)).mean() - 2 * ((torch.exp(-dglm / (2 * s * s)) @ Wn).mean(0) @ MUn)
        return out

    def L_rff(g):
        out = 0
        for phi, MU in Ws:
            out = out + ((phi(g).mean(0) - MU) ** 2).sum()
        return out

    def L_sw(g, real):
        Dr = torch.randn(D, L_SW)
        Dr = Dr / Dr.norm(dim=0, keepdim=True)
        gs, _ = (g @ Dr).sort(0)
        rs, _ = (real @ Dr).sort(0)
        return ((gs - rs.detach()) ** 2).mean()

    for _ in range(n_iters):
        z = torch.randn(bs, ZDIM)
        g = gen(z)
        idx = torch.randperm(Nr)[:bs]
        real = realD[idx]
        opt.zero_grad()
        if method == "exact":
            L_exact(g, real).backward()
        elif method == "nystrom":
            L_nys(g).backward()
        elif method == "rff":
            L_rff(g).backward()
        elif method == "sw":
            L_sw(g, real).backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
        opt.step()
        sch.step()
    with torch.no_grad():
        p2 = gen(torch.randn(4000, ZDIM)) @ P
    return (p2,) + metrics(p2)


def faithful_mb(g, real, R_list):
    """Faithful coupled mass-conserving drift field with a real memory bank (M != N)."""
    N, Dd = g.shape
    Mr = real.shape[0]
    old = g.detach()
    targets = torch.cat([old, real], 0)
    dist = torch.cdist(old, targets)
    scale = dist.mean()
    si = (scale / math.sqrt(Dd)).clamp(min=1e-3)
    old_s = old / si
    tgt_s = targets / si
    dn = dist / scale.clamp(min=1e-3)
    mask = torch.zeros(N, N + Mr)
    mask[torch.arange(N), torch.arange(N)] = 100.0
    dn = dn + mask
    force = torch.zeros_like(old_s)
    for R in R_list:
        lo = -dn / R
        ar = torch.softmax(lo, 1)
        ac = torch.softmax(lo, 0)
        aff = torch.sqrt((ar * ac).clamp(min=1e-6))
        an = aff[:, :N]
        ap = aff[:, N:]
        sp = ap.sum(1, keepdim=True)
        sn = an.sum(1, keepdim=True)
        Rc = torch.cat([-an * sp, ap * sn], 1)
        tf = Rc @ tgt_s - Rc.sum(1)[:, None] * old_s
        force = force + tf / (tf ** 2).mean().clamp(min=1e-8).sqrt()
    goal = (old_s + force).detach()
    return ((g / si - goal) ** 2).mean()


def run_drift(gen_bs, R_list=(0.06, 0.2), real_bs=128, seed=42, n_iters=8000):
    """Train under the drifting force field (its home is small batch; collapses at large)."""
    torch.manual_seed(seed)
    P = torch.linalg.qr(torch.randn(D, 2))[0]
    realD = real_tr @ P.t()
    Nr = len(realD)
    gen = MLP(D)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters, eta_min=1e-4)
    for _ in range(n_iters):
        z = torch.randn(gen_bs, ZDIM)
        g = gen(z)
        idx = torch.randperm(Nr)[:real_bs]
        real = realD[idx]
        opt.zero_grad()
        faithful_mb(g, real, R_list).backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
        opt.step()
        sch.step()
    with torch.no_grad():
        p2 = gen(torch.randn(4000, ZDIM)) @ P
    return (p2,) + metrics(p2)


def sqrtm_psd(C, eps=1e-8):
    ev, evec = torch.linalg.eigh(C)
    ev = ev.clamp(min=eps)
    return (evec * ev.sqrt()) @ evec.t()


def run_fd(bs, seed=42, n_iters=8000):
    """Train under the Gaussian 2-Wasserstein (Frechet) on a frozen global mean + covariance."""
    torch.manual_seed(seed)
    P = torch.linalg.qr(torch.randn(D, 2))[0]
    realD = real_tr @ P.t()
    Nr = len(realD)
    mu_r = realD.mean(0)
    dr = realD - mu_r
    Cov_r = (dr.t() @ dr) / (Nr - 1)
    gen = MLP(D)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters, eta_min=1e-4)
    for _ in range(n_iters):
        z = torch.randn(bs, ZDIM)
        g = gen(z)
        opt.zero_grad()
        mu_g = g.mean(0)
        dg = g - mu_g
        Cov_g = (dg.t() @ dg) / (bs - 1)
        diff = ((mu_g - mu_r) ** 2).sum()
        Cg = sqrtm_psd(Cov_g)
        Mx = sqrtm_psd(Cg @ Cov_r @ Cg)
        loss = diff + torch.trace(Cov_g + Cov_r - 2 * Mx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 2.0)
        opt.step()
        sch.step()
    with torch.no_grad():
        p2 = gen(torch.randn(4000, ZDIM)) @ P
    return (p2,) + metrics(p2)


def train_distance(method, bs, seed=42, n_iters=8000):
    """Dispatch to the trainer for ``method`` in {fd, sw, drift, rff, exact, nystrom}.

    Threads are pinned (default 4, env ``RDM_TOY_THREADS``) so the parallel-reduction order
    matches the published fixture and seed 42/0 reproduces the EXPECTED recall / medDist.
    """
    torch.set_num_threads(int(os.environ.get("RDM_TOY_THREADS", "4")))
    if method == "fd":
        return run_fd(bs, seed=seed, n_iters=n_iters)
    if method == "drift":
        return run_drift(bs, seed=seed, n_iters=n_iters)
    return run_kernel(bs, method, seed=seed, n_iters=n_iters)

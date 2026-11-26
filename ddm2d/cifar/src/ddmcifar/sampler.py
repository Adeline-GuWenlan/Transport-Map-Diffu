"""Samplers: Algorithm 2 of arXiv 2502.02483 v3 (Eq. (4) bridge, churn 1) for arms B / C, and the same bridge with the
deterministic x0_hat = x_t + t v_hat for arm A. Bridge coefficients, grid and time clamp are ddm2d's (`ddm2d.schedule`,
`ddm2d.samplers.sample`): uniform grid t_k = k/N, denoiser at clamp(t_{k+1}, safety, 1 - safety), bridge at the true grid times,
last step to s = 0 (std 0, output = last x0_hat). One network call per NFE for every arm.

Sign check for A: x_t = (1 - t) x_0 + t x_1 and v = x_0 - x_1 give x_t + t v = (1 - t) x_0 + t x_1 + t x_0 - t x_1 = x_0
(tests/test_cifar.py::test_A_x0_from_v). The network is told t_in (the clamped time), and x0_hat uses the same t_in.

Randomness: three CPU generators seeded from `seed` (X_1, xi, bridge noise z), so arms A and B/C share X_1 and z for the same seed,
and the draws do not depend on the device or the batch size used for the network calls.
"""
from __future__ import annotations

import torch

from ddm2d.schedule import bridge_coeffs, sampling_grid


def x0_hat(model, t_in: float, x: torch.Tensor, c: torch.Tensor, xi: torch.Tensor | None, bf16: bool = False):
    """x0 prediction (float32) for a batch x at the scalar time t_in. xi is used only by the stochastic arms."""
    t = torch.full((x.shape[0],), float(t_in), device=x.device)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16 and x.device.type == "cuda"):
        out = model(t, x, c, xi) if model.stochastic else model(t, x, c)
    out = out.float()
    return out if model.stochastic else x + float(t_in) * out


class _Gens:
    def __init__(self, seed: int):
        self.init = torch.Generator().manual_seed(seed * 1000 + 1)
        self.xi = torch.Generator().manual_seed(seed * 1000 + 2)
        self.z = torch.Generator().manual_seed(seed * 1000 + 3)


@torch.no_grad()
def sample(model, labels: torch.Tensor, n_steps: int, seed: int = 0, churn: float = 1.0, safety: float = 0.01,
           device="cpu", batch: int = 1000, bf16: bool = False, return_nfe: bool = False):
    """Images (float32, CPU, unclamped) for the given labels, with n_steps network calls each. Deterministic given seed:
    the whole set is drawn in fixed CPU-generator order, then evaluated on `device` in batches of `batch`."""
    model.eval()
    n = len(labels)
    shape = (n, 3, model.input_size, model.input_size)
    g = _Gens(seed)
    grid = sampling_grid(n_steps)
    x = torch.randn(shape, generator=g.init)
    nfe = 0
    for k in range(n_steps - 1, -1, -1):
        s, t = grid[k], grid[k + 1]
        t_in = min(max(t, safety), 1.0 - safety)
        xi = torch.randn(shape, generator=g.xi) if model.stochastic else None
        x0 = torch.empty(shape)
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            x0[sl] = x0_hat(model, t_in, x[sl].to(device), labels[sl].to(device),
                            xi[sl].to(device) if xi is not None else None, bf16).cpu()
        nfe += 1
        c_xt, c_x0, std = bridge_coeffs(s, t, churn)
        z = torch.randn(shape, generator=g.z)
        x = c_xt * x + c_x0 * x0 + std * z
    return (x, nfe) if return_nfe else x


@torch.no_grad()
def xi_spread(model, x0: torch.Tensor, c: torch.Tensor, ts, n_draws: int = 8, seed: int = 0, device="cpu",
              bf16: bool = False, batch: int = 256):
    """Per-pixel std (unbiased, over n_draws xi) of x0_hat(t, x_t, xi), averaged over pixels and images, for each t in ts.
    x_t = (1 - t) x0 + t z with fixed z per (t, image). Returns {t: spread}. 0 by construction for arm A."""
    model.eval()
    gz = torch.Generator().manual_seed(seed * 1000 + 11)
    gx = torch.Generator().manual_seed(seed * 1000 + 12)
    out = {}
    for t in ts:
        z = torch.randn(x0.shape, generator=gz)
        xt = (1 - t) * x0 + t * z
        if not model.stochastic:
            out[float(t)] = 0.0
            continue
        draws = []
        for _ in range(n_draws):
            xi = torch.randn(x0.shape, generator=gx)
            d = [x0_hat(model, t, xt[i:i + batch].to(device), c[i:i + batch].to(device), xi[i:i + batch].to(device),
                        bf16).cpu() for i in range(0, len(x0), batch)]
            draws.append(torch.cat(d))
        out[float(t)] = float(torch.stack(draws).std(0, unbiased=True).mean())
    return out

"""Check (e): what PyTorch returns for the gradient of ||u||^beta at u = 0, and the NaN trap
of computing the full pairwise matrix (zero diagonal) and masking the diagonal afterwards.
Output: results/check_e.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import torch_pow_dist

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
out = {"torch_version": torch.__version__}


def grad_of(fn):
    x = torch.tensor([[0.3, -0.2], [1.0, 2.0], [0.3, -0.2]], requires_grad=True)  # rows 0 and 2 coincide
    y = fn(x)
    y.backward()
    g = x.grad
    return dict(value=float(y.detach()), grad=g.tolist(), has_nan=bool(torch.isnan(g).any()))


def offdiag(K):
    m = K.shape[0]
    return (K.sum() - torch.diagonal(K).sum()) / (m * (m - 1))


mask = 1 - torch.eye(3)
# 1. norm of an exactly-zero vector
u = torch.zeros(2, requires_grad=True)
torch.linalg.vector_norm(u).backward()
out["vector_norm_at_0_grad"] = u.grad.tolist()
u = torch.zeros(2, requires_grad=True)
((u * u).sum().sqrt()).backward()
out["sqrt_sum_sq_at_0_grad"] = u.grad.tolist()
# 2. pairwise matrices that contain zeros (diagonal, and the duplicated row pair 0/2)
out["naive_sqrt_full_matrix_masked"] = grad_of(
    lambda x: (((x[:, None] - x[None]) ** 2).sum(-1).sqrt() * mask).sum())
out["vector_norm_full_matrix_masked"] = grad_of(
    lambda x: (torch.linalg.vector_norm(x[:, None] - x[None], dim=-1) * mask).sum())
out["cdist_full_matrix_masked"] = grad_of(lambda x: (torch.cdist(x, x) * mask).sum())
out["cdist_beta0.5_masked"] = grad_of(lambda x: ((torch.cdist(x, x) ** 0.5) * mask).sum())
out["safe_pow_dist_beta1"] = grad_of(lambda x: offdiag(torch_pow_dist(x, x, 1.0)))
out["safe_pow_dist_beta0.5"] = grad_of(lambda x: offdiag(torch_pow_dist(x, x, 0.5)))
out["safe_pow_dist_beta1.5"] = grad_of(lambda x: offdiag(torch_pow_dist(x, x, 1.5)))
with open(os.path.join(RES, "check_e.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))

# 3. torch.cdist on a batch large enough for its matmul path (> 25 rows), float32, with an exact duplicate
torch.manual_seed(0)
x = torch.randn(64, 3072)
x = torch.cat([x, x[:1]], 0).requires_grad_(True)       # row 64 duplicates row 0
K = torch.cdist(x, x)                                    # default compute_mode
diag = torch.diagonal(K)
mask = 1 - torch.eye(65)
(K * mask).sum().backward()
out["cdist_mm_path_65x3072"] = dict(max_abs_diag=float(diag.abs().max()), dup_pair_dist=float(K[0, 64]),
                                    grad_has_nan=bool(torch.isnan(x.grad).any()))
x2 = x.detach().clone().requires_grad_(True)
K2 = torch.cdist(x2, x2, compute_mode="donot_use_mm_for_euclid_dist")
(K2 * mask).sum().backward()
out["cdist_exact_path_65x3072"] = dict(max_abs_diag=float(torch.diagonal(K2).abs().max()),
                                       dup_pair_dist=float(K2[0, 64]), grad_has_nan=bool(torch.isnan(x2.grad).any()),
                                       max_abs_offdiag_diff_vs_mm=float(((K2 - K) * mask).abs().max()))
with open(os.path.join(RES, "check_e.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("cdist_mm_path_65x3072", "cdist_exact_path_65x3072")}, indent=1))

from __future__ import annotations

from typing import Iterable


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("my_work.target_guidance.losses requires PyTorch to be installed.")


def _normalize_dims(rank: int, dims: Iterable[int]) -> tuple[int, ...]:
    normalized = []
    for dim in dims:
        normalized.append(dim if dim >= 0 else rank + dim)
    return tuple(normalized)


def _flatten_reduce_dims(tensor, dims: tuple[int, ...]):
    keep_dims = [idx for idx in range(tensor.dim()) if idx not in dims]
    permute_order = keep_dims + list(dims)
    transposed = tensor.permute(*permute_order)
    kept_shape = [tensor.shape[idx] for idx in keep_dims]
    reduced_size = 1
    for dim in dims:
        reduced_size *= tensor.shape[dim]
    return transposed.reshape(*kept_shape, reduced_size)


def torch_scr(core, ring, noise_floor: float = 1e-8, dims: tuple[int, int] = (-2, -1)):
    _require_torch()
    reduce_dims = _normalize_dims(core.dim(), dims)
    core_max = torch.amax(core, dim=reduce_dims)
    ring_mean = torch.mean(ring, dim=reduce_dims)
    ring_std = torch.std(ring, dim=reduce_dims, unbiased=False)
    stable_std = torch.clamp(ring_std, min=noise_floor)
    return (core_max - ring_mean) / stable_std


def torch_hoyer(core, ring=None, subtract_ring_mean: bool = True, dims: tuple[int, int] = (-2, -1)):
    _require_torch()
    reduce_dims = _normalize_dims(core.dim(), dims)
    centered = core
    if subtract_ring_mean and ring is not None:
        ring_mean = torch.mean(ring, dim=reduce_dims, keepdim=True)
        centered = core - ring_mean

    flattened = _flatten_reduce_dims(centered, reduce_dims)
    l1 = torch.sum(torch.abs(flattened), dim=-1)
    l2 = torch.linalg.norm(flattened, ord=2, dim=-1)

    n = flattened.shape[-1]
    sqrt_n = float(n) ** 0.5
    safe_ratio = torch.where(l2 > 0.0, l1 / torch.clamp(l2, min=1e-12), torch.zeros_like(l1))
    hoyer = (sqrt_n - safe_ratio) / (sqrt_n - 1.0)
    return torch.where(l2 > 0.0, hoyer, torch.zeros_like(hoyer))


def target_score_tensor(
    scr,
    hoyer,
    scr_soft_threshold: float = 2.0,
    scr_hard_threshold: float = 3.5,
    hoyer_weight: float = 1.0,
):
    _require_torch()
    return torch.relu(scr - scr_hard_threshold) + hoyer_weight * torch.relu(scr - scr_soft_threshold) * hoyer


def logsumexp_guidance_loss(score_map, beta: float = 10.0):
    _require_torch()
    flat_scores = score_map.reshape(score_map.shape[0], -1) if score_map.dim() > 1 else score_map.reshape(1, -1)
    return -(1.0 / beta) * torch.logsumexp(beta * flat_scores, dim=-1).mean()


def preservation_loss(
    sr_core,
    sr_ring,
    base_core,
    base_ring,
    noise_floor: float = 1e-8,
    scr_weight: float = 1.0,
    hoyer_weight: float = 0.25,
):
    _require_torch()
    sr_scr = torch_scr(sr_core, sr_ring, noise_floor=noise_floor)
    base_scr = torch_scr(base_core, base_ring, noise_floor=noise_floor)
    sr_hoyer = torch_hoyer(sr_core, sr_ring)
    base_hoyer = torch_hoyer(base_core, base_ring)

    scr_penalty = torch.relu(base_scr - sr_scr)
    hoyer_penalty = torch.relu(base_hoyer - sr_hoyer)
    return scr_weight * scr_penalty.mean() + hoyer_weight * hoyer_penalty.mean()


def target_aware_sr_loss(
    diffusion_loss,
    sr_core,
    sr_ring,
    base_core,
    base_ring,
    noise_floor: float = 1e-8,
    scr_weight: float = 0.1,
    hoyer_weight: float = 0.025,
):
    preserve = preservation_loss(
        sr_core=sr_core,
        sr_ring=sr_ring,
        base_core=base_core,
        base_ring=base_ring,
        noise_floor=noise_floor,
        scr_weight=scr_weight,
        hoyer_weight=hoyer_weight,
    )
    return diffusion_loss + preserve
from __future__ import annotations

import numpy as np


def compute_scr(core_patch: np.ndarray, ring_patch: np.ndarray, noise_floor: float = 1e-8) -> float:
    core_f = np.asarray(core_patch, dtype=np.float32)
    ring_f = np.asarray(ring_patch, dtype=np.float32)

    core_max = float(np.max(core_f))
    ring_mean = float(np.mean(ring_f))
    ring_std = float(np.std(ring_f))
    stable_std = max(ring_std, float(noise_floor))
    return (core_max - ring_mean) / stable_std


def compute_hoyer(
    core_patch: np.ndarray,
    ring_patch: np.ndarray | None = None,
    subtract_ring_mean: bool = True,
) -> float:
    core_f = np.asarray(core_patch, dtype=np.float32).reshape(-1)

    if subtract_ring_mean and ring_patch is not None:
        ring_mean = float(np.mean(np.asarray(ring_patch, dtype=np.float32)))
        core_f = core_f - ring_mean

    l1 = float(np.sum(np.abs(core_f)))
    l2 = float(np.linalg.norm(core_f, ord=2))
    if l2 <= 0.0:
        return 0.0

    sqrt_n = float(np.sqrt(core_f.size))
    return (sqrt_n - (l1 / l2)) / (sqrt_n - 1.0)


def score_patch(
    scr: float,
    hoyer: float,
    scr_soft_threshold: float = 2.0,
    scr_hard_threshold: float = 3.5,
    hoyer_weight: float = 1.0,
) -> float:
    scr = float(scr)
    hoyer = float(hoyer)
    soft_gate = max(scr - scr_soft_threshold, 0.0)
    hard_gate = max(scr - scr_hard_threshold, 0.0)
    return hard_gate + hoyer_weight * soft_gate * hoyer


def score_patch_map(
    scr_map: np.ndarray,
    hoyer_map: np.ndarray,
    scr_soft_threshold: float = 2.0,
    scr_hard_threshold: float = 3.5,
    hoyer_weight: float = 1.0,
) -> np.ndarray:
    scr_f = np.asarray(scr_map, dtype=np.float32)
    hoyer_f = np.asarray(hoyer_map, dtype=np.float32)
    soft_gate = np.maximum(scr_f - np.float32(scr_soft_threshold), 0.0)
    hard_gate = np.maximum(scr_f - np.float32(scr_hard_threshold), 0.0)
    return hard_gate + np.float32(hoyer_weight) * soft_gate * hoyer_f
# target_guidance

Reusable SCR/Hoyer utilities for target-aware infrared SR.

Included pieces:

- `compute_scr`: local contrast / SCR using core max, ring mean, and stable ring std
- `compute_hoyer`: Hoyer sparsity on the core, optionally after subtracting the ring mean
- `score_patch`: simple zero-shot score
- `torch_scr` and `torch_hoyer`: PyTorch versions for training or guidance
- `logsumexp_guidance_loss`: spatial LogSumExp aggregation over a score map
- `preservation_loss`: penalizes SR outputs that reduce target SCR or Hoyer relative to baseline
- `target_aware_sr_loss`: adds the preservation term to a diffusion loss

Example heuristic score:

```python
score = score_patch(scr, hoyer, scr_soft_threshold=2.0, scr_hard_threshold=3.5, hoyer_weight=1.0)
```

Example diffusion-side loss:

```python
loss = target_aware_sr_loss(
    diffusion_loss=diff_loss,
    sr_core=sr_core,
    sr_ring=sr_ring,
    base_core=base_core,
    base_ring=base_ring,
    noise_floor=1e-8,
)
```
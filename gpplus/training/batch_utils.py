"""Helpers for batched multi-init training (leading init batch dimension)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..config import logger


def wave_sizes(num_inits: int, init_batch_size: int) -> list[int]:
    """Split ``num_inits`` into sequential wave sizes of at most ``init_batch_size``."""
    if num_inits < 1:
        raise ValueError(f"num_inits must be >= 1, got {num_inits}.")
    if init_batch_size < 1:
        raise ValueError(f"init_batch_size must be >= 1, got {init_batch_size}.")
    batch = min(int(init_batch_size), int(num_inits))
    sizes: list[int] = []
    remaining = int(num_inits)
    while remaining > 0:
        wave = min(batch, remaining)
        sizes.append(wave)
        remaining -= wave
    return sizes


def resolve_batch_shape(num_inits: int, batch_shape: torch.Size | None = None) -> torch.Size:
    """Return ``torch.Size([num_inits])`` when multi-init batching is requested."""
    if batch_shape is not None:
        if len(batch_shape) != 1 or int(batch_shape[0]) != int(num_inits):
            raise ValueError(
                f"batch_shape={batch_shape} must be torch.Size([{num_inits}]) for batched multi-init."
            )
        return batch_shape
    if num_inits < 1:
        raise ValueError(f"num_inits must be >= 1, got {num_inits}.")
    return torch.Size([num_inits]) if num_inits > 1 else torch.Size([])


def model_init_batch_size(model: nn.Module) -> int:
    """Infer leading init-batch size from model, or 1 if unbatched."""
    bs = getattr(model, "batch_shape", None)
    if bs is not None and len(bs) > 0:
        return int(bs[0])
    for module in model.modules():
        bs = getattr(module, "batch_shape", None)
        if bs is not None and len(bs) > 0:
            return int(bs[0])
    return 1


def slice_state_dict(
    state_dict: dict[str, Any],
    init_index: int,
    batch_size: int | None = None,
    reference_state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Drop the leading init-batch dimension from a batched ``state_dict``.

    When ``reference_state_dict`` (unbatched shapes) is provided, a tensor is
    sliced only if ``value.shape == (batch_size, *ref.shape)``. This avoids
    mistaking task dims (e.g. ``num_tasks == num_inits``) for the init batch.
    """
    if reference_state_dict is not None:
        out: dict[str, Any] = {}
        for key, ref in reference_state_dict.items():
            if key not in state_dict:
                continue
            value = state_dict[key]
            if not torch.is_tensor(value):
                out[key] = value
                continue
            if value.shape == ref.shape:
                out[key] = value.detach().clone()
            elif (
                batch_size is not None
                and value.dim() == ref.dim() + 1
                and value.shape[0] == batch_size
                and tuple(value.shape[1:]) == tuple(ref.shape)
            ):
                out[key] = value[init_index].detach().clone()
            elif (
                batch_size is not None
                and ref.dim() == 0
                and value.dim() == 1
                and value.shape[0] == batch_size
            ):
                out[key] = value[init_index].detach().clone()
            else:
                # Leave unmatched keys out; caller may remap (e.g. LRNN nets).
                continue
        # Preserve LRNN per-init nets for later remapping.
        for key, value in state_dict.items():
            if "feature_nets." in key:
                out[key] = value.detach().clone() if torch.is_tensor(value) else value
        return out

    if batch_size is None:
        counts: dict[int, int] = {}
        for v in state_dict.values():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] > 1:
                counts[int(v.shape[0])] = counts.get(int(v.shape[0]), 0) + 1
        if counts:
            batch_size = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    out = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if batch_size is not None and value.dim() >= 1 and value.shape[0] == batch_size:
            out[key] = value[init_index].detach().clone()
        else:
            out[key] = value.detach().clone()
    return out


def select_best_init_state_dict(
    state_dict: dict[str, Any],
    losses: Tensor,
    reference_state_dict: dict[str, Any] | None = None,
) -> tuple[int, float, dict[str, Any]]:
    """Return ``(best_index, best_loss, unbatched_state_dict)``."""
    if losses.numel() == 0:
        raise ValueError("losses must be non-empty.")
    best_index = int(torch.argmin(losses).item())
    best_loss = float(losses[best_index].item())
    batch_size = int(losses.numel())
    return (
        best_index,
        best_loss,
        slice_state_dict(
            state_dict,
            best_index,
            batch_size=batch_size,
            reference_state_dict=reference_state_dict,
        ),
    )


def stack_init_state_dicts(
    state_dicts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack unbatched per-init state dicts into a leading batch dimension."""
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty.")
    keys = state_dicts[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        values = [sd[key] for sd in state_dicts]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values[0]
    return out


def expand_noise_for_features(noise: Tensor, feature_ndim: int) -> Tensor:
    """Broadcast noise ``(B,)`` / ``()`` against feature mats with ``feature_ndim`` trailing dims."""
    while noise.dim() < feature_ndim:
        noise = noise.unsqueeze(-1)
    return noise


def reduce_sum_data(t: Tensor, y_ndim: int = 1) -> Tensor:
    """Sum over the last ``y_ndim`` dims, keeping any leading batch dims."""
    if y_ndim <= 0:
        return t
    dims = tuple(range(t.dim() - y_ndim, t.dim()))
    return t.sum(dim=dims)


def assert_batched_trainable_parameters(model: nn.Module, batch_size: int) -> None:
    """
    Fail fast if any trainable parameter lacks a true leading init-batch dim.

    Compares against an unbatched architecture shell so task dims that happen to
    equal ``batch_size`` (e.g. ``num_tasks == num_inits``) are not mistaken for
    an init batch. Shared unbatched trainable params couple inits under
    ``loss.sum()``. Per-init LRNN nets ``feature_nets.{i}.*`` are allowlisted.
    """
    if batch_size <= 1:
        return

    try:
        shell = build_unbatched_model_shell(model)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"train_mode='batched' could not build an unbatched shell to validate "
            f"batch dims ({type(exc).__name__}: {exc})."
        ) from exc

    ref_params = dict(shell.named_parameters())
    bad: list[tuple[str, tuple[int, ...], tuple[int, ...] | None]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.numel() == 0:
            continue
        if "feature_nets." in name:
            continue
        ref = ref_params.get(name)
        if ref is None:
            # Batched-only key (e.g. remapped); require leading batch dim.
            if param.dim() < 1 or int(param.shape[0]) != int(batch_size):
                bad.append((name, tuple(param.shape), None))
            continue
        expected_leading = (batch_size, *tuple(ref.shape))
        ok = tuple(param.shape) == expected_leading or (
            ref.dim() == 0 and tuple(param.shape) == (batch_size,)
        )
        if not ok:
            bad.append((name, tuple(param.shape), tuple(ref.shape)))
    if bad:
        details = ", ".join(f"{n}{s} (unbatched{r})" for n, s, r in bad[:8])
        more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        raise ValueError(
            f"train_mode='batched' requires every trainable parameter to have "
            f"leading dim batch_size={batch_size} relative to the unbatched model. "
            f"Unbatched/shared params couple inits and make init_batch_size affect "
            f"training quality. Offenders: {details}{more}."
        )


def snapshot_init_into_state_dict(
    best_state: dict[str, Any],
    model: nn.Module,
    init_index: int,
    batch_size: int,
) -> None:
    """Copy live tensors for one init into ``best_state`` (batched slices / feature nets)."""
    cur = model.state_dict()
    net_token = f"feature_nets.{init_index}."
    for key, value in cur.items():
        if not torch.is_tensor(value):
            best_state[key] = value
            continue
        if value.dim() >= 1 and value.shape[0] == batch_size:
            if key not in best_state or not torch.is_tensor(best_state[key]):
                best_state[key] = value.detach().clone()
            else:
                # In-place copy_ into the existing buffer (avoid rebinding views).
                best_state[key][init_index].copy_(value[init_index].detach())
        elif net_token in key:
            best_state[key] = value.detach().clone()
        # Do not refresh shared unbatched tensors (would cross-contaminate inits).


def restore_init_from_state_dict(
    model: nn.Module,
    best_state: dict[str, Any],
    init_index: int,
    batch_size: int,
) -> None:
    """Write ``best_state`` slice ``init_index`` back into the live model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in best_state or not torch.is_tensor(best_state[name]):
                continue
            src = best_state[name]
            if (
                param.dim() >= 1
                and param.shape[0] == batch_size
                and src.dim() >= 1
                and src.shape[0] == batch_size
            ):
                param.data[init_index].copy_(src[init_index])
            elif f"feature_nets.{init_index}." in name:
                param.data.copy_(src)
        for name, buf in model.named_buffers():
            if name not in best_state or not torch.is_tensor(best_state[name]):
                continue
            src = best_state[name]
            if (
                buf.dim() >= 1
                and buf.shape[0] == batch_size
                and src.dim() >= 1
                and src.shape[0] == batch_size
            ):
                buf.data[init_index].copy_(src[init_index])


def zero_grads_for_inactive_inits(
    model: nn.Module,
    inactive: Tensor,
    batch_size: int,
) -> None:
    """Zero parameter grads for inactive init indices (leading batch dim or feature_nets)."""
    inactive_idx = torch.where(inactive)[0].tolist()
    if not inactive_idx:
        return
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if param.dim() >= 1 and param.shape[0] == batch_size:
            param.grad[inactive_idx] = 0
            continue
        for i in inactive_idx:
            if f"feature_nets.{i}." in name:
                param.grad.zero_()


def clear_optimizer_state_for_inits(
    optimizer: torch.optim.Optimizer,
    init_indices: list[int],
    batch_size: int,
    *,
    model: nn.Module | None = None,
) -> None:
    """Zero Adam moments for deactivated init slices (and LRNN per-init nets)."""
    if not init_indices:
        return
    net_prefixes = tuple(f"feature_nets.{i}." for i in init_indices)
    named: dict[int, str] = {}
    if model is not None:
        named = {id(p): n for n, p in model.named_parameters()}

    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            name = named.get(id(param), "")
            clear_whole = bool(name) and any(p in name for p in net_prefixes)
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                buf = state.get(key)
                if buf is None or not torch.is_tensor(buf):
                    continue
                if clear_whole:
                    buf.zero_()
                elif buf.dim() >= 1 and buf.shape[0] == batch_size:
                    buf[init_indices] = 0


def log_batched_training_banner(num_inits: int, device: torch.device) -> None:
    logger.info(
        "Batched multi-init training: num_inits=%s on device=%s (single model, leading batch dim).",
        num_inits,
        device,
    )


def _remap_lrnn_feature_net_keys(state_dict: dict[str, Any], init_index: int) -> dict[str, Any]:
    """Map ``feature_nets.{i}.*`` winner keys onto unbatched ``feature_net.*``."""
    prefix = f"covar_module.feature_nets.{init_index}."
    alt_prefix = f"feature_nets.{init_index}."
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("covar_module.feature_nets.") or key.startswith("feature_nets."):
            if key.startswith(prefix):
                out["covar_module.feature_net." + key[len(prefix) :]] = value
            elif key.startswith(alt_prefix):
                out["feature_net." + key[len(alt_prefix) :]] = value
            # drop other inits' nets
            continue
        out[key] = value
    return out


def build_unbatched_model_shell(batched_model: nn.Module) -> nn.Module:
    """Construct an empty unbatched model matching ``batched_model``'s architecture."""
    from ..models.gpr import GPR
    from ..models.lrnn_gpr import LRNNGPR
    from ..models.mtgpr import MTGPR
    from ..models.rff_gpr import RFFGPR
    from ..models.rff_mtgpr import RFFMTGPR

    train_x = batched_model.train_inputs[0]
    train_y = batched_model.train_targets
    if torch.is_tensor(train_x) and train_x.dim() >= 3 and train_x.shape[0] == 1:
        train_x = train_x.squeeze(0)
    if (
        torch.is_tensor(train_y)
        and train_y.dim() == 2
        and train_y.shape[0] == 1
        and not isinstance(batched_model, (MTGPR, RFFMTGPR))
    ):
        train_y = train_y.squeeze(0)

    ref = next(batched_model.parameters())
    device, dtype = ref.device, ref.dtype

    if isinstance(batched_model, RFFGPR):
        ard = bool(batched_model._rff_kernel.ard_num_dims)
        new_model = RFFGPR(
            train_x,
            train_y,
            num_rff=batched_model.num_rff,
            ard=ard,
            rff_sampling=batched_model.rff_sampling,
            correct_sorf=bool(getattr(batched_model, "correct_sorf", False)),
            spectral_kernel=getattr(batched_model, "spectral_kernel", "rbf"),
            batch_shape=torch.Size([]),
        )
    elif isinstance(batched_model, RFFMTGPR):
        ard = bool(batched_model._rff_kernel.ard_num_dims)
        new_model = RFFMTGPR(
            train_x,
            train_y,
            num_tasks=batched_model.num_tasks,
            num_rff=batched_model.num_rff,
            ard=ard,
            rff_sampling=batched_model.rff_sampling,
            correct_sorf=bool(getattr(batched_model, "correct_sorf", False)),
            spectral_kernel=getattr(batched_model, "spectral_kernel", "rbf"),
            rank_kernel=batched_model.rank_kernel,
            rank_likelihood=batched_model.rank_likelihood,
            batch_shape=torch.Size([]),
        )
    elif isinstance(batched_model, LRNNGPR):
        new_model = LRNNGPR(
            train_x,
            train_y,
            hidden_dims=batched_model.hidden_dims,
            feature_rank=batched_model.feature_rank,
            variance_correction=bool(batched_model.variance_correction),
            batch_shape=torch.Size([]),
        )
    elif isinstance(batched_model, MTGPR):
        new_model = MTGPR(
            train_x,
            train_y,
            num_tasks=batched_model.num_tasks,
            rank_likelihood=batched_model.rank_likelihood,
            rank_kernel=batched_model.rank_kernel,
            batch_shape=torch.Size([]),
        )
    elif isinstance(batched_model, GPR):
        new_model = GPR(train_x, train_y, batch_shape=torch.Size([]))
    else:
        raise TypeError(
            f"build_unbatched_model_shell does not support model type {type(batched_model).__name__}."
        )

    new_model = new_model.to(device=device, dtype=dtype)
    new_model.batch_shape = torch.Size([])
    return new_model


def materialize_unbatched_model(
    batched_model: nn.Module,
    state_dict: dict[str, Any],
    *,
    init_index: int | None = None,
    batch_size: int | None = None,
) -> nn.Module:
    """
    Build a fresh unbatched model and load the winner weights.

    ``state_dict`` may already be unbatched, or still batched (pass ``init_index`` /
    ``batch_size`` to slice using the unbatched shell as a shape reference).
    """
    new_model = build_unbatched_model_shell(batched_model)
    ref_sd = new_model.state_dict()

    if init_index is not None and batch_size is not None:
        state = slice_state_dict(
            state_dict,
            init_index,
            batch_size=batch_size,
            reference_state_dict=ref_sd,
        )
        state = _remap_lrnn_feature_net_keys(state, init_index)
    else:
        # Already sliced / remapped by caller.
        state = _remap_lrnn_feature_net_keys(state_dict, init_index or 0)

    incompatible = new_model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        logger.warning(
            "Unbatched winner load: missing=%s unexpected=%s",
            missing,
            unexpected,
        )
    if hasattr(new_model, "invalidate_feature_cache"):
        new_model.invalidate_feature_cache()
    return new_model


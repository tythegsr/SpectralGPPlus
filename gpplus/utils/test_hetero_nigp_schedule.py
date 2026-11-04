"""Unit tests for S5 Hetero-NIGP three-phase schedule and tiny smoke train."""

from __future__ import annotations

import torch

from gpplus.models.rff_gpr import RFFGPR
from gpplus.training import GPTrainer, HeteroNIGPScheduleCallback, nigp_woodbury_mll_class


def _make_hetero_model(n=24, d=3, num_rff=8, num_rff_het=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    model = RFFGPR(
        x,
        y,
        num_rff=num_rff,
        ard=True,
        rff_sampling="rff",
        nigp=True,
        hetero_sorf=True,
        num_rff_het=num_rff_het,
        hetero_mode="additive",
        hetero_seed=seed,
    )
    model.train()
    return model, y


def _raw_y(model):
    return HeteroNIGPScheduleCallback._raw_likelihood_noise(model)


def _raw_x(model):
    return HeteroNIGPScheduleCallback._raw_input_noise(model)


def _noise_sorf_trainable(model) -> bool:
    return any(p.requires_grad for p in model.noise_sorf.parameters())


def _assert_phase(model, phase: str) -> None:
    raw_y = _raw_y(model)
    raw_x = _raw_x(model)
    if phase == "hom":
        assert model.use_hetero_noise is False
        assert model.nigp_correction_enabled is False
        assert raw_y.requires_grad is True
        assert _noise_sorf_trainable(model) is False
        assert raw_x.requires_grad is False
    elif phase == "nigp":
        assert model.use_hetero_noise is False
        assert model.nigp_correction_enabled is True
        assert raw_y.requires_grad is False
        assert _noise_sorf_trainable(model) is False
        assert raw_x.requires_grad is True
    elif phase == "het":
        assert model.use_hetero_noise is True
        assert model.nigp_correction_enabled is True
        assert raw_y.requires_grad is False
        assert _noise_sorf_trainable(model) is True
        assert raw_x.requires_grad is True
    else:
        raise AssertionError(phase)


def test_hetero_schedule_phase_flags_at_boundaries():
    """Hom / NIGP / Het flags at E_hom-1, E_hom, E_hom+E_het, E-1."""
    epochs_hom, epochs_het, total = 3, 2, 8
    nigp_unlock = epochs_hom
    het_unlock = epochs_hom + epochs_het
    model, _ = _make_hetero_model(seed=1)
    cb = HeteroNIGPScheduleCallback(
        epochs_hom=epochs_hom,
        epochs_het=epochs_het,
        total_epochs=total,
        verbose=False,
    )
    ctx = {"model": model}
    cb.on_train_start(ctx)
    _assert_phase(model, "hom")

    for epoch, expected in (
        (epochs_hom - 1, "hom"),
        (nigp_unlock, "nigp"),
        (het_unlock, "het"),
        (total - 1, "het"),
    ):
        model.nigp_force_slope_refresh = False
        cb.on_epoch_start({"model": model, "epoch": epoch})
        _assert_phase(model, expected)
        if epoch == nigp_unlock or epoch == total - 1:
            assert model.nigp_force_slope_refresh is True
        elif expected == "hom":
            assert model.nigp_force_slope_refresh is False


def test_noise_sorf_warmstart_from_scalar_raw_noise():
    model, _ = _make_hetero_model(seed=3)
    with torch.no_grad():
        # sigma^2 = 10^{-2}
        model.likelihood.noise_covar.raw_noise.fill_(
            model.likelihood.noise_covar.raw_noise_constraint.inverse_transform(
                torch.tensor(-2.0, dtype=torch.float64)
            )
        )
    target = float(model.likelihood.noise.detach().reshape(-1)[0])
    # epochs_het=0: Hom -> Het directly (still warm-starts Noise SORF).
    cb = HeteroNIGPScheduleCallback(epochs_hom=2, epochs_het=0, total_epochs=6, verbose=False)
    cb.on_train_start({"model": model})
    cb.on_epoch_start({"model": model, "epoch": 2})  # enter het
    x = model.train_inputs[0]
    het = model.sigma_y_het(x)
    assert torch.allclose(het, torch.full_like(het, target), rtol=1e-5, atol=1e-8)
    assert torch.allclose(model.noise_sorf.weight, torch.zeros_like(model.noise_sorf.weight))


def test_schedule_flags_roundtrip_state_dict():
    model, _ = _make_hetero_model(seed=4)
    model.use_hetero_noise = True
    model.nigp_correction_enabled = True
    sd = model.state_dict()
    assert "_flag_use_hetero_noise" in sd
    assert "_flag_nigp_correction_enabled" in sd
    model2, _ = _make_hetero_model(seed=4)
    assert model2.use_hetero_noise is False
    assert model2.nigp_correction_enabled is False
    model2.load_state_dict(sd)
    assert model2.use_hetero_noise is True
    assert model2.nigp_correction_enabled is True


def test_early_stop_requests_endpoint_slope_refresh():
    epochs_hom, epochs_het, total = 1, 1, 10
    nigp_unlock = epochs_hom
    model, _ = _make_hetero_model(seed=5)
    cb = HeteroNIGPScheduleCallback(
        epochs_hom=epochs_hom,
        epochs_het=epochs_het,
        total_epochs=total,
        verbose=False,
    )
    cb.on_train_start({"model": model})
    assert cb.wants_endpoint_slope_refresh_epoch() is False
    cb.on_epoch_start({"model": model, "epoch": nigp_unlock})
    assert cb._unlock_refresh_done is True
    assert cb.wants_endpoint_slope_refresh_epoch() is True
    model.nigp_force_slope_refresh = False
    cb.on_epoch_start(
        {"model": model, "epoch": nigp_unlock, "endpoint_slope_refresh": True}
    )
    assert model.nigp_force_slope_refresh is True
    assert cb._endpoint_refresh_done is True
    assert cb.wants_endpoint_slope_refresh_epoch() is False


def test_evaluate_rff_uses_hetero_nugget():
    """S4/S5 predict path must use σ_y,het on the diagonal, not scalar σ_y."""
    from gpplus.training.eval import evaluate_rff_gp_model

    g = torch.Generator().manual_seed(11)
    n, d = 20, 2
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    model = RFFGPR(
        x,
        y,
        num_rff=6,
        ard=True,
        rff_sampling="rff",
        nigp=True,
        hetero_sorf=True,
        num_rff_het=3,
        hetero_mode="additive",
        hetero_seed=11,
    )
    model.eval()
    model.use_hetero_noise = True
    model.nigp_correction_enabled = True
    # Make hetero and scalar noise clearly different.
    with torch.no_grad():
        model.noise_sorf.bias.fill_(-1.0)  # σ² ≈ 0.1
        model.likelihood.noise_covar.raw_noise.fill_(-4.0)  # much smaller

    x_te = torch.randn(8, d, generator=g, dtype=torch.float64)
    _, _, _, std_het = evaluate_rff_gp_model(model, x_te, chunk_size=0)

    model.use_hetero_noise = False
    _, _, _, std_scalar = evaluate_rff_gp_model(model, x_te, chunk_size=0)
    assert not torch.allclose(std_het, std_scalar, rtol=0, atol=0)
    assert torch.isfinite(std_het).all()


def test_effective_noise_accepts_per_point_noise_y():
    from gpplus.utils.nigp_utils import effective_noise_variance

    n, d = 5, 3
    noise_y = torch.linspace(0.01, 0.05, n, dtype=torch.float64)
    input_var = torch.ones(d, dtype=torch.float64) * 0.02
    grad = torch.ones(n, d, dtype=torch.float64)
    d_out = effective_noise_variance(noise_y, input_var, grad)
    assert d_out.shape == (n,)
    # Σ_d σ_x² (∇μ)² = 3 * 0.02 = 0.06
    expected = noise_y + 0.06
    assert torch.allclose(d_out, expected)



def test_hetero_smoke_train_short_schedule():
    """Tiny RFFGPR Adam train through all three phases."""
    epochs_hom, epochs_het, total = 2, 2, 6
    model, y = _make_hetero_model(n=16, d=2, num_rff=6, num_rff_het=3, seed=7)
    cb = HeteroNIGPScheduleCallback(
        epochs_hom=epochs_hom,
        epochs_het=epochs_het,
        total_epochs=total,
        verbose=False,
    )
    mll_class = nigp_woodbury_mll_class("dual", nigp_slope_manual=True)
    trainer = GPTrainer(
        model,
        mll_class=mll_class,
        num_epochs=total,
        num_inits=1,
        seed=7,
        device="cpu",
        dtype=torch.float64,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-2},
        callbacks=[cb],
        min_epochs=epochs_hom + epochs_het,
        parallel_verbose=0,
    )
    results = trainer.train()
    assert results is not None
    assert len(results) >= 1
    best = results[0]
    assert best.get("error") is None, best.get("error")
    assert "loss" in best and "state_dict" in best
    assert torch.isfinite(torch.as_tensor(best["loss"]))
    # Schedule flags are runtime (not in state_dict); mirror checkpoint restore.
    model.use_hetero_noise = True
    model.nigp_correction_enabled = True
    het = model.sigma_y_het(model.train_inputs[0])
    assert het.shape == (model.train_inputs[0].shape[0],)
    assert torch.isfinite(het).all()
    assert (het > 0).all()

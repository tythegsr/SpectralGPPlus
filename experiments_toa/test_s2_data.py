"""Tests for S2 11-QoI data loading and band-config parsing."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pytest
import torch

from experiments_toa.s2_cli import add_s2_common_args, parse_task_names, selected_task_names
from experiments_toa.s2_bands import (
    band_config_metadata,
    default_keep_indices,
    expand_band_spec,
    load_task_band_config,
    normalize_task_input_band_indices,
)
from experiments_toa.s2_constants import (
    S2_DEFAULT_BAND_CONFIG_PATH,
    S2_DEFAULT_DATA_PATH,
    S2_DEFAULT_DROP_INDICES,
    S2_INPUT_DIM,
    S2_TASK_NAMES,
)
from experiments_toa.s2_data import load_s2_arrays, load_s2_toa_data
from experiments_toa.s2_utils import select_bands


def test_expand_band_spec_ranges_and_indices():
    bands = expand_band_spec([[0, 2], 5, [7, 8]])
    assert bands == [0, 1, 2, 5, 7, 8]


def test_expand_band_spec_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        expand_band_spec([S2_INPUT_DIM])


def test_default_keep_excludes_drop_indices():
    keep = set(default_keep_indices())
    assert keep.isdisjoint(set(S2_DEFAULT_DROP_INDICES))
    assert len(keep) == S2_INPUT_DIM - len(S2_DEFAULT_DROP_INDICES)


def test_default_band_config_loads_all_tasks():
    cfg = load_task_band_config(S2_DEFAULT_BAND_CONFIG_PATH)
    assert list(cfg.keys()) == list(S2_TASK_NAMES)
    for name, bands in cfg.items():
        assert bands, f"{name} must select at least one band"
        assert len(bands) == len(set(bands))
        assert all(0 <= idx < S2_INPUT_DIM for idx in bands)


def test_normalize_task_input_band_indices_per_task_override():
    mapping = normalize_task_input_band_indices(
        {"cos_i": [[0, 10]], "aot": [20, 21, 22]},
        task_names=["cos_i", "aot", "cwv"],
    )
    assert mapping["cos_i"] == list(range(0, 11))
    assert mapping["aot"] == [20, 21, 22]
    assert mapping["cwv"] == default_keep_indices()


def test_load_task_band_config_custom_json():
    payload = {
        "input_dim": 285,
        "default": [[0, 4]],
        "tasks": {"algae": [[10, 12]]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bands.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        cfg = load_task_band_config(path, task_names=["algae", "aot"])
    assert cfg["algae"] == [10, 11, 12]
    assert cfg["aot"] == [0, 1, 2, 3, 4]


def test_qoi_selection_accepts_sequence_and_none_means_all():
    assert parse_task_names(None) is None
    assert parse_task_names(["fsnow", "cos_i"]) == ["fsnow", "cos_i"]
    assert parse_task_names(["fsnow,cos_i", "grain_size"]) == [
        "fsnow",
        "cos_i",
        "grain_size",
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_task_names(["cos_i", "cos_i"])


def test_script_default_qoi_supports_ide_run_without_cli():
    parser = argparse.ArgumentParser()
    add_s2_common_args(parser, default_qoi=["fsnow", "algae"])
    args = parser.parse_args([])
    assert selected_task_names(args) == ["fsnow", "algae"]

    parser_all = argparse.ArgumentParser()
    add_s2_common_args(parser_all, default_qoi=None)
    assert selected_task_names(parser_all.parse_args([])) is None


def test_band_and_wavelength_mapping_is_by_qoi_not_task_order():
    payload = {
        "input_dim": 285,
        "default": [0],
        "tasks": {
            "cos_i": [9, 2, 5],
            "fsnow": [100, 7],
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bands.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        # Deliberately request a different order from the JSON object.
        cfg = load_task_band_config(path, task_names=["fsnow", "cos_i"])

    wavelengths = [float(i) + 0.25 for i in range(285)]
    meta = band_config_metadata(cfg, wavelengths_nm=wavelengths)
    assert list(cfg) == ["fsnow", "cos_i"]
    assert cfg["fsnow"] == [100, 7]
    assert meta["fsnow"]["wavelength_nm"] == [100.25, 7.25]
    assert cfg["cos_i"] == [9, 2, 5]
    assert meta["cos_i"]["wavelength_nm"] == [9.25, 2.25, 5.25]


@pytest.mark.skipif(not S2_DEFAULT_DATA_PATH.is_file(), reason="S2 NetCDF not present")
def test_load_s2_arrays_shapes():
    X, Y, wl, names, meta = load_s2_arrays()
    assert X.shape == (59000, 285)
    assert Y.shape == (59000, 11)
    assert wl.shape == (285,)
    assert names == list(S2_TASK_NAMES)
    assert meta["input_variable"] == "toa_reflectance"
    assert meta["input_dim"] == 285


@pytest.mark.skipif(not S2_DEFAULT_DATA_PATH.is_file(), reason="S2 NetCDF not present")
def test_load_s2_arrays_preserves_requested_qoi_order():
    _, Y, wl, names, _ = load_s2_arrays(task_names=["fsnow", "cos_i"])
    _, Y_ref, _, ref_names, _ = load_s2_arrays(task_names=["cos_i", "fsnow"])
    assert names == ["fsnow", "cos_i"]
    assert ref_names == ["cos_i", "fsnow"]
    assert wl.shape == (285,)
    torch.testing.assert_close(torch.as_tensor(Y[:, 0]), torch.as_tensor(Y_ref[:, 1]))
    torch.testing.assert_close(torch.as_tensor(Y[:, 1]), torch.as_tensor(Y_ref[:, 0]))


@pytest.mark.skipif(not S2_DEFAULT_DATA_PATH.is_file(), reason="S2 NetCDF not present")
def test_load_s2_splits_deterministic_and_disjoint():
    a = load_s2_toa_data(n_train=100, n_test=50, n_val=40, seed=42)
    b = load_s2_toa_data(n_train=100, n_test=50, n_val=40, seed=42)
    (
        Xtr,
        ytr,
        Xva,
        yva,
        Xte,
        yte,
        train_idx,
        val_idx,
        test_idx,
        wl,
        meta,
    ) = a
    assert torch.equal(train_idx, b[6])
    assert torch.equal(val_idx, b[7])
    assert torch.equal(test_idx, b[8])
    assert Xtr.shape == (100, 285)
    assert ytr.shape == (100, 11)
    assert Xva.shape == (40, 285)
    assert Xte.shape == (50, 285)
    assert wl.shape == (285,)
    assert meta["split_mode"] == "random"

    all_idx = torch.cat([train_idx, val_idx, test_idx])
    assert len(torch.unique(all_idx)) == len(all_idx)


@pytest.mark.skipif(not S2_DEFAULT_DATA_PATH.is_file(), reason="S2 NetCDF not present")
def test_select_bands_width_matches_config():
    X, _, _, _, _ = load_s2_arrays(task_names=["cos_i"])
    cfg = load_task_band_config(task_names=["cos_i"])
    x = torch.as_tensor(X[:8], dtype=torch.float64)
    xs = select_bands(x, cfg["cos_i"])
    assert xs.shape == (8, len(cfg["cos_i"]))


def test_s2_log_scale_forward_inverse_roundtrip():
    from experiments_toa.s2_constants import S2_LOG_SCALE_TASK_NAMES
    from experiments_toa.s2_y_transform import (
        forward_y_s2,
        inverse_y_s2,
        task_uses_log_scale,
    )
    from gpplus.utils import StandardScaler

    assert S2_LOG_SCALE_TASK_NAMES == frozenset(
        {"algae", "dust", "grain_size", "liquid_water"}
    )
    assert task_uses_log_scale("grain_size")
    assert not task_uses_log_scale("cos_i")
    assert not task_uses_log_scale("grain_size", log_scale=False)

    y = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
    y_log = forward_y_s2(y, "dust", log_scale=True)
    assert torch.allclose(y_log, torch.log(y))

    scaler = StandardScaler()
    scaler.fit(y_log.unsqueeze(-1))
    y_fit = scaler.transform(y_log.unsqueeze(-1)).squeeze(-1)
    pred_mean = y_fit
    pred_std = torch.zeros_like(pred_mean)
    point, std, lo, hi = inverse_y_s2(
        pred_mean,
        pred_std,
        pred_mean,
        pred_mean,
        task_name="dust",
        y_scaler=scaler,
        standardize_y=True,
        log_scale=True,
    )
    assert torch.allclose(point, y, rtol=1e-5, atol=1e-8)
    assert torch.allclose(lo, y, rtol=1e-5, atol=1e-8)
    assert torch.allclose(hi, y, rtol=1e-5, atol=1e-8)
    assert torch.all(std >= 0)

    y_cos = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    assert torch.equal(forward_y_s2(y_cos, "cos_i"), y_cos)

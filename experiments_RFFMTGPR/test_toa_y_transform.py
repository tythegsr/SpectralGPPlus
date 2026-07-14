"""Unit tests for TOA target transforms (log grain, logit cos_i)."""

from __future__ import annotations

import math

import torch

from toa_y_transform import (
    COS_EPS,
    forward_y,
    inverse_y_predictions,
    logit_normal_summaries_from_logit_params,
    lognormal_summaries_from_log_params,
)


def _scalar(x) -> float:
    return float(torch.as_tensor(x).reshape(-1)[0])


def test_logit_cos_bounds_on_inverse():
    mu = torch.tensor([2.0])
    sigma = torch.tensor([0.4])
    median, mean, std, lower, upper = logit_normal_summaries_from_logit_params(mu, sigma)
    assert 0.0 <= _scalar(lower) <= _scalar(upper) <= 1.0
    assert 0.0 <= _scalar(median) <= 1.0
    assert abs(_scalar(median) - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-6
    expected_lower = torch.sigmoid(torch.tensor(2.0 - 1.96 * 0.4)).item()
    expected_upper = torch.sigmoid(torch.tensor(2.0 + 1.96 * 0.4)).item()
    assert abs(_scalar(lower) - expected_lower) < 1e-6
    assert abs(_scalar(upper) - expected_upper) < 1e-6


def test_lognormal_worked_example():
    mu = torch.tensor([math.log(500.0)])
    sigma = torch.tensor([0.15])
    median, mean, mode, std, lower, upper = lognormal_summaries_from_log_params(mu, sigma)
    assert abs(_scalar(median) - 500.0) < 1.0
    assert abs(_scalar(lower) - math.exp(math.log(500.0) - 1.96 * 0.15)) < 1.0
    assert abs(_scalar(upper) - math.exp(math.log(500.0) + 1.96 * 0.15)) < 1.0
    assert _scalar(lower) > 0.0


def test_forward_inverse_roundtrip_interior():
    y = torch.tensor([[0.25, 50.0], [0.75, 800.0]])
    y_model = forward_y(y, log_grain=True, logit_cos=True)
    mu = y_model.clone()
    sigma = torch.full_like(mu, 0.05)
    lower = mu - 1.96 * sigma
    upper = mu + 1.96 * sigma
    inv = inverse_y_predictions(
        mu,
        sigma,
        lower,
        upper,
        y_scaler=None,
        standardize_y=False,
        log_grain=True,
        logit_cos=True,
        extended=True,
    )
    cos = inv.point[:, 0]
    grain = inv.point[:, 1]
    assert torch.all(cos >= 0.0)
    assert torch.all(cos <= 1.0)
    assert torch.all(grain > 0.0)
    assert torch.all(inv.lower[:, 0] >= 0.0)
    assert torch.all(inv.upper[:, 0] <= 1.0)


def test_forward_clamps_cos_boundaries():
    y = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
    y_model = forward_y(y, log_grain=True, logit_cos=True)
    assert torch.isfinite(y_model).all()
    expected_lo = math.log(COS_EPS / (1.0 - COS_EPS))
    expected_hi = math.log((1.0 - COS_EPS) / COS_EPS)
    assert abs(float(y_model[0, 0]) - expected_lo) < 1e-5
    assert abs(float(y_model[1, 0]) - expected_hi) < 1e-3

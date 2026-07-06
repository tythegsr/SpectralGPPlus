"""Unit tests for log-grain log-normal metrics and summaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpplus.utils.metrics_functions import (
    compute_crps_gaussian,
    compute_crps_lognormal,
    compute_metrics,
    compute_nlpd_lognormal,
)
from toa_y_transform import lognormal_summaries_from_log_params


def _mc_crps_lognormal(y, mu, sigma, n: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    y_samples = np.exp(mu + sigma * z1)
    y_samples2 = np.exp(mu + sigma * z2)
    return float(np.mean(np.abs(y_samples - y)) - 0.5 * np.mean(np.abs(y_samples - y_samples2)))


def test_lognormal_summaries_closed_form():
    mu = np.array([3.5, 4.0])
    sigma = np.array([0.3, 0.5])
    median, mean, mode, std, lower, upper = lognormal_summaries_from_log_params(mu, sigma)

    np.testing.assert_allclose(median, np.exp(mu))
    np.testing.assert_allclose(mean, np.exp(mu + 0.5 * sigma**2))
    np.testing.assert_allclose(mode, np.exp(mu - sigma**2))
    np.testing.assert_allclose(std, np.exp(mu) * np.sqrt(np.expm1(sigma**2)))
    np.testing.assert_allclose(lower, np.exp(mu - 1.96 * sigma))
    np.testing.assert_allclose(upper, np.exp(mu + 1.96 * sigma))


def test_compute_nlpd_lognormal_matches_manual():
    y = np.array([40.0, 55.0, 120.0])
    mu = np.log(y) + np.array([0.05, -0.02, 0.01])
    sigma = np.array([0.25, 0.3, 0.2])

    manual = (
        np.log(y)
        + np.log(sigma)
        + 0.5 * np.log(2.0 * np.pi)
        + 0.5 * ((np.log(y) - mu) ** 2) / (sigma**2)
    )
    nlpd = compute_nlpd_lognormal(y, mu, sigma)
    assert nlpd is not None
    np.testing.assert_allclose(nlpd, float(np.mean(manual)))


@pytest.mark.parametrize(
    "y,mu,sigma",
    [
        (50.0, 3.9, 0.4),
        (120.0, 4.8, 0.3),
        (30.0, 3.4, 0.6),
    ],
)
def test_compute_crps_lognormal_vs_monte_carlo(y, mu, sigma):
    mc = _mc_crps_lognormal(y, mu, sigma)
    cf = compute_crps_lognormal(np.array([y]), np.array([mu]), np.array([sigma]))
    assert abs(mc - cf) < 0.1


def test_compute_metrics_uses_lognormal_crps_not_gaussian():
    y = np.array([45.0, 60.0, 95.0])
    mu_log = np.log(y)
    sigma_log = np.full(3, 0.35)
    median = np.exp(mu_log)

    metrics_ln = compute_metrics(
        y,
        median,
        output_std=median * 0.2,
        lower_95=np.exp(mu_log - 1.96 * sigma_log),
        upper_95=np.exp(mu_log + 1.96 * sigma_log),
        log_mu=mu_log,
        log_sigma=sigma_log,
    )
    metrics_g = compute_metrics(
        y,
        median,
        output_std=median * 0.2,
        lower_95=np.exp(mu_log - 1.96 * sigma_log),
        upper_95=np.exp(mu_log + 1.96 * sigma_log),
    )

    expected_crps = compute_crps_lognormal(y, mu_log, sigma_log)
    gaussian_crps = compute_crps_gaussian(y, median, median * 0.2)

    np.testing.assert_allclose(metrics_ln["CRPS"], expected_crps)
    assert metrics_ln["CRPS"] != pytest.approx(gaussian_crps, rel=1e-3)
    assert metrics_g["CRPS"] == pytest.approx(gaussian_crps)

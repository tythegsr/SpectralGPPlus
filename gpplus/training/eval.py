from __future__ import annotations

import gpytorch
import torch
from typing import TYPE_CHECKING

from ..config import get_settings, logger

if TYPE_CHECKING:
    from ..models.rff_gpr import RFFGPR
from ..likelihoods import MultiLikelihood
from ..models.rff_gpr import _drop_singleton_batch
from ..utils.rff_utils import (
    woodbury_factor,
    woodbury_predictive_mean,
    woodbury_predictive_var_diag,
)


def evaluate_gp_model(
    model,
    test_x: torch.Tensor,
):
    """
    Evaluates the Gaussian Process model on test data.

    Args:
        model (GPModel):
            The Gaussian Process model to evaluate.
        test_x (torch.Tensor):
            Test data features.

    Returns:
        tuple:
            A tuple containing:
                - **mean** (torch.Tensor): Predictive mean for each test point.
                - **lower** (torch.Tensor): Lower confidence bound for each test point.
                - **upper** (torch.Tensor): Upper confidence bound for each test point.
                - **stddev** (torch.Tensor): Standard deviation of the predictions.
    """
    get_settings().apply()
    model.eval()

    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ),
    ):  # gpytorch.settings.fast_pred_var():
        # Make predictions
        # Option 1: Without nugget (latent function f) - follows Equation 29b structure
        # observed_pred = model(test_x)

        # Option 2: With nugget (noisy observations y) - follows Equation 31b structure
        # This adds +δI to the predictive covariance: +δ* = K_test_test - ... + +δI
        # For MultiLikelihood, we need to set test fidelity indices from test data
        # so each test point gets the correct nugget based on its source
        train_inputs = getattr(model, "train_inputs", None)
        if train_inputs and len(train_inputs) > 0:
            reference = train_inputs[0]
            test_x = test_x.to(device=reference.device, dtype=reference.dtype)

        if isinstance(model.likelihood, MultiLikelihood):
            model.likelihood.set_fidelity_indices(test_x, is_test=True)
        observed_pred = model.likelihood(model(test_x))

        # Get the mean, lower and upper confidence bounds
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        stddev = observed_pred.stddev

        logger.info("Evaluation completed.")
        return mean, lower, upper, stddev


def evaluate_rff_gp_model(
    model: RFFGPR,
    test_x: torch.Tensor,
    jitter: float = 1e-6,
    chunk_size: int = 512,
):
    """
    Evaluate an :class:`~gpplus.models.RFFGPR` model using Woodbury prediction.

    Predictions are computed in chunks of ``chunk_size`` test points so the
    Woodbury solve RHS stays ``(n_train, chunk_size)`` instead of ``(n_train, n_test)``.

    Prefer this over :func:`evaluate_gp_model` for RFF models so inference avoids
    dense n x n linear algebra.
    """
    model.eval()
    train_inputs = getattr(model, "train_inputs", None)
    if train_inputs and len(train_inputs) > 0:
        reference = train_inputs[0]
        test_x = test_x.to(device=reference.device, dtype=reference.dtype)

    n_test = test_x.shape[0]
    if n_test == 0:
        empty = test_x.new_zeros(0)
        return empty, empty, empty, empty

    with torch.no_grad():
        if chunk_size <= 0 or n_test <= chunk_size:
            mean, lower, upper = model.predict(test_x, jitter=jitter)
            stddev = (upper - lower) / 4.0
        else:
            train_x = _drop_singleton_batch(model.train_inputs[0])
            train_y = _drop_singleton_batch(model.train_targets)
            z_train = model.train_features()
            noise = model.likelihood.noise
            chol, noise_clamped = woodbury_factor(noise, z_train, jitter=jitter)
            mean_train = model.mean_module(train_x)
            if mean_train.dim() > 1 and mean_train.shape[0] == 1:
                mean_train = mean_train.squeeze(0)
            y_centered = train_y - mean_train

            mean_chunks = []
            lower_chunks = []
            upper_chunks = []
            for start in range(0, n_test, chunk_size):
                chunk_x = test_x[start : start + chunk_size]
                z_test = model.scaled_features(chunk_x)
                f_mean = woodbury_predictive_mean(
                    noise,
                    z_train,
                    z_test,
                    y_centered,
                    jitter=jitter,
                    chol=chol,
                    noise=noise_clamped,
                )
                f_mean = f_mean + model.mean_module(chunk_x)
                f_var = woodbury_predictive_var_diag(
                    noise,
                    z_train,
                    z_test,
                    jitter=jitter,
                    chol=chol,
                    noise=noise_clamped,
                )
                obs_std = (f_var + noise).clamp_min(0.0).sqrt()
                mean_chunks.append(f_mean)
                lower_chunks.append(f_mean - 2 * obs_std)
                upper_chunks.append(f_mean + 2 * obs_std)
            mean = torch.cat(mean_chunks, dim=0)
            lower = torch.cat(lower_chunks, dim=0)
            upper = torch.cat(upper_chunks, dim=0)
            stddev = (upper - lower) / 4.0

    logger.info("RFF evaluation completed.")
    return mean, lower, upper, stddev

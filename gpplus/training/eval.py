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
    WoodburyForm,
    WoodburyMtMethod,
    flatten_multitask_targets,
    unflatten_multitask_targets,
    woodbury_factor,
    woodbury_factor_dual,
    woodbury_factor_mt,
    woodbury_predictive_mean,
    woodbury_predictive_mean_dual,
    woodbury_predictive_mean_mt,
    woodbury_predictive_obs_std,
    woodbury_predictive_var_diag,
    woodbury_predictive_var_diag_dual,
    woodbury_predictive_var_diag_mt,
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

        from ..utils.nigp_utils import nigp_correction_enabled

        use_nigp = nigp_correction_enabled(model)
        if use_nigp:
            from ..utils.nigp_utils import (
                effective_noise_variance,
                exact_posterior_mean_grad_wrt_x,
            )

            latent_pred = model(test_x)
            mean = latent_pred.mean
            f_var = latent_pred.variance
            noise = model.likelihood.noise
            if noise.dim() > 1:
                noise = noise.reshape(noise.shape[0])
            with torch.enable_grad():
                grad_mu_test = exact_posterior_mean_grad_wrt_x(
                    model, test_x, noise, jitter=1e-6
                )
            d_test = effective_noise_variance(
                noise, model.input_noise_var, grad_mu_test
            )
            obs_var = (f_var + d_test).clamp_min(0.0)
            stddev = obs_var.sqrt()
            lower = mean - 2.0 * stddev
            upper = mean + 2.0 * stddev
            logger.info("Evaluation completed (exact NIGP observation noise).")
            return mean, lower, upper, stddev

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
    return_latent_var: bool = False,
    woodbury_form: WoodburyForm = "primal",
):
    """
    Evaluate an :class:`~gpplus.models.RFFGPR` model using Woodbury prediction.

    Predictions are computed in chunks of ``chunk_size`` test points so the
    Woodbury solve RHS stays ``(n_train, chunk_size)`` instead of ``(n_train, n_test)``.

    Default ``woodbury_form="primal"`` uses ``M = I + ΦᵀΦ/σ²``.
    ``woodbury_form="dual"`` factors ``Λ = ΦᵀΦ + σ² I`` (stable for small ``σ²``).

    When ``model.nigp`` is enabled, uses diag-noise Woodbury
    ``Σ = diag(d) + ΦΦᵀ`` with McHutchon effective noise on train and test
    (same correction as :meth:`~gpplus.models.rff_gpr.RFFGPR.predict`).

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
        if return_latent_var:
            return empty, empty, empty, empty, empty
        return empty, empty, empty, empty

    from ..utils.nigp_utils import nigp_correction_enabled

    use_nigp = nigp_correction_enabled(model)

    with torch.no_grad():
        train_x = _drop_singleton_batch(model.train_inputs[0])
        train_y = _drop_singleton_batch(model.train_targets)
        z_train = model.train_features()
        noise = model.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])
        mean_train = model.mean_module(train_x)
        if mean_train.dim() > 1 and mean_train.shape[0] == 1:
            mean_train = mean_train.squeeze(0)
        y_centered = train_y - mean_train

        mean_chunks = []
        lower_chunks = []
        upper_chunks = []
        f_var_chunks: list[torch.Tensor] = []
        step = n_test if chunk_size <= 0 else chunk_size

        if use_nigp:
            from ..utils.nigp_utils import (
                effective_noise_variance,
                posterior_mean_grad_wrt_x,
            )
            from ..utils.rff_utils import woodbury_posterior_weights_diag_noise

            # Gradients need an enabled graph; temporarily allow grad for ∇μ only.
            with torch.enable_grad():
                grad_mu_train = posterior_mean_grad_wrt_x(
                    model, train_x, y_centered, noise, jitter=jitter
                )
            d_train = effective_noise_variance(
                noise, model.input_noise_var, grad_mu_train
            )
            w, chol, _z_hat = woodbury_posterior_weights_diag_noise(
                d_train, z_train, y_centered, jitter=jitter
            )
            ones = z_train.new_ones(())

            for start in range(0, n_test, step):
                chunk_x = test_x[start : start + step]
                z_test = model.scaled_features(chunk_x)
                z_test_c = z_test.to(dtype=chol.dtype)
                if w.dim() == 1:
                    f_mean = (z_test_c @ w).to(dtype=z_test.dtype)
                else:
                    f_mean = (
                        (z_test_c @ w.unsqueeze(-1)).squeeze(-1).to(dtype=z_test.dtype)
                    )
                f_mean = f_mean + model.mean_module(chunk_x)
                f_var = woodbury_predictive_var_diag_dual(ones, z_test, chol=chol)
                with torch.enable_grad():
                    grad_mu_test = posterior_mean_grad_wrt_x(
                        model,
                        chunk_x,
                        y_centered,
                        noise,
                        train_x=train_x,
                        jitter=jitter,
                    )
                d_test = effective_noise_variance(
                    noise, model.input_noise_var, grad_mu_test
                )
                obs_std = woodbury_predictive_obs_std(f_var, d_test)
                mean_chunks.append(f_mean)
                lower_chunks.append(f_mean - 2 * obs_std)
                upper_chunks.append(f_mean + 2 * obs_std)
                if return_latent_var:
                    f_var_chunks.append(f_var)
        else:
            use_dual = woodbury_form == "dual"
            if use_dual:
                chol, noise_clamped, z_lin = woodbury_factor_dual(
                    noise, z_train, jitter=jitter
                )
            else:
                chol, noise_clamped = woodbury_factor(noise, z_train, jitter=jitter)
                z_lin = z_train

            for start in range(0, n_test, step):
                chunk_x = test_x[start : start + step]
                z_test = model.scaled_features(chunk_x)
                if use_dual:
                    f_mean = woodbury_predictive_mean_dual(
                        noise_clamped, z_lin, z_test, y_centered, chol=chol
                    )
                    f_var = woodbury_predictive_var_diag_dual(
                        noise_clamped, z_test, chol=chol
                    )
                else:
                    f_mean = woodbury_predictive_mean(
                        noise,
                        z_train,
                        z_test,
                        y_centered,
                        jitter=jitter,
                        chol=chol,
                        noise=noise_clamped,
                    )
                    f_var = woodbury_predictive_var_diag(
                        noise,
                        z_train,
                        z_test,
                        jitter=jitter,
                        chol=chol,
                        noise=noise_clamped,
                    )
                f_mean = f_mean + model.mean_module(chunk_x)
                obs_std = woodbury_predictive_obs_std(f_var, noise)
                mean_chunks.append(f_mean)
                lower_chunks.append(f_mean - 2 * obs_std)
                upper_chunks.append(f_mean + 2 * obs_std)
                if return_latent_var:
                    f_var_chunks.append(f_var)

        mean = torch.cat(mean_chunks, dim=0)
        lower = torch.cat(lower_chunks, dim=0)
        upper = torch.cat(upper_chunks, dim=0)
        stddev = (upper - lower) / 4.0

    logger.info(
        "RFF evaluation completed%s.",
        " (NIGP observation noise)" if use_nigp else "",
    )
    if return_latent_var:
        f_var_out = torch.cat(f_var_chunks, dim=0)
        return mean, lower, upper, stddev, f_var_out
    return mean, lower, upper, stddev


def evaluate_vi_rff_gp_model(
    model,
    test_x: torch.Tensor,
    chunk_size: int = 512,
    return_latent_var: bool = False,
):
    """
    Evaluate a :class:`~gpplus.models.vi_rff_gpr.VIRFFGPR` from its variational posterior.

    Unlike :func:`evaluate_rff_gp_model`, no Woodbury solve over the training set
    is needed: ``q(w) = N(m_w, S)`` already summarizes the data, so prediction is
    ``μ(x*) = m(x*) + φ(x*)ᵀ m_w`` and ``Var[f(x*)] = φ(x*)ᵀ S φ(x*)``.

    Observation bounds add the per-point noise ``d(x*)``, which includes the NIGP
    input-noise correction when it is active. Chunked over ``chunk_size`` test
    points so the ``(chunk, m) x (m, m)`` variance matmul stays bounded.
    """
    model.eval()
    train_inputs = getattr(model, "train_inputs", None)
    if train_inputs and len(train_inputs) > 0:
        reference = train_inputs[0]
        test_x = test_x.to(device=reference.device, dtype=reference.dtype)

    n_test = test_x.shape[0]
    if n_test == 0:
        empty = test_x.new_zeros(0)
        if return_latent_var:
            return empty, empty, empty, empty, empty
        return empty, empty, empty, empty

    from ..utils.nigp_utils import nigp_correction_enabled

    use_nigp = nigp_correction_enabled(model)
    step = n_test if chunk_size <= 0 else chunk_size

    mean_chunks = []
    lower_chunks = []
    upper_chunks = []
    f_var_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n_test, step):
            chunk_x = test_x[start : start + step]
            phi = model.scaled_features(chunk_x)
            f_mean = model.mean_module(chunk_x) + phi @ model.variational_mean
            f_var = model.variational_quad(phi).clamp_min(0.0)
            obs_std = (f_var + model.observation_noise_var(chunk_x)).clamp_min(0.0).sqrt()
            mean_chunks.append(f_mean)
            lower_chunks.append(f_mean - 2 * obs_std)
            upper_chunks.append(f_mean + 2 * obs_std)
            if return_latent_var:
                f_var_chunks.append(f_var)

        mean = torch.cat(mean_chunks, dim=0)
        lower = torch.cat(lower_chunks, dim=0)
        upper = torch.cat(upper_chunks, dim=0)
        stddev = (upper - lower) / 4.0

    logger.info(
        "Variational RFF evaluation completed%s.",
        " (NIGP observation noise)" if use_nigp else "",
    )
    if return_latent_var:
        return mean, lower, upper, stddev, torch.cat(f_var_chunks, dim=0)
    return mean, lower, upper, stddev


def evaluate_svgp_gp_model(
    model,
    test_x: torch.Tensor,
    chunk_size: int = 4096,
    return_latent_var: bool = False,
):
    """
    Evaluate an :class:`~gpplus.models.svgp_gpr.SVGPR` from its variational posterior.

    The training set plays no role at prediction time: ``q(u)`` at the inducing
    locations already summarizes it, so each chunk costs ``O(chunk * M^2)``
    regardless of ``N``. Observation bounds add the per-point noise ``d(x*)``,
    which carries the NIGP input-noise correction when it is active.
    """
    model.eval()
    reference = next(model.parameters())
    test_x = test_x.to(device=reference.device, dtype=getattr(model, "dtype", reference.dtype))

    n_test = test_x.shape[0]
    if n_test == 0:
        empty = test_x.new_zeros(0)
        if return_latent_var:
            return empty, empty, empty, empty, empty
        return empty, empty, empty, empty

    from ..utils.nigp_utils import nigp_correction_enabled

    use_nigp = nigp_correction_enabled(model)
    step = n_test if chunk_size <= 0 else chunk_size

    mean_chunks = []
    lower_chunks = []
    upper_chunks = []
    f_var_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n_test, step):
            chunk_x = test_x[start : start + step]
            dist = model(chunk_x)
            f_mean = dist.mean
            f_var = dist.variance.clamp_min(0.0)
            # grad_mu re-enables autograd internally when NIGP is on.
            obs_std = (f_var + model.observation_noise_var(chunk_x)).clamp_min(0.0).sqrt()
            mean_chunks.append(f_mean)
            lower_chunks.append(f_mean - 2 * obs_std)
            upper_chunks.append(f_mean + 2 * obs_std)
            if return_latent_var:
                f_var_chunks.append(f_var)

        mean = torch.cat(mean_chunks, dim=0)
        lower = torch.cat(lower_chunks, dim=0)
        upper = torch.cat(upper_chunks, dim=0)
        stddev = (upper - lower) / 4.0

    logger.info(
        "SVGP evaluation completed%s (M=%s inducing points).",
        " (NIGP observation noise)" if use_nigp else "",
        getattr(model, "num_inducing", "?"),
    )
    if return_latent_var:
        return mean, lower, upper, stddev, torch.cat(f_var_chunks, dim=0)
    return mean, lower, upper, stddev


def evaluate_lrnn_gp_model(
    model,
    test_x: torch.Tensor,
    jitter: float = 1e-6,
    chunk_size: int = 512,
    return_latent_var: bool = False,
    variance_correction: bool | None = None,
):
    """
    Evaluate an :class:`~gpplus.models.LRNNGPR` model using Woodbury prediction.

    Supports DBK variance correction (diagonal correction) when enabled on the model
    or passed explicitly via ``variance_correction``.
    """
    from ..utils.lrnn_utils import woodbury_predict_lrnn

    model.eval()
    if variance_correction is None:
        variance_correction = bool(getattr(model, "variance_correction", True))

    train_inputs = getattr(model, "train_inputs", None)
    if train_inputs and len(train_inputs) > 0:
        reference = train_inputs[0]
        test_x = test_x.to(device=reference.device, dtype=reference.dtype)

    n_test = test_x.shape[0]
    if n_test == 0:
        empty = test_x.new_zeros(0)
        if return_latent_var:
            return empty, empty, empty, empty, empty
        return empty, empty, empty, empty

    with torch.no_grad():
        train_x = _drop_singleton_batch(model.train_inputs[0])
        train_y = _drop_singleton_batch(model.train_targets)
        z_train = model.train_features()
        noise = model.likelihood.noise
        mean_train = model.mean_module(train_x)
        if mean_train.dim() > 1 and mean_train.shape[0] == 1:
            mean_train = mean_train.squeeze(0)
        y_centered = train_y - mean_train

        mean_chunks = []
        lower_chunks = []
        upper_chunks = []
        f_var_chunks: list[torch.Tensor] = []
        step = n_test if chunk_size <= 0 else chunk_size
        for start in range(0, n_test, step):
            chunk_x = test_x[start : start + step]
            z_test = model.scaled_features(chunk_x)
            f_mean, f_var, obs_std = woodbury_predict_lrnn(
                noise,
                z_train,
                z_test,
                y_centered,
                jitter=jitter,
                variance_correction=variance_correction,
            )
            f_mean = f_mean + model.mean_module(chunk_x)
            mean_chunks.append(f_mean)
            lower_chunks.append(f_mean - 2 * obs_std)
            upper_chunks.append(f_mean + 2 * obs_std)
            if return_latent_var:
                f_var_chunks.append(f_var)
        mean = torch.cat(mean_chunks, dim=0)
        lower = torch.cat(lower_chunks, dim=0)
        upper = torch.cat(upper_chunks, dim=0)
        stddev = (upper - lower) / 4.0

    logger.info("LRNN evaluation completed.")
    if return_latent_var:
        f_var_out = torch.cat(f_var_chunks, dim=0)
        return mean, lower, upper, stddev, f_var_out
    return mean, lower, upper, stddev


def evaluate_rff_mt_gp_model(
    model,
    test_x: torch.Tensor,
    jitter: float = 1e-6,
    chunk_size: int = 512,
    return_latent_var: bool = False,
    method: WoodburyMtMethod = "eigen",
):
    """
    Evaluate an :class:`~gpplus.models.RFFMTGPR` model using multitask Woodbury prediction.

    Default ``method="eigen"`` (alias of ``primal_eigen``). Factors the Woodbury
    middle matrix once per call and reuses it across test chunks.

    When ``model.nigp`` correction is enabled, uses per-(i,t) diag-noise Woodbury
    with McHutchon effective noise on train and test.

    Returns mean, lower, upper, stddev each of shape ``(n_test, T)``.
    If ``return_latent_var`` is True, also returns latent ``f_var`` (before adding noise).
    """
    from ..models.rff_mtgpr import RFFMTGPR
    from ..utils.nigp_utils import nigp_correction_enabled

    if not isinstance(model, RFFMTGPR):
        raise TypeError("evaluate_rff_mt_gp_model requires RFFMTGPR.")

    model.eval()
    train_inputs = getattr(model, "train_inputs", None)
    if train_inputs and len(train_inputs) > 0:
        reference = train_inputs[0]
        test_x = test_x.to(device=reference.device, dtype=reference.dtype)

    n_test = test_x.shape[0]
    num_tasks = model.num_tasks
    if n_test == 0:
        empty = test_x.new_zeros(0, num_tasks)
        if return_latent_var:
            return empty, empty, empty, empty, empty
        return empty, empty, empty, empty

    use_nigp = nigp_correction_enabled(model)

    with torch.no_grad():
        train_x = _drop_singleton_batch(model.train_inputs[0])
        train_y = _drop_singleton_batch(model.train_targets)
        n_train = train_x.shape[0]
        phi_train = model.train_spatial_features()
        r_b = model.task_psd_factor()
        mean_train = model.mean_module(train_x)
        y_centered = flatten_multitask_targets(train_y - mean_train)
        task_noises = model.task_noises()

        step = n_test if chunk_size <= 0 else chunk_size
        mean_chunks: list[torch.Tensor] = []
        lower_chunks: list[torch.Tensor] = []
        upper_chunks: list[torch.Tensor] = []
        f_var_chunks: list[torch.Tensor] = []

        if use_nigp:
            from ..utils.nigp_utils import (
                effective_noise_variance_mt,
                posterior_mean_grad_wrt_x_mt,
            )
            from ..utils.rff_utils import woodbury_predict_mt_diag_noise

            with torch.enable_grad():
                grad_mu_train = posterior_mean_grad_wrt_x_mt(
                    model, train_x, y_centered, task_noises, jitter=jitter
                )
            d_train = effective_noise_variance_mt(
                task_noises, model.input_noise_var, grad_mu_train
            )

            for start in range(0, n_test, step):
                chunk_x = test_x[start : start + step]
                phi_test = model.scaled_spatial_features(chunk_x)
                with torch.enable_grad():
                    grad_mu_test = posterior_mean_grad_wrt_x_mt(
                        model,
                        chunk_x,
                        y_centered,
                        task_noises,
                        train_x=train_x,
                        jitter=jitter,
                    )
                d_test = effective_noise_variance_mt(
                    task_noises, model.input_noise_var, grad_mu_test
                )
                f_mean, f_var, obs_std = woodbury_predict_mt_diag_noise(
                    d_train,
                    phi_train,
                    phi_test,
                    r_b,
                    n_train,
                    num_tasks,
                    y_centered,
                    jitter=jitter,
                    d_test=d_test,
                )
                f_mean = f_mean + model.mean_module(chunk_x)
                mean_chunks.append(f_mean)
                lower_chunks.append(f_mean - 2 * obs_std)
                upper_chunks.append(f_mean + 2 * obs_std)
                if return_latent_var:
                    f_var_chunks.append(f_var)
        else:
            factor, noise = woodbury_factor_mt(
                task_noises, phi_train, r_b, jitter=jitter, method=method
            )
            for start in range(0, n_test, step):
                chunk_x = test_x[start : start + step]
                phi_test = model.scaled_spatial_features(chunk_x)
                f_mean = woodbury_predictive_mean_mt(
                    task_noises,
                    phi_train,
                    phi_test,
                    r_b,
                    n_train,
                    y_centered,
                    jitter=jitter,
                    factor=factor,
                    noise=noise,
                    method=method,
                )
                f_mean = unflatten_multitask_targets(f_mean, num_tasks) + model.mean_module(
                    chunk_x
                )
                f_var = unflatten_multitask_targets(
                    woodbury_predictive_var_diag_mt(
                        task_noises,
                        phi_train,
                        phi_test,
                        r_b,
                        n_train,
                        jitter=jitter,
                        factor=factor,
                        noise=noise,
                        method=method,
                    ),
                    num_tasks,
                )
                noise_rows = task_noises.view(1, -1).expand(f_mean.shape[0], -1)
                obs_std = woodbury_predictive_obs_std(f_var, noise_rows)
                mean_chunks.append(f_mean)
                lower_chunks.append(f_mean - 2 * obs_std)
                upper_chunks.append(f_mean + 2 * obs_std)
                if return_latent_var:
                    f_var_chunks.append(f_var)

        mean = torch.cat(mean_chunks, dim=0)
        lower = torch.cat(lower_chunks, dim=0)
        upper = torch.cat(upper_chunks, dim=0)
        stddev = (upper - lower) / 4.0

    logger.info(
        "RFF multitask evaluation completed%s.",
        " (NIGP observation noise)" if use_nigp else "",
    )
    if return_latent_var:
        f_var_out = torch.cat(f_var_chunks, dim=0)
        return mean, lower, upper, stddev, f_var_out
    return mean, lower, upper, stddev

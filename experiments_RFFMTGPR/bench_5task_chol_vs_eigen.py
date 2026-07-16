"""Chol vs eigen Woodbury on the 20-D / 5-task synthetic multitask problem.

Stage A: synthetic Phi microbench at T=5 (n matching train size, D in {200,400,800}).
Stage B: real RFFMTGPR on generated data — one MLL forward (+ optional short Adam).

Usage (gpplus2026 / CUDA)::

  set PYTHONPATH=%CD%
  conda run -n gpplus2026 python experiments_RFFMTGPR/bench_5task_chol_vs_eigen.py --device cuda
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_MT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_MT) not in sys.path:
    sys.path.insert(0, str(_MT))

from gpplus.models.rff_mtgpr import RFFMTGPR
from gpplus.training.rff_mt_mll import RFFMTWoodburyMarginalLogLikelihood
from gpplus.utils import StandardScaler, UniformScaler, set_seed
from gpplus.utils.rff_utils import (
    flatten_multitask_targets,
    woodbury_factor_mt,
    woodbury_marginal_log_likelihood_mt,
)
from synthetic_5task_data import (
    INPUT_DIM,
    NUM_TASKS,
    TASK_NAMES,
    generate_5task_20d_data,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, repeats: int, warmup: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
        _sync(device)
    times: list[float] = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def _make_phi_inputs(n: int, d_freq: int, num_tasks: int, device: torch.device, dtype: torch.dtype):
    m = 2 * d_freq
    torch.manual_seed(0)
    phi = torch.randn(n, m, device=device, dtype=dtype) / (m**0.5)
    a = torch.randn(num_tasks, num_tasks, device=device, dtype=dtype)
    r_b = a @ a.T + 0.1 * torch.eye(num_tasks, device=device, dtype=dtype)
    task_noises = torch.full((num_tasks,), 0.05, device=device, dtype=dtype)
    y = torch.randn(n * num_tasks, device=device, dtype=dtype)
    return phi, r_b, task_noises, y


def bench_phi_stage(
    n: int,
    d_list: list[int],
    device: torch.device,
    dtype: torch.dtype,
    repeats: int,
    warmup: int,
) -> None:
    print("\n=== Stage A: synthetic Phi microbench (T=5) ===")
    print(
        f"n={n} T={NUM_TASKS} dtype={dtype} device={device} repeats={repeats} (median)"
    )
    print(
        f"{'D':>5} {'mT':>6} {'method':>6} {'factor_s':>10} {'mll_s':>10} "
        f"{'vs_chol':>8} {'dense_M_GiB':>12} {'abs_dmll':>12}"
    )
    for d_freq in d_list:
        rows: dict[str, dict | None] = {}
        for method in ("chol", "eigen"):
            try:
                phi, r_b, task_noises, y = _make_phi_inputs(
                    n, d_freq, NUM_TASKS, device, dtype
                )
                m = phi.shape[-1]
                m_t = m * NUM_TASKS

                def run_factor(_tn=task_noises, _phi=phi, _rb=r_b, _m=method):
                    return woodbury_factor_mt(_tn, _phi, _rb, jitter=1e-6, method=_m)

                def run_mll(_tn=task_noises, _phi=phi, _rb=r_b, _y=y, _m=method):
                    return woodbury_marginal_log_likelihood_mt(
                        _tn, _phi, _rb, n, _y, jitter=1e-6, method=_m
                    )

                factor_times = _time_call(
                    run_factor, repeats=repeats, warmup=warmup, device=device
                )
                mll_times = _time_call(
                    run_mll, repeats=repeats, warmup=warmup, device=device
                )
                mll_val = float(run_mll().detach().cpu())
                rows[method] = {
                    "D": d_freq,
                    "mT": m_t,
                    "method": method,
                    "factor_s": statistics.median(factor_times),
                    "mll_s": statistics.median(mll_times),
                    "mll": mll_val,
                    "dense_GiB": (m_t * m_t * 8) / (1024**3),
                }
            except Exception as exc:  # noqa: BLE001
                print(f"{d_freq:5d} {'?':>6} {method:>6} FAILED: {exc}")
                rows[method] = None

        if rows.get("chol") and rows.get("eigen"):
            speed = rows["chol"]["mll_s"] / max(rows["eigen"]["mll_s"], 1e-12)
            delta = abs(rows["chol"]["mll"] - rows["eigen"]["mll"])
            for method in ("chol", "eigen"):
                r = rows[method]
                sp = speed if method == "eigen" else 1.0
                print(
                    f"{r['D']:5d} {r['mT']:6d} {r['method']:>6} "
                    f"{r['factor_s']:10.4f} {r['mll_s']:10.4f} "
                    f"{sp:8.2f}x {r['dense_GiB']:12.3f} "
                    f"{(delta if method == 'eigen' else 0.0):12.3e}"
                )


def _prepare_model_data(
    n_train: int,
    n_test: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
):
    set_seed(seed)
    X_tr, Y_tr, X_te, Y_te = generate_5task_20d_data(
        n_train, n_test, seed=seed, train_noise=0.01, test_noise=0.0
    )
    x_scaler = UniformScaler(scale_to_neg_one=True)
    x_scaler.fit(X_tr)
    X_tr_s = x_scaler.transform(X_tr).to(device=device, dtype=dtype)
    X_te_s = x_scaler.transform(X_te).to(device=device, dtype=dtype)
    y_scaler = StandardScaler()
    y_scaler.fit(Y_tr)
    Y_tr_s = y_scaler.transform(Y_tr).to(device=device, dtype=dtype)
    Y_te_s = y_scaler.transform(Y_te).to(device=device, dtype=dtype)
    return X_tr_s, Y_tr_s, X_te_s, Y_te_s, Y_tr, Y_te


def _build_model(X: torch.Tensor, Y: torch.Tensor, num_rff: int, seed: int) -> RFFMTGPR:
    set_seed(seed)
    return RFFMTGPR(
        X,
        Y,
        num_tasks=NUM_TASKS,
        num_rff=num_rff,
        ard=True,
        rff_sampling="sorf",
        rank_kernel=1,
        rank_likelihood=0,
    )


def bench_real_model_stage(
    n_train: int,
    n_test: int,
    num_rff: int,
    device: torch.device,
    dtype: torch.dtype,
    repeats: int,
    warmup: int,
    adam_epochs: int,
    seed: int,
) -> None:
    print("\n=== Stage B: real RFFMTGPR on 20-D / 5-task data ===")
    X_tr, Y_tr, _, _, Y_raw_tr, Y_raw_te = _prepare_model_data(
        n_train, n_test, seed, device, dtype
    )
    print(f"tasks={TASK_NAMES}")
    print(
        f"raw Y train mean={Y_raw_tr.mean(0).tolist()}  "
        f"std={Y_raw_tr.std(0).tolist()}"
    )
    print(
        f"n_train={n_train} n_test={n_test} d_in={INPUT_DIM} "
        f"num_rff={num_rff} m={2 * num_rff} mT={2 * num_rff * NUM_TASKS}"
    )

    base_model = _build_model(X_tr, Y_tr, num_rff, seed=seed).to(device)
    base_state = copy.deepcopy(base_model.state_dict())

    # One MLL forward from the same features / params for both methods
    phi = base_model.train_spatial_features().detach()
    r_b = base_model.task_psd_factor().detach()
    task_noises = base_model.task_noises().detach()
    mean = base_model.mean_module(X_tr).detach()
    y_centered = flatten_multitask_targets(Y_tr - mean).detach()
    n = X_tr.shape[0]

    print(
        f"\n{('method'):>6} {'mll_fwd_s':>10} {'mll':>14} {'abs_d_vs_chol':>14}"
    )
    mll_vals: dict[str, float] = {}
    for method in ("chol", "eigen"):

        def run(_m=method):
            return woodbury_marginal_log_likelihood_mt(
                task_noises, phi, r_b, n, y_centered, jitter=1e-6, method=_m
            )

        times = _time_call(run, repeats=repeats, warmup=warmup, device=device)
        val = float(run().detach().cpu())
        mll_vals[method] = val
        delta = abs(val - mll_vals["chol"]) if method == "eigen" else 0.0
        print(f"{method:>6} {statistics.median(times):10.4f} {val:14.6g} {delta:14.3e}")

    if adam_epochs <= 0:
        return

    print(f"\nShort Adam ({adam_epochs} epochs) wall-clock:")
    print(f"{'method':>6} {'train_s':>10} {'final_loss':>14}")
    for method in ("chol", "eigen"):
        model = _build_model(X_tr, Y_tr, num_rff, seed=seed).to(device)
        model.load_state_dict(copy.deepcopy(base_state))
        model.train()
        mll = RFFMTWoodburyMarginalLogLikelihood(model.likelihood, model, method=method)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)

        def train_loop():
            last = None
            for _ in range(adam_epochs):
                opt.zero_grad(set_to_none=True)
                output = model(X_tr)
                loss = -mll(output, Y_tr)
                loss.backward()
                opt.step()
                last = float(loss.detach().cpu())
            return last

        # Single timed run (warmup skip for short loops)
        _sync(device)
        t0 = time.perf_counter()
        final = train_loop()
        _sync(device)
        elapsed = time.perf_counter() - t0
        print(f"{method:>6} {elapsed:10.4f} {final:14.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--num-rff", type=int, default=400)
    parser.add_argument("--d-list", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--adam-epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-phi", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if not args.skip_phi:
        bench_phi_stage(
            n=args.n_train,
            d_list=list(args.d_list),
            device=device,
            dtype=dtype,
            repeats=args.repeats,
            warmup=args.warmup,
        )
    if not args.skip_model:
        bench_real_model_stage(
            n_train=args.n_train,
            n_test=args.n_test,
            num_rff=args.num_rff,
            device=device,
            dtype=dtype,
            repeats=args.repeats,
            warmup=args.warmup,
            adam_epochs=args.adam_epochs,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

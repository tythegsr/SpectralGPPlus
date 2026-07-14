"""Summarize Phase 0–3 diagnostics into a writeup (primary metric: RRMSE)."""

from __future__ import annotations

from pathlib import Path

from common import load_json, results_dir, save_json


def _fmt(x, fmt=".4g"):
    if x is None:
        return "n/a"
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)


def summarize() -> str:
    root = results_dir()
    lines: list[str] = []
    lines.append("# Why SORF Beats RFF and ORF — Investigation Writeup\n")
    lines.append(
        "Shared stack: `RFFKernel` -> `RFFGPR` + Woodbury. "
        "Only `rff_sampling ∈ {rff, orf, sorf}` changes how frequency matrix `W` is drawn.\n"
    )
    lines.append("**Primary comparison metric: RRMSE** (RMSE / std(y_true)).\n")

    # Phase 0
    p0 = root / "phase0_matched_summary.json"
    lines.append("## Phase 0 — Matched TOA bake-off\n")
    if p0.is_file():
        d0 = load_json(p0)
        proto = d0.get("protocol", {})
        if proto:
            lines.append(
                f"Protocol: n_train={proto.get('n_train')}, D={proto.get('num_rff')}, "
                f"seed={proto.get('seed')}, num_inits={proto.get('num_inits')}, "
                f"train_subset={proto.get('train_subset')}, epochs={proto.get('num_epochs')}, "
                f"logit_cos={proto.get('logit_cos')}.\n"
            )
        if d0.get("source"):
            lines.append(f"Source: {d0['source']}\n")
        lines.append(
            "| Method | cos RRMSE | grain RRMSE | cos noise | grain noise | cos loss |"
        )
        lines.append("|--------|-----------|-------------|-----------|-------------|----------|")
        for r in d0["rows"]:
            cos_r = r.get("y_cos_RRMSE", r.get("y_cos_RMSE"))
            grain_r = r.get("y_grain_RRMSE", r.get("y_grain_RMSE"))
            lines.append(
                f"| {r['rff_sampling']} | {_fmt(cos_r)} | {_fmt(grain_r)} | "
                f"{_fmt(r.get('y_cos_noise'))} | {_fmt(r.get('y_grain_noise'))} | "
                f"{_fmt(r.get('y_cos_best_train_loss'))} |"
            )
        by_cos = sorted(
            d0["rows"],
            key=lambda r: (r.get("y_cos_RRMSE") if r.get("y_cos_RRMSE") is not None else 1e9),
        )
        if by_cos and by_cos[0].get("y_cos_RRMSE") is not None:
            gap_survives = by_cos[0]["rff_sampling"] == "sorf" and (
                (by_cos[1].get("y_cos_RRMSE") or 0)
                / max(by_cos[0].get("y_cos_RRMSE") or 1e-12, 1e-12)
                > 5
            )
            lines.append(
                f"\nGap survives matched protocol (by cos RRMSE): "
                f"**{'yes' if gap_survives else 'no / unclear'}** "
                f"(best={by_cos[0]['rff_sampling']}).\n"
            )
    else:
        lines.append("_Phase 0 results missing._\n")

    # Phase 1
    p1 = root / "phase1_kernel_quality.json"
    lines.append("## Phase 1 — Kernel approximation quality (fixed lengthscale)\n")
    if p1.is_file():
        d1 = load_json(p1)
        lines.append(
            "At fixed raw lengthscale=0 (scale=1), kernel MSE of `z(x)ᵀz(x')` vs exact RBF:\n"
        )
        lines.append("| D | Mode | kernel MSE | bias | within-block |cos| | cond(WᵀW) |")
        lines.append("|---|------|------------|------|----------------------|-----------|")
        for D in (800, 1600):
            for mode in ("rff", "orf", "sorf"):
                row = next(
                    (r for r in d1["rows"] if r["num_rff"] == D and r["rff_sampling"] == mode),
                    None,
                )
                if not row:
                    continue
                ws = row["w_stats"]
                lines.append(
                    f"| {D} | {mode} | {_fmt(row['kernel_mse'], '.3e')} | "
                    f"{_fmt(row['kernel_bias'], '+.3e')} | "
                    f"{_fmt(ws.get('within_block_abs_cos_mean'))} | "
                    f"{_fmt(ws.get('cond_WtW'), '.2e')} |"
                )
        at800 = {r["rff_sampling"]: r for r in d1["rows"] if r["num_rff"] == 800}
        if len(at800) == 3:
            mse_order = sorted(at800, key=lambda m: at800[m]["kernel_mse"])
            lines.append(
                f"\nKernel-MSE ranking at D=800: {' < '.join(mse_order)}. "
                "If ORF≈SORF are not clearly better than RFF here but GP **RRMSE** "
                "still has SORF ≫ ORF, the TOA win is **not** explained by Yu-style "
                "kernel MC variance alone.\n"
            )
    else:
        lines.append("_Phase 1 results missing._\n")

    # Phase 2
    p2 = root / "phase2_opt_and_wswap.json"
    lines.append("## Phase 2 — Optimization / conditioning + W-swap\n")
    if p2.is_file():
        d2 = load_json(p2)
        lines.append("### Learned noise and Woodbury conditioning\n")
        lines.append("| Method | Task | noise | M cond | train loss |")
        lines.append("|--------|------|-------|--------|------------|")
        for c in d2["conditioning"]:
            lines.append(
                f"| {c['rff_sampling']} | {c['task_name']} | "
                f"{_fmt(c['hypers']['noise'])} | {_fmt(c['woodbury']['M_cond'], '.2e')} | "
                f"{_fmt(c['best_train_loss'])} |"
            )
        lines.append("\n### W-swap ablation (freeze SORF hypers, redraw W only)\n")
        for label, block in (
            ("y_cos", d2.get("w_swap_y_cos")),
            ("y_grain", d2.get("w_swap_y_grain")),
        ):
            if not block:
                continue
            lines.append(f"**{label}**\n")
            lines.append("| Weights | RRMSE | M cond |")
            lines.append("|---------|-------|--------|")
            for r in block["rows"]:
                lines.append(
                    f"| {r['weights']} | {_fmt(r['test'].get('rrmse', r['test'].get('rmse')))} | "
                    f"{_fmt(r['woodbury']['M_cond'], '.2e')} |"
                )
            redraws = [r for r in block["rows"] if r["weights"].startswith("redraw_")]
            if redraws:
                best = min(redraws, key=lambda r: r["test"].get("rrmse", r["test"]["rmse"]))
                worst = max(redraws, key=lambda r: r["test"].get("rrmse", r["test"]["rmse"]))
                b = best["test"].get("rrmse", best["test"]["rmse"])
                w = worst["test"].get("rrmse", worst["test"]["rmse"])
                ratio = (w + 1e-12) / (b + 1e-12)
                if ratio < 2:
                    lines.append(
                        "\nAt fixed hypers, redrawing W under rff/orf/sorf gives **similar** RRMSE "
                        "→ gap is largely an **optimization** interaction (SORF reaches better hypers).\n"
                    )
                else:
                    lines.append(
                        f"\nAt fixed hypers, best redraw={best['rff_sampling']} vs "
                        f"worst={worst['rff_sampling']} (RRMSE ratio≈{ratio:.1f}) "
                        "→ **approximation/coverage** of W still matters even without re-training.\n"
                    )
    else:
        lines.append("_Phase 2 results missing._\n")

    # Phase 3
    p3 = root / "phase3_sweeps.json"
    lines.append("## Phase 3 — D-sweep and low-d Ackley\n")
    if p3.is_file():
        d3 = load_json(p3)
        if d3.get("ackley"):
            for name, block in d3["ackley"].items():
                lines.append(f"### {name}\n")
                lines.append("| Mode | RRMSE | RMSE | loss | noise |")
                lines.append("|------|-------|------|------|-------|")
                for r in block["rows"]:
                    lines.append(
                        f"| {r['rff_sampling']} | {_fmt(r.get('RRMSE'))} | {_fmt(r.get('RMSE'))} | "
                        f"{_fmt(r['best_train_loss'])} | {_fmt(r['noise'])} |"
                    )
                order = sorted(
                    block["rows"],
                    key=lambda r: r.get("RRMSE") if r.get("RRMSE") is not None else r["RMSE"],
                )
                lines.append(f"\nRanking (RRMSE): {' < '.join(r['rff_sampling'] for r in order)}. ")
                b = order[0].get("RRMSE") or order[0]["RMSE"]
                w = order[1].get("RRMSE") or order[1]["RMSE"]
                if order[0]["rff_sampling"] == "sorf" and w / max(b, 1e-12) > 5:
                    lines.append("SORF uniquely dominates even on Ackley → more than TOA-specific.\n")
                else:
                    lines.append(
                        "On Ackley the SORF-unique blowout is weaker/absent → "
                        "TOA high-d structure amplifies the effect.\n"
                    )
        if d3.get("toa_d_sweep"):
            lines.append("### TOA D-sweep (short train)\n")
            lines.append("| D | Mode | cos RRMSE | grain RRMSE | cos noise |")
            lines.append("|---|------|-----------|-------------|-----------|")
            for r in d3["toa_d_sweep"]["rows"]:
                lines.append(
                    f"| {r['num_rff']} | {r['rff_sampling']} | "
                    f"{_fmt(r.get('y_cos_RRMSE', r.get('y_cos_RMSE')))} | "
                    f"{_fmt(r.get('y_grain_RRMSE', r.get('y_grain_RMSE')))} | "
                    f"{_fmt(r['y_cos_noise'])} |"
                )
    else:
        lines.append("_Phase 3 results missing._\n")

    lines.append("## Conclusion\n")
    conclusion = _infer_conclusion(root)
    lines.append(conclusion + "\n")
    lines.append("Artifacts under `experiments_SORF/results/why_sorf_wins/`.\n")

    text = "\n".join(lines)
    out = root / "WRITEUP.md"
    out.write_text(text, encoding="utf-8")
    save_json(root / "writeup_meta.json", {"path": str(out), "conclusion": conclusion})
    print(text)
    print(f"\nWrote {out}")
    return text


def _infer_conclusion(root: Path) -> str:
    bits = []
    p0 = root / "phase0_matched_summary.json"
    p1 = root / "phase1_kernel_quality.json"
    p2 = root / "phase2_opt_and_wswap.json"
    p3 = root / "phase3_sweeps.json"

    if p0.is_file():
        rows = load_json(p0)["rows"]
        by = {r["rff_sampling"]: r for r in rows}
        if "sorf" in by and "orf" in by and by["sorf"].get("y_cos_RRMSE") is not None:
            gap = (by["orf"]["y_cos_RRMSE"] or 1) / max(by["sorf"]["y_cos_RRMSE"] or 1e-12, 1e-12)
            bits.append(
                f"Matched TOA protocol preserves a large SORF advantage "
                f"(ORF/SORF cos-RRMSE ratio ≈ {gap:.0f}×)."
            )

    kernel_explains = None
    if p1.is_file():
        at800 = {r["rff_sampling"]: r for r in load_json(p1)["rows"] if r["num_rff"] == 800}
        if len(at800) == 3:
            ratio = at800["orf"]["kernel_mse"] / max(at800["sorf"]["kernel_mse"], 1e-30)
            kernel_explains = ratio > 5
            bits.append(
                f"Fixed-hypers kernel MSE: ORF/SORF ratio ≈ {ratio:.2f} "
                f"({'supports approximation story' if kernel_explains else 'too small to explain GP RRMSE gap'})."
            )

    opt_story = None
    if p2.is_file():
        d2 = load_json(p2)
        block = d2.get("w_swap_y_cos") or {}
        redraws = [r for r in block.get("rows", []) if str(r.get("weights", "")).startswith("redraw_")]
        if redraws:
            best = min(redraws, key=lambda r: r["test"].get("rrmse", r["test"]["rmse"]))
            worst = max(redraws, key=lambda r: r["test"].get("rrmse", r["test"]["rmse"]))
            b = best["test"].get("rrmse", best["test"]["rmse"])
            w = worst["test"].get("rrmse", worst["test"]["rmse"])
            ratio = (w + 1e-12) / (b + 1e-12)
            opt_story = ratio < 2
            bits.append(
                f"W-swap at frozen SORF hypers: redraw RRMSE ratio ≈ {ratio:.2f} "
                f"({'optimization-dominated' if opt_story else 'W geometry still decisive'})."
            )
        noises = {
            (c["rff_sampling"], c["task_name"]): c["hypers"]["noise"]
            for c in d2.get("conditioning", [])
        }
        if ("sorf", "y_cos") in noises and ("orf", "y_cos") in noises:
            bits.append(
                f"Learned cos noise: SORF={noises[('sorf','y_cos')]:.3g}, "
                f"ORF={noises[('orf','y_cos')]:.3g} (underfit signal for ORF/RFF)."
            )

    if p3.is_file():
        ack = (load_json(p3).get("ackley") or {}).get("ackley_10d")
        if ack:
            order = sorted(
                ack["rows"],
                key=lambda r: r.get("RRMSE") if r.get("RRMSE") is not None else r["RMSE"],
            )
            b = order[0].get("RRMSE") or order[0]["RMSE"]
            w = order[1].get("RRMSE") or order[1]["RMSE"]
            toa_specific = not (order[0]["rff_sampling"] == "sorf" and w / max(b, 1e-12) > 5)
            bits.append(
                "Ackley: "
                + (
                    "SORF does not uniquely dominate → TOA amplifies the effect."
                    if toa_specific
                    else "SORF also uniquely dominates → more general than TOA."
                )
            )

    if opt_story is True and kernel_explains is False:
        hyp = (
            "**Primary hypothesis supported: H1 (optimization landscape).** "
            "SORF's structured W lets Adam reach near-zero noise / low MLL; "
            "ORF/RFF underfit with large learned noise. Kernel MC variance (Yu) is secondary."
        )
    elif kernel_explains is True and opt_story is False:
        hyp = (
            "**Primary hypothesis supported: H2 (approximation/coverage).** "
            "Even at fixed hypers, SORF frequencies explain the targets much better (RRMSE)."
        )
    elif opt_story is False and kernel_explains is False:
        hyp = (
            "**Mixed: optimization and W geometry both matter**, beyond textbook ORF≈SORF theory. "
            "Check H3 (implementation / column-norm differences) in Phase 1 W stats."
        )
    else:
        hyp = (
            "**Evidence incomplete or mixed.** See phase JSON artifacts for details; "
            "the empirical TOA RRMSE gap far exceeds Yu et al.'s modest variance-reduction claim."
        )
    return " ".join(bits) + "\n\n" + hyp


if __name__ == "__main__":
    summarize()

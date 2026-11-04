"""Deep-dive MidlineSparse slides: derivation, parameters, UQ validity.

Native PowerPoint equations via LaTeX -> MathML -> Office MML2OMML.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import latex2mathml.converter
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent / "MidlineSparse_deep_dive_slides.pptx"
MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

etree.register_namespace("a", A_NS)
etree.register_namespace("a14", A14_NS)
etree.register_namespace("m", M_NS)
_XSLT = etree.XSLT(etree.parse(str(MML2OMML)))

NAVY = RGBColor(0x1A, 0x36, 0x5D)
SLATE = RGBColor(0x2D, 0x37, 0x48)
TEAL = RGBColor(0x2C, 0x7A, 0x7B)
ORANGE = RGBColor(0xC0, 0x56, 0x21)
GREEN = RGBColor(0x27, 0x6B, 0x49)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF7, 0xFA, 0xFC)
GRAY = RGBColor(0x71, 0x80, 0x96)
LINE = RGBColor(0xA0, 0xAE, 0xC0)
RED = RGBColor(0x9B, 0x2C, 0x2C)


def _set_run(run, text, *, size=16, bold=False, color=SLATE, font="Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def _title(slide, text, *, top=0.28, size=24):
    box = slide.shapes.add_textbox(Inches(0.4), Inches(top), Inches(9.2), Inches(0.5))
    p = box.text_frame.paragraphs[0]
    r = p.add_run()
    _set_run(r, text, size=size, bold=True, color=NAVY)


def _text(slide, left, top, width, height, text, *, size=14, bold=False, color=SLATE, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    _set_run(r, text, size=size, bold=bold, color=color)
    return box


def _bullets(slide, left, top, width, height, lines, *, size=13):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(3)
        r = p.add_run()
        _set_run(r, "•  " + line, size=size, color=SLATE)
    return box


def _defs(slide, left, top, width, height, pairs, *, size=12):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (sym, defn) in enumerate(pairs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(2)
        r = p.add_run()
        _set_run(r, f"{sym}  —  {defn}", size=size, color=SLATE)
    return box


def _panel(slide, left, top, width, height, fill=LIGHT):
    sh = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = LINE
    sh.line.width = Pt(1.0)
    return sh


def latex_to_a14m(latex: str, *, font_sz: int = 1600) -> etree._Element:
    mathml = latex2mathml.converter.convert(latex)
    tree = etree.fromstring(mathml.encode("utf-8"))
    omml_root = _XSLT(tree).getroot()
    if omml_root.tag == f"{{{M_NS}}}oMathPara":
        omath = omml_root.find(f"{{{M_NS}}}oMath") or omml_root
    else:
        omath = omml_root
    omath = deepcopy(omath)
    for r in omath.xpath(".//m:r", namespaces={"m": M_NS}):
        for child in list(r):
            tag = child.tag
            if tag.startswith(f"{{{W_NS}}}") or (
                tag.endswith("}rPr") and not tag.startswith(f"{{{A_NS}}}")
            ):
                r.remove(child)
        if r.find(f"{{{A_NS}}}rPr") is None:
            rPr = etree.Element(f"{{{A_NS}}}rPr")
            rPr.set("sz", str(font_sz))
            rPr.set("i", "1")
            latin = etree.SubElement(rPr, f"{{{A_NS}}}latin")
            latin.set("typeface", "Cambria Math")
            r.insert(0, rPr)
        # Guard: drop runs whose text contains replacement chars (bad MathML glyphs)
        t_el = r.find(f"{{{M_NS}}}t")
        if t_el is not None and t_el.text and "\ufffd" in t_el.text:
            raise ValueError(f"Corrupt OMML text from LaTeX: {latex!r} -> {t_el.text!r}")
    wrapper = etree.Element(f"{{{A14_NS}}}m", nsmap={"a14": A14_NS, "m": M_NS, "a": A_NS})
    wrapper.append(omath)
    return wrapper


def add_eq(slide, latex: str, *, left, top, width, height, font_sz=1500, panel=True):
    if panel:
        _panel(slide, left, top, width, height)
    box = slide.shapes.add_textbox(
        Inches(left + 0.08), Inches(top + 0.1), Inches(width - 0.16), Inches(height - 0.18)
    )
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    for child in list(p._element):
        if child.tag.endswith("}r"):
            p._element.remove(child)
    p._element.append(latex_to_a14m(latex, font_sz=font_sz))
    return box


def _normalize(path: Path) -> None:
    try:
        import win32com.client
    except ImportError:
        return
    app = win32com.client.Dispatch("PowerPoint.Application")
    try:
        app.Visible = 0
    except Exception:
        app.Visible = 1
    pres = app.Presentations.Open(str(path.resolve()), WithWindow=False)
    pres.SaveAs(str(path.resolve()), 24)
    pres.Close()
    try:
        app.Quit()
    except Exception:
        pass


def build() -> Path:
    eqs = {
        "prior": r"f_0\sim\mathcal{GP}(m(\cdot),k(\cdot,\cdot)),\quad y=f_0(x)+\varepsilon",
        "lik": r"y\mid f_0\sim\mathrm{N}(f_0,\sigma_{\varepsilon}^2 I)",
        "gram": r"K_{ij}=k(x_i,x_j),\quad K_{\varepsilon}=K+\sigma_{\varepsilon}^2 I",
        "mu_post_full": (
            r"\mu_0(x_*)=m(x_*)+k(x_*,X)K_{\varepsilon}^{-1}(y-m(X))"
        ),
        "var_f_full": (
            r"v_f(x_*)=k(x_*,x_*)-k(x_*,X)K_{\varepsilon}^{-1}k(X,x_*)"
        ),
        "var_y_full": r"\sigma_0^2(x_*)=v_f(x_*)+\sigma_{\varepsilon}^2",
        "post_short": (
            r"\mu_0(x)=\mathrm{E}[f_0(x)\mid D],\quad"
            r"\sigma_0^2(x)=\mathrm{Var}[y(x)\mid D]"
        ),
        "resid": r"r_i=y_i-\mu_0(x_i),\quad\hat{s}=\mathrm{Std}(\{r_i\})",
        "cap": r"\sigma_0^{\mathrm{cap}}(x)=\min(\sigma_0(x),\,\max(c\hat{s},0.05))",
        "mu_k": r"\mu_k(x)=\mu_0(x)+\alpha_k\,\delta_k(x)",
        "var_k": (
            r"\mathrm{Var}_k(x)=\sigma_0(x)^2(1-\alpha_k\gamma_k)+\varepsilon"
        ),
        "py": (
            r"p(y\mid x)=\sum_{k=1}^{K_{\max}}\pi_k(x)\,"
            r"\mathrm{N}\big(y;\mu_k(x),\mathrm{Var}_k(x)\big)"
        ),
        "nll": (
            r"\mathrm{NLL}=-\frac{1}{n}\sum_i\log\sum_k"
            r"\pi_k(x_i)\,\mathrm{N}(y_i;\mu_k,\mathrm{Var}_k)"
        ),
        "alpha_det": (
            r"\alpha_k=\mathrm{clip}_{[0,1]}"
            r"\big(\sigma(\log\alpha_k)(\zeta_{\mathrm{HC}}-\gamma_{\mathrm{HC}})"
            r"+\gamma_{\mathrm{HC}}\big)"
        ),
        "hc_def": (
            r"\gamma_{\mathrm{HC}}=-0.1,\ "
            r"\zeta_{\mathrm{HC}}=1.1,\ "
            r"\beta_{\mathrm{HC}}=2/3"
        ),
        "pi_const": r"\pi=\mathrm{softmax}(z/T),\quad z\in\mathrm{R}^{K_{\max}}",
        "pin": (
            r"w_{ik}\propto\pi_k(x_i)\alpha_k,\quad"
            r"\delta_{ik}=\delta_{ik}^{\mathrm{raw}}"
            r"-\sum_j w_{ij}\delta_{ij}^{\mathrm{raw}}"
        ),
        "gamma": r"\gamma_k=\sigma(\mathrm{raw\_gamma}_k)\in(0,1)",
        "loss": (
            r"\mathcal{L}=\mathrm{NLL}+\sum_{\ell}\lambda_{\ell}\,\mathcal{L}_{\ell}"
        ),
        "mix_mom": (
            r"\mathrm{E}[y\mid x]=\sum_k\pi_k\mu_k,\quad"
            r"\mathrm{Var}[y\mid x]=\sum_k\pi_k(\mathrm{Var}_k+\mu_k^2)"
            r"-(\mathrm{E}[y\mid x])^2"
        ),
        "tilde": (
            r"\tilde{\alpha}_k=\sigma\big("
            r"(\alpha_k\bar{\pi}_k-\varepsilon)/\tau\big)"
        ),
        "rbf": (
            r"k(x,x')=\sigma_f^2\exp\!\big(-\|x-x'\|^2/(2\ell^2)\big)"
        ),
    }
    for L in eqs.values():
        latex_to_a14m(L)

    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1 Title
    s = prs.slides.add_slide(blank)
    _title(s, "MidlineSparseMoE-GP — Deep Derivation", top=1.8, size=30)
    _text(
        s, 0.5, 2.7, 9.0, 2.0,
        "Forward → likelihood → Stage-2 optimization → prediction\n"
        "What is prior vs posterior · what Adam learns · what is MLP · UQ validity",
        size=17, align=PP_ALIGN.CENTER,
    )
    _text(s, 0.5, 6.5, 9.0, 0.4, "Internal technical deck  ·  matches midline_sparse_moe.py", size=12, color=GRAY, align=PP_ALIGN.CENTER)

    # 2 Learning goals
    s = prs.slides.add_slide(blank)
    _title(s, "What this deck answers")
    _bullets(
        s, 0.5, 1.2, 9.0, 5.8,
        [
            "Derive μ₀ and σ₀ from the GP posterior (kernel + noise) before shorthand.",
            "How the kernel enters Stage 1 and what is frozen into Stage 2.",
            "Forward story: p(y|x) from μ₀, σ₀, α, δ, π, γ (no σ₀ needed).",
            "Champion: which params are MLPs vs Adam scalars vs fixed λ’s.",
            "UQ validity: conditional density + frozen GP envelope (not joint Bayes).",
        ],
        size=15,
    )

    # 3 Problem
    s = prs.slides.add_slide(blank)
    _title(s, "Problem this model targets")
    _bullets(
        s, 0.5, 1.2, 9.0, 5.5,
        [
            "Co-located multimodality: several modes of p(y|x) at the same x (mode ⊥ x).",
            "Standard MoE-GP gates that partition x are misspecified for this.",
            "A single GP posterior is unimodal Gaussian — cannot represent multiple y-modes.",
            "Scientific metric: number of modes of p(y|x), not number of experts.",
            "Design: one shared GP midline + sparse residual mixture on top.",
        ],
        size=15,
    )

    # 4 Two-stage
    s = prs.slides.add_slide(blank)
    _title(s, "Two-stage procedure (high level)")
    _panel(s, 0.5, 1.2, 4.3, 2.2, TEAL)
    _text(s, 0.7, 1.4, 3.9, 1.8, "Stage 1\nFit exact GPR on all data\nStore μ₀(x), σ₀(x)\nFREEZE the GP", size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _panel(s, 5.2, 1.2, 4.3, 2.2, ORANGE)
    _text(
        s, 5.4, 1.4, 3.9, 1.8,
        "Stage 2\nAdam on mixture params\nMLP gate π(x) + MLP δ(x)\n+ scalars α, γ\nGP never updates",
        size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
    )
    _bullets(
        s, 0.5, 3.7, 9.0, 3.2,
        [
            "Stage 1 is Bayesian GP regression (multi-init L-BFGS on marginal likelihood).",
            "Stage 2 is plug-in Adam optimization of a Gaussian mixture given frozen (μ₀, σ₀).",
            "Champion: both gate and offsets are MLPs; α and γ are Adam scalars (not nets).",
            "Residuals only seed δ (and optionally cap σ₀) — not Stage-2 targets.",
        ],
        size=14,
    )

    # 5 Prior
    s = prs.slides.add_slide(blank)
    _title(s, "Stage 1 prior — before any data")
    add_eq(s, eqs["prior"], left=0.4, top=1.1, width=9.2, height=0.95, font_sz=1500)
    add_eq(s, eqs["lik"], left=0.4, top=2.2, width=9.2, height=0.85, font_sz=1600)
    _bullets(
        s, 0.5, 3.3, 9.0, 3.6,
        [
            "f₀ is a random function. The prior mean m(·) is ConstantMean (a scalar bias).",
            "The prior covariance is the kernel k — this is what makes nearby x share similar f₀.",
            "ε ~ N(0, σ_ε²) is observation noise (learned with the GP likelihood).",
            "μ₀ is NOT m(·). μ₀ appears only after conditioning on D = {(xᵢ,yᵢ)}.",
        ],
        size=14,
    )

    # 6 Kernel
    s = prs.slides.add_slide(blank)
    _title(s, "Where the kernel fits in")
    add_eq(s, eqs["rbf"], left=0.4, top=1.05, width=9.2, height=0.9, font_sz=1500)
    add_eq(s, eqs["gram"], left=0.4, top=2.1, width=9.2, height=0.9, font_sz=1500)
    _bullets(
        s, 0.5, 3.2, 9.0, 3.8,
        [
            "Code: LogScaleKernel(GaussianKernel) — RBF with learnable ℓ (lengthscale) and σ_f².",
            "K is the n×n Gram matrix on training inputs X; it is the prior covariance of f₀(X).",
            "k(x_*, X) is the cross-covariance row that couples a new x_* to the training set.",
            "Stage-1 training fits kernel hypers + σ_ε by maximizing the GP marginal likelihood.",
            "After Stage 1, k and σ_ε are frozen. Stage 2 never re-fits the kernel.",
            "The mixture experts do not have their own kernels — they ride the shared midline GP.",
        ],
        size=13,
    )

    # 7 Posterior mean derivation
    s = prs.slides.add_slide(blank)
    _title(s, "Derivation — posterior mean μ₀ (exact GP)")
    _text(s, 0.5, 1.0, 9.0, 0.35, "Condition the GP prior on D; evaluate at a query x_*", size=13, bold=True, color=NAVY)
    add_eq(s, eqs["mu_post_full"], left=0.4, top=1.4, width=9.2, height=1.05, font_sz=1400)
    _bullets(
        s, 0.5, 2.65, 9.0, 4.3,
        [
            "This is the classical GP posterior mean of the latent f₀(x_*) | D.",
            "With additive Gaussian noise, E[y_*|D] equals the same μ₀(x_*).",
            "Shorthand used everywhere after this slide:",
        ],
        size=14,
    )
    add_eq(s, eqs["post_short"], left=0.4, top=4.7, width=9.2, height=0.9, font_sz=1500)
    _text(
        s, 0.5, 5.8, 9.0, 1.2,
        "So writing μ₀(x) means “the GP posterior mean function”, not the prior mean m(x).",
        size=14, color=SLATE,
    )

    # 8 Posterior variance derivation
    s = prs.slides.add_slide(blank)
    _title(s, "Derivation — posterior / predictive variance σ₀²")
    _text(s, 0.5, 1.0, 9.0, 0.35, "Latent posterior variance of f₀(x_*) | D", size=13, bold=True, color=NAVY)
    add_eq(s, eqs["var_f_full"], left=0.4, top=1.4, width=9.2, height=0.95, font_sz=1400)
    _text(s, 0.5, 2.5, 9.0, 0.35, "Observation predictive variance (what Stage 2 uses)", size=13, bold=True, color=NAVY)
    add_eq(s, eqs["var_y_full"], left=0.4, top=2.9, width=9.2, height=0.85, font_sz=1600)
    _bullets(
        s, 0.5, 4.0, 9.0, 3.0,
        [
            "v_f shrinks where the kernel says training data are informative.",
            "σ₀² = v_f + σ_ε² is Var[y_*|D] — includes aleatoric noise (code: likelihood(model(x))).",
            "We write σ₀(x) = √σ₀²(x). Expert variances use σ₀(x)² as the ceiling to carve.",
            "Notation: just σ₀ — no σ₀. Capping (p12) is an optional practical replace of σ₀.",
        ],
        size=13,
    )

    # 9 Notation: no mix needed
    s = prs.slides.add_slide(blank)
    _title(s, "Notation: use σ₀ — “mix” is not necessary")
    _bullets(
        s, 0.5, 1.2, 9.0, 5.8,
        [
            "In the model equations, Stage 2 only needs the GP predictive std: σ₀(x).",
            "Var_k(x) = σ₀(x)² (1 − α_k γ_k) + ε.",
            "Earlier docs wrote σ₀ for “whichever ceiling Stage 2 uses”. That was bookkeeping only.",
            "Optional p12: replace σ₀(x) by min(σ₀(x), max(c ŝ, 0.05)) before Stage 2 — still call it σ₀ in slides.",
            "Theory / derivation: always σ₀ = GP posterior predictive std from the kernel formulas above.",
        ],
        size=15,
    )

    # 10 Stage 1 summary
    s = prs.slides.add_slide(blank)
    _title(s, "Stage 1 summary — freeze the GP posterior")
    _defs(
        s, 0.5, 1.15, 9.0, 5.8,
        [
            ("Fit", "maximize log p(y|X,θ) over kernel hypers θ and noise σ_ε (L-BFGS)"),
            ("Store", "μ₀(xᵢ), σ₀(xᵢ) on train via evaluate_gp_model; core GP object kept"),
            ("Freeze", "kernel, noise, μ₀, σ₀ — Stage 2 never updates them"),
            ("Why", "stable midline; Stage 2 only learns residual multimodality"),
            ("UQ role", "only full GP Bayesian envelope in the pipeline"),
        ],
        size=14,
    )

    # 11 Residuals
    s = prs.slides.add_slide(blank)
    _title(s, "Residuals — only for setup, not the Stage-2 loss")
    add_eq(s, eqs["resid"], left=0.4, top=1.1, width=9.2, height=0.9, font_sz=1600)
    add_eq(s, eqs["cap"], left=0.4, top=2.15, width=9.2, height=0.95, font_sz=1400)
    _bullets(
        s, 0.5, 3.35, 9.0, 3.6,
        [
            "Use 1: seed δ_k from residual quantiles (0.25/0.75/…), then center.",
            "Use 2 (optional p12): temporarily cap σ₀ using ŝ — still denote the result σ₀.",
            "Not used as regression targets: Stage 2 fits mixture NLL of y, not of r.",
        ],
        size=14,
    )

    # Parameter inventory
    s = prs.slides.add_slide(blank)
    _title(s, "Parameter inventory — what is learned how")
    _text(
        s, 0.45, 0.95, 9.2, 0.4,
        "Important: Stage-2 MLPs and scalars are both optimized by Adam. "
        "“MLP” = neural-net weights; “scalar” = free vectors of size K.",
        size=12, bold=True, color=NAVY,
    )
    _defs(
        s, 0.45, 1.4, 9.2, 5.7,
        [
            ("Stage 1 (L-BFGS, then FROZEN)", "μ₀(x), σ₀(x), GP kernel k, noise σ_ε"),
            ("Adam · MLP (champion)", "GatingNetwork: π(x)=softmax(MLP(x)/T)"),
            ("Adam · MLP (champion)", "OffsetNetwork: δ(x)=MLP(x) ∈ R^K (shared trunk)"),
            ("Adam · scalars (K each)", "log α_k (Hard-Concrete); raw_gamma_k → γ_k=σ(·)"),
            ("Optional · not champion", "ConstantGate logits z; constant δ∈R^K; raw_beta if free scale"),
            ("Muted (champion)", "β_k ≡ 1  (free_expert_scale=False)"),
            ("Fixed λ (lean champion)", "λ_k, alive_w, λ_δ, λ_α  — not learned"),
            ("Off by default now", "λ_sep, λ_rec, λ_ov, λ_γ, λ_orphan = 0"),
            ("Controller (not Adam)", "merge_pressure EMA from holdout probe → merge_pull"),
            ("Fixed HC constants", "γ_HC=-0.1, ζ_HC=1.1, β_HC=2/3 (Louizos)"),
        ],
        size=12,
    )

    # Champion MLP vs Adam scalars callout
    s = prs.slides.add_slide(blank)
    _title(s, "Champion fit — MLP nets vs Adam scalars")
    _panel(s, 0.4, 1.1, 4.5, 3.6, TEAL)
    _text(s, 0.55, 1.25, 4.2, 0.4, "Neural nets (Adam on weights)", size=14, bold=True, color=WHITE)
    _bullets(
        s, 0.55, 1.8, 4.2, 2.7,
        [
            "π(x): GatingNetwork MLP (32,32) → K logits → softmax",
            "δ(x): OffsetNetwork MLP (32,32) → K offsets",
            "Both are input-dependent functions of x",
            "Default in foundation_adapters / compare",
        ],
        size=13,
    )
    _panel(s, 5.1, 1.1, 4.5, 3.6, ORANGE)
    _text(s, 5.25, 1.25, 4.2, 0.4, "Adam scalars (not nets)", size=14, bold=True, color=WHITE)
    _bullets(
        s, 5.25, 1.8, 4.2, 2.7,
        [
            "log α_k  (K Hard-Concrete locations)",
            "raw_gamma_k  → γ_k = sigmoid(·)",
            "No per-x network for α or γ",
            "β muted (≡1); not in Adam param list",
        ],
        size=13,
    )
    _bullets(
        s, 0.45, 4.9, 9.2, 2.2,
        [
            "All of the above Stage-2 quantities share one Adam optimizer (lr≈5e-2).",
            "Stage-1 GP hypers use L-BFGSScipy on the marginal likelihood — different optimizer, then frozen.",
            "λ’s and merge_pressure are not Adam parameters (fixed / controller).",
        ],
        size=13,
    )

    # 9 Symbol meanings
    s = prs.slides.add_slide(blank)
    _title(s, "Symbol meanings — α, δ, π, γ, λ")
    _defs(
        s, 0.45, 1.05, 9.2, 6.1,
        [
            ("α_k ∈ [0,1]", "Hard-Concrete switch: how “on” expert k is (can be soft)"),
            ("δ_k(x) ∈ R", "residual offset from the midline; mean shift of expert k"),
            ("π_k(x) ∈ Δ", "mixture weight / gate; champion MLP ⇒ π(x); ConstantGate ⇒ π independent of x"),
            ("γ_k ∈ (0,1)", "fraction of σ₀² an active expert may carve away"),
            ("λ_· ≥ 0", "fixed trade-off weights in front of regularizers (not parameters of p(y|x))"),
            ("ε > 0", "tiny variance floor for numerical stability"),
            ("K_max", "overcomplete expert budget (e.g. 6); sparsity should kill unused slots"),
            ("α̃_k", "soft usage σ((α_k π̄_k − ε)/τ) — diagnostic / regularizer weight, not a generative param"),
        ],
        size=13,
    )

    # 10 Forward pass
    s = prs.slides.add_slide(blank)
    _title(s, "Forward pass (fixed x) — build p(y|x)")
    _bullets(
        s, 0.45, 1.05, 9.2, 2.2,
        [
            "1) Query frozen GP → μ₀(x), σ₀(x) (posterior mean / predictive std).",
            "2) Form α (stochastic in train / deterministic in eval), γ=σ(raw_gamma), π(x), δ^raw(x).",
            "3) Pin δ so π·α-weighted mean offset is 0; then μ_k = μ₀ + α_k δ_k, Var_k = σ₀²(1−α_k γ_k)+ε.",
            "4) Mixture density: p(y|x) = Σ_k π_k N(y; μ_k, Var_k).",
        ],
        size=13,
    )
    add_eq(s, eqs["mu_k"], left=0.4, top=3.4, width=9.2, height=0.8, font_sz=1600)
    add_eq(s, eqs["var_k"], left=0.4, top=4.35, width=9.2, height=0.85, font_sz=1400)
    add_eq(s, eqs["py"], left=0.4, top=5.35, width=9.2, height=0.9, font_sz=1400)

    # 11 Alpha
    s = prs.slides.add_slide(blank)
    _title(s, "α — Hard-Concrete switch (learned scalars log α_k)")
    add_eq(s, eqs["hc_def"], left=0.4, top=1.0, width=9.2, height=0.75, font_sz=1500)
    add_eq(s, eqs["alpha_det"], left=0.4, top=1.9, width=9.2, height=1.0, font_sz=1400)
    _bullets(
        s, 0.45, 3.1, 9.2, 3.9,
        [
            "Optimized quantity: unconstrained log α_k (Adam). α_k is a transform of it.",
            "Stretch into [−0.1, 1.1] then clip → exact 0/1 in the tails; soft values allowed in between.",
            "Train: stochastic Concrete sample (reparameterized). Eval/predict: deterministic formula.",
            "Enters the mean as a multiplier on δ_k and the variance as a multiplier on γ_k.",
            "Not a mixture weight — that is π. α can be on while π≈0 (orphan) unless regularized.",
        ],
        size=13,
    )

    # 12 Pi
    s = prs.slides.add_slide(blank)
    _title(s, "π — mixture / gate weights")
    add_eq(s, eqs["pi_const"], left=0.4, top=1.1, width=9.2, height=0.9, font_sz=1600)
    _bullets(
        s, 0.45, 2.25, 9.2, 4.6,
        [
            "Champion / foundation compare: GatingNetwork MLP — π(x)=softmax(MLP(x)/T).",
            "Optional ConstantGate: K logits z; π=softmax(z/T) same for all x (not default).",
            "π lives on the simplex: π_k ≥ 0, Σ_k π_k = 1 → mixture is a valid density in k.",
            "MLP gate can represent x-dependent mixing; still works for co-located modes.",
            "π̄_k = (1/n) Σ_i π_k(x_i) used in soft usage α̃_k (and legacy orphan if enabled).",
        ],
        size=14,
    )

    # 13 Delta
    s = prs.slides.add_slide(blank)
    _title(s, "δ — residual offsets (+ pinning)")
    add_eq(s, eqs["pin"], left=0.4, top=1.05, width=9.2, height=1.0, font_sz=1400)
    add_eq(s, eqs["mu_k"], left=0.4, top=2.2, width=9.2, height=0.85, font_sz=1600)
    _bullets(
        s, 0.45, 3.25, 9.2, 3.8,
        [
            "Champion: δ_k(x) from OffsetNetwork (shared MLP trunk + K heads) — Adam on weights.",
            "Optional constant mode: δ_k are K scalars (Adam) — not the foundation default.",
            "Pinning removes a global residual mean shift so the mixture does not re-learn the midline.",
            "With α_k≈0, expert k contributes ≈μ₀ regardless of δ_k (switch off).",
            "Seeding: residual quantiles → initial δ (bias) so arms start separated.",
            "L_δ = mean(δ²) with λ_δ=0.01 in lean champion keeps offsets from exploding.",
        ],
        size=13,
    )

    # 14 Gamma
    s = prs.slides.add_slide(blank)
    _title(s, "γ — variance carve-out (learned scalars)")
    add_eq(s, eqs["gamma"], left=0.4, top=1.1, width=9.2, height=0.85, font_sz=1700)
    add_eq(s, eqs["var_k"], left=0.4, top=2.15, width=9.2, height=0.95, font_sz=1400)
    _bullets(
        s, 0.45, 3.35, 9.2, 3.6,
        [
            "raw_gamma_k optimized by Adam; γ_k = sigmoid(raw_gamma_k) ∈ (0,1).",
            "γ→1 and α→1 ⇒ expert much sharper than σ₀ (carve). γ→0 ⇒ keep fat GP width.",
            "L_γ = Σ α̃_k (1−γ_k)² pushes active experts to carve (p12 lever).",
            "β_k scale above σ₀ was tried and muted: free_expert_scale=False ⇒ β≡1.",
        ],
        size=14,
    )

    # 15 Likelihood
    s = prs.slides.add_slide(blank)
    _title(s, "Likelihood — the only data-fit term")
    add_eq(s, eqs["py"], left=0.4, top=1.05, width=9.2, height=0.95, font_sz=1400)
    add_eq(s, eqs["nll"], left=0.4, top=2.15, width=9.2, height=1.0, font_sz=1400)
    _bullets(
        s, 0.45, 3.4, 9.2, 3.5,
        [
            "i.i.d. assumption: y_i | x_i ~ p(·|x_i) independently given frozen midline & Stage-2 params.",
            "NLL is the Monte Carlo / empirical average of −log p(y_i|x_i) (logsumexp over experts).",
            "No mixture-mean MSE: that would pull all modes toward μ₀ and destroy co-located structure.",
            "Correctness: for fixed parameters, p(y|x) ≥ 0 and ∫ p(y|x) dy = 1 (finite Gaussian mixture).",
        ],
        size=13,
    )

    # 16 Lambdas
    s = prs.slides.add_slide(blank)
    _title(s, "λ’s — fixed regularizer weights (not generative)")
    add_eq(s, eqs["loss"], left=0.4, top=1.05, width=9.2, height=0.85, font_sz=1600)
    _defs(
        s, 0.45, 2.1, 9.2, 4.9,
        [
            ("λ_k(t)", "weight on L_usage=Σ α̃_k (may anneal start→end: form modes, then prune)"),
            ("λ_sep", "weight on usage-weighted δ-curve separation"),
            ("λ_α", "weight on Hard-Concrete L0 complexity L_HC"),
            ("λ_orphan", "weight on bidirectional π–α coupling"),
            ("0.5 × L_alive", "soft floor so sparsity cannot kill all experts"),
            ("λ_ov / λ_rec", "standardized mean-gap penalties (overlap / Behboodian recovery)"),
            ("λ_δ, λ_γ", "δ ridge; γ carve-out prior on used slots"),
            ("Important", "λ’s shape the optimizer landscape; they are not part of p(y|x)"),
        ],
        size=12,
    )

    # 17 Soft usage
    s = prs.slides.add_slide(blank)
    _title(s, "Soft usage α̃_k — regularizer / reporting glue")
    add_eq(s, eqs["tilde"], left=0.4, top=1.15, width=9.2, height=1.0, font_sz=1700)
    _bullets(
        s, 0.45, 2.4, 9.2, 4.5,
        [
            "α̃_k ≈ 1 only if both switch α_k and average mass π̄_k clear threshold ε.",
            "Used to weight L_usage, L_sep, L_ov, L_γ, L_rec and to form K_eff = Σ α̃_k s_k.",
            "s_k = tanh(min pairwise mean|δ_k−δ_j| / sep_length) — duplicates count less.",
            "K_hat = #{k : α_k π̄_k ≥ ε} hard count; K_modes = peaks of p(y|x) (scientific).",
            "α̃_k is not sampled in the generative model — diagnostics + continuous sparsity.",
        ],
        size=14,
    )

    # 18 Backward
    s = prs.slides.add_slide(blank)
    _title(s, "Backward pass — what Adam updates")
    _bullets(
        s, 0.45, 1.15, 9.2, 5.8,
        [
            "Lean L ≈ NLL + λ_k usage + alive + λ_δ δ + λ_α HC + merge_pull (when merge_p>0).",
            "Gradients flow to: log α, raw_gamma, GatingNetwork MLP, OffsetNetwork MLP.",
            "Gradients do NOT flow to: μ₀, σ₀, GP kernel lengthscales, GP noise (detached).",
            "NLL: move μ_k toward data arms; sharpen Var_k; adjust π(x) to explain mass.",
            "L_usage / L_HC / alive: sparsify unused capacity without total collapse.",
            "L_δ: shrink offsets; merge_pull (controller): collapse extras when merge preferred.",
            "Structure terms (sep/rec/ov/γ/orphan) are OFF in lean champion (λ=0).",
        ],
        size=13,
    )

    # 19 Prediction
    s = prs.slides.add_slide(blank)
    _title(s, "Prediction at a new x*")
    _bullets(
        s, 0.45, 1.1, 9.2, 2.5,
        [
            "Recompute μ₀(x*), σ₀(x*) from frozen core GP (same cap rule).",
            "Deterministic α; π(x*), pinned δ(x*); γ from sigmoid(raw_gamma).",
            "Return full mixture params + law-of-total-variance moments:",
        ],
        size=14,
    )
    add_eq(s, eqs["mix_mom"], left=0.4, top=3.7, width=9.2, height=1.15, font_sz=1400)
    _bullets(
        s, 0.45, 5.1, 9.2, 1.9,
        [
            "For multimodal UQ, use the full mixture density — not only mix mean ± mix std.",
            "mix std can look “wide unimodal” even when p(y|x) has several peaks.",
        ],
        size=13,
    )

    # 20 UQ validity
    s = prs.slides.add_slide(blank)
    _title(s, "UQ validity — what is rigorously true")
    _panel(s, 0.4, 1.05, 9.2, 2.5, LIGHT)
    _text(
        s, 0.55, 1.2, 8.9, 2.2,
        "VALID as a conditional density model\n"
        "For fixed Stage-2 params and frozen (μ₀,σ₀), p(y|x)=Σ π_k N(·) is a\n"
        "proper probability density. Training ≈ empirical risk min of −log p(y|x).\n"
        "Mixture mean/variance formulas are the correct moments of that density.",
        size=14, color=SLATE,
    )
    _bullets(
        s, 0.45, 3.75, 9.2, 3.3,
        [
            "Stage 1 alone: standard GP Bayesian predictive UQ (prior → posterior).",
            "Stage 2: plug-in / MAP mixture given frozen midline — not joint Bayesian p(θ,f₀|D).",
            "σ₀ carve-out + cap are inductive biases for mode recovery, not posterior updates.",
            "Soft α makes experts continuous; still a valid mixture, not a discrete selection model.",
            "Report modes of p(y|x) when multimodality matters; do not over-interpret mix std alone.",
        ],
        size=13,
    )

    # 21 Math checks
    s = prs.slides.add_slide(blank)
    _title(s, "Math / stats checks (double-checked vs code)")
    _defs(
        s, 0.45, 1.05, 9.2, 6.1,
        [
            ("μ₀ posterior?", "YES — E[f₀|D], not prior mean m(·)"),
            ("σ₀ predictive?", "YES — √Var[y|D] from evaluate_gp_model"),
            ("∫ p(y|x) dy = 1?", "YES — finite Gaussian mixture with simplex π"),
            ("Pinning identity?", "YES — Σ_k w_ik δ_ik = 0 with w∝πα (per row)"),
            ("Var_k ≥ ε?", "YES — floor; shrink (1−αγ)∈[0,1] when α,γ∈[0,1]"),
            ("NLL = −E log p?", "YES — mean over i of −logsumexp_k(log π_k + log N_k)"),
            ("λ in p(y|x)?", "NO — optimization only; predictive density ignores λ"),
            ("Joint Bayes Stage1+2?", "NO — frozen plug-in; honest UQ claim is conditional density + GP envelope"),
            ("α always 0/1?", "NO — soft values allowed; tails clip to 0/1"),
            ("β in champion?", "Muted β≡1; historical free scale was inert on A04"),
        ],
        size=12,
    )

    # 22 Caveats / failure modes
    s = prs.slides.add_slide(blank)
    _title(s, "Statistical caveats (be honest with advisor)")
    _bullets(
        s, 0.45, 1.15, 9.2, 5.8,
        [
            "Not a full hierarchical Bayes model: Stage-2 uncertainty is not marginalized.",
            "Capping σ₀ biases the GP predictive envelope downward by design (anti fat-blob).",
            "Many λ’s ⇒ the fitted mixture is regularized MLE, not pure MLE.",
            "Identifiability: label switching / duplicate experts still possible; sep + α̃ mitigate.",
            "Champion already uses MLP gate/offsets; ConstantGate is an ablation option.",
            "K_eff is a diagnostic, not a posterior over K.",
            "Adaptive merge replaces static recovery/sep priors for uni vs multi regimes.",
        ],
        size=14,
    )

    # 23 End-to-end recipe
    s = prs.slides.add_slide(blank)
    _title(s, "End-to-end recipe (forward → back)")
    _bullets(
        s, 0.45, 1.1, 9.2, 5.9,
        [
            "Data D={(xᵢ,yᵢ)} → Stage-1 GP prior → fit → posterior predictive (μ₀,σ₀).",
            "Residuals → seed δ; optionally σ₀ = cap(σ₀, c ŝ).",
            "Initialize overcomplete K_max: log α, γ, MLP gate, MLP offsets.",
            "Forward: α,π(x),δ(x),γ → μ_k, Var_k → p(y|x) → lean L (+ merge controller).",
            "Backward: Adam updates Stage-2 MLP weights + α/γ scalars; GP stays frozen.",
            "Predict: plug-in mixture density; report K_modes vs K_hat/K_eff.",
            "UQ story: GP posterior envelope + learned multimodal residual density (plug-in).",
        ],
        size=14,
    )

    # 24 Takeaways
    s = prs.slides.add_slide(blank)
    _title(s, "Takeaways")
    _bullets(
        s, 0.45, 1.2, 9.2, 5.5,
        [
            "μ₀, σ₀ are Stage-1 GP posterior predictives — frozen thereafter (L-BFGS).",
            "Champion Stage-2: Adam on MLP gate π(x) + MLP offsets δ(x) + scalars α, γ.",
            "α = switch, π = mixture weight, δ = residual offset, γ = variance carve-out.",
            "Lean λ’s fixed: usage, alive, δ, α (+ adaptive merge). Structure/orphan off.",
            "UQ is valid as a conditional density + frozen GP envelope; not joint Bayesian Stage1+2.",
            "For multimodality, trust peaks of p(y|x), not only mixture mean ± std.",
        ],
        size=15,
    )

    prs.save(OUT)
    print(f"Wrote {OUT}")
    print("Normalizing through PowerPoint…")
    _normalize(OUT)
    print(f"Done: {OUT}")
    return OUT


if __name__ == "__main__":
    build()

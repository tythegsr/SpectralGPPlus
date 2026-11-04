"""Build MidlineSparseMoE-GP slides with native PowerPoint equations (OMML).

Equations are created via LaTeX -> MathML -> Office MML2OMML.XSL, then injected
as a14:m / m:oMath — the same structure PowerPoint's Insert > Equation uses.
A final PowerPoint COM resave normalizes the XML (oMathPara, Cambria Math).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import latex2mathml.converter
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent / "MidlineSparse_two_stage_slides.pptx"
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


def _set_run(run, text, *, size=18, bold=False, color=SLATE, font="Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def _add_title(slide, text, *, top=0.35, size=26):
    box = slide.shapes.add_textbox(Inches(0.45), Inches(top), Inches(9.1), Inches(0.55))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    _set_run(r, text, size=size, bold=True, color=NAVY)


def _textbox(slide, left, top, width, height, text, *, size=15, bold=False, color=SLATE, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    _set_run(r, text, size=size, bold=bold, color=color)
    return box


def _rounded(slide, left, top, width, height, fill, text, *, size=14, bold=True, text_color=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = LINE
    shape.line.width = Pt(1.25)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    p = tf.paragraphs[0]
    r = p.add_run()
    fg = text_color if text_color is not None else (NAVY if fill == LIGHT else WHITE)
    _set_run(r, text, size=size, bold=bold, color=fg)
    return shape


def _arrow(slide, x1, y1, x2, y2):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    conn.line.color.rgb = NAVY
    conn.line.width = Pt(2.0)
    return conn


def _bullet_block(slide, left, top, width, height, lines, *, size=14):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(4)
        r = p.add_run()
        _set_run(r, "•  " + line, size=size, color=SLATE)
    return box


def _def_block(slide, left, top, width, height, pairs, *, size=12):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (sym, defn) in enumerate(pairs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(2)
        r = p.add_run()
        _set_run(r, f"{sym}  —  {defn}", size=size, color=SLATE)
    return box


def latex_to_a14m(latex: str, *, font_sz: int = 1800) -> etree._Element:
    """Convert LaTeX to a PowerPoint a14:m wrapper containing m:oMath."""
    mathml = latex2mathml.converter.convert(latex)
    tree = etree.fromstring(mathml.encode("utf-8"))
    omml_root = _XSLT(tree).getroot()

    if omml_root.tag == f"{{{M_NS}}}oMathPara":
        omath = omml_root.find(f"{{{M_NS}}}oMath")
        if omath is None:
            omath = omml_root
    else:
        omath = omml_root

    omath = deepcopy(omath)

    # Strip Word rPr; inject DrawingML a:rPr (Cambria Math) for PowerPoint
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

    wrapper = etree.Element(f"{{{A14_NS}}}m")
    wrapper.append(omath)
    return wrapper


def add_omml_equation(
    slide,
    latex: str,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    font_sz: int = 1800,
    panel: bool = True,
):
    """Place a native PowerPoint equation in a text box (optional light panel behind)."""
    if panel:
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(left),
            Inches(top),
            Inches(width),
            Inches(height),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT
        shape.line.color.rgb = LINE
        shape.line.width = Pt(1.0)

    box = slide.shapes.add_textbox(
        Inches(left + 0.08),
        Inches(top + 0.12),
        Inches(width - 0.16),
        Inches(height - 0.2),
    )
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    # Remove empty default run content if present, then append OMML
    for child in list(p._element):
        if child.tag.endswith("}r") and not (child.find(f"{{{A_NS}}}t") is not None and (child.find(f"{{{A_NS}}}t").text or "").strip()):
            # keep structure simple: clear runs
            pass
    # Clear paragraph children that are empty runs
    for child in list(p._element):
        if child.tag.endswith("}r"):
            p._element.remove(child)
    p._element.append(latex_to_a14m(latex, font_sz=font_sz))
    return box


def _powerpoint_normalize(path: Path) -> None:
    """Open/save in PowerPoint so equations become fully native (oMathPara + fallback)."""
    try:
        import win32com.client
    except ImportError:
        print("win32com not available; skipping PowerPoint normalize pass")
        return

    app = win32com.client.Dispatch("PowerPoint.Application")
    # Prefer invisible; some installs require Visible
    try:
        app.Visible = 0
    except Exception:
        app.Visible = 1
    pres = app.Presentations.Open(str(path.resolve()), WithWindow=False)
    # Save over itself (ppSaveAsOpenXMLPresentation = 24)
    pres.SaveAs(str(path.resolve()), 24)
    pres.Close()
    try:
        app.Quit()
    except Exception:
        pass


def build() -> Path:
    # Proper LaTeX (not matplotlib mathtext) — PowerPoint equation engine
    eqs = {
        "mu_k": r"\mu_k(x)=\mu_0(x)+\alpha_k\,\delta_k(x)",
        "var_k": (
            r"\mathrm{Var}_k(x)=\bigl[\sigma_0^{\mathrm{mix}}(x)\bigr]^2"
            r"(1-\alpha_k\gamma_k)+\varepsilon"
        ),
        "py_x": (
            r"p(y\mid x)=\sum_{k=1}^{K_{\max}}\pi_k\,"
            r"\mathcal{N}\!\bigl(y;\,\mu_k(x),\mathrm{Var}_k(x)\bigr)"
        ),
        "sigma_cap": (
            r"\sigma_0^{\mathrm{cap}}(x_i)=\min\bigl(\sigma_0(x_i),"
            r"\,\max(c\,\hat{s},\,0.05)\bigr)"
        ),
        "loss_total_1": (
            r"\mathcal{L}=\mathrm{NLL}"
            r"+\lambda_k(t)\,\mathcal{L}_{\mathrm{usage}}"
            r"+\lambda_{\mathrm{sep}}\,\mathcal{L}_{\mathrm{sep}}"
            r"+\lambda_{\alpha}\,\mathcal{L}_{\mathrm{HC}}"
            r"+\lambda_{\mathrm{orphan}}\,\mathcal{L}_{\mathrm{orphan}}"
        ),
        "loss_total_2": (
            r"+\,0.5\,\mathcal{L}_{\mathrm{alive}}"
            r"+\lambda_{\mathrm{ov}}\,\mathcal{L}_{\mathrm{ov}}"
            r"+\lambda_{\delta}\,\mathcal{L}_{\delta}"
            r"+\lambda_{\gamma}\,\mathcal{L}_{\gamma}"
            r"+\lambda_{\mathrm{rec}}\,\mathcal{L}_{\mathrm{rec}}"
        ),
        "nll": (
            r"\mathrm{NLL}=-\frac{1}{n}\sum_{i=1}^{n}"
            r"\log\sum_{k=1}^{K_{\max}}"
            r"\pi_k\,\mathcal{N}\!\bigl(y_i;\,\mu_k(x_i),\mathrm{Var}_k(x_i)\bigr)"
        ),
        "usage": (
            r"\tilde{\alpha}_k=\sigma\!\left("
            r"\frac{\alpha_k\bar{\pi}_k-\varepsilon}{\tau}\right),\quad"
            r"\mathcal{L}_{\mathrm{usage}}=\sum_k\tilde{\alpha}_k"
        ),
        "sep": (
            r"\mathcal{L}_{\mathrm{sep}}=\sum_{i<j}\tilde{\alpha}_i\tilde{\alpha}_j\,"
            r"\mathrm{softplus}\!\left(m_{\mathrm{sep}}-\frac{1}{n}\sum_{\ell}"
            r"|\delta_{\ell i}-\delta_{\ell j}|\right)"
        ),
        # Avoid \left(...a-b...) — MML2OMML turns the minus into sepChr (renders as an arrow).
        "hc": (
            r"\mathcal{L}_{\mathrm{HC}}=\sum_k"
            r"\sigma(\log\alpha_k-\beta_{\mathrm{HC}}\log t),"
            r"\quad t=-\gamma_{\mathrm{HC}}/\zeta_{\mathrm{HC}}"
        ),
        "orphan": (
            r"\mathcal{L}_{\mathrm{orphan}}=\sum_k\Big["
            r"\alpha_k\,\mathrm{softplus}(\varepsilon-\bar{\pi}_k)"
            r"+\bar{\pi}_k\,\mathrm{softplus}(\varepsilon-\alpha_k)\Big]"
        ),
        "alive": (
            r"\mathcal{L}_{\mathrm{alive}}=\mathrm{softplus}"
            r"(0.8-\sum_k\tilde{\alpha}_k)"
        ),
        "ov": (
            r"\mathrm{gap}_{ij}=\frac{\mathrm{abs}(\bar{\mu}_i-\bar{\mu}_j)}"
            r"{\bar{\sigma}_i+\bar{\sigma}_j},"
            r"\quad"
            r"\mathcal{L}_{\mathrm{ov}}=\sum_{i<j}\tilde{\alpha}_i\tilde{\alpha}_j\,"
            r"\mathrm{softplus}(m_{\mathrm{ov}}-\mathrm{gap}_{ij})"
        ),
        "delta": r"\mathcal{L}_{\delta}=\mathrm{mean}(\delta^2)",
        "gamma": r"\mathcal{L}_{\gamma}=\sum_k\tilde{\alpha}_k\,(1-\gamma_k)^2",
        "rec": (
            r"\mathcal{L}_{\mathrm{rec}}=\sum_{i<j}\tilde{\alpha}_i\tilde{\alpha}_j\,"
            r"\mathrm{softplus}(m_{\mathrm{rec}}-\mathrm{gap}_{ij}),"
            r"\quad m_{\mathrm{rec}}=1"
        ),
        "lambda_anneal": (
            r"\lambda_k(t)=\lambda_k^{\mathrm{start}}"
            r"+\frac{t-1}{T-1}\bigl(\lambda_k-\lambda_k^{\mathrm{start}}\bigr)"
        ),
        "tilde_alpha": (
            r"\tilde{\alpha}_k=\sigma\!\left("
            r"\frac{\alpha_k\bar{\pi}_k-\varepsilon}{\tau}\right)"
        ),
        "kmodes": (
            r"K_{\mathrm{modes}}(x_{\star})"
            r"=\#\{\text{local maxima of }p(y\mid x_{\star})\}"
        ),
        # Hard-Concrete: stretch-and-clip (defaults from HardConcreteGate)
        "hc_defaults": (
            r"\gamma_{\mathrm{HC}}=-0.1,\quad"
            r"\zeta_{\mathrm{HC}}=1.1,\quad"
            r"\beta_{\mathrm{HC}}=2/3"
        ),
        "alpha_det": (
            r"\alpha_k=\mathrm{clip}_{[0,1]}"
            r"\big(\sigma(\log\alpha_k)\,(\zeta_{\mathrm{HC}}-\gamma_{\mathrm{HC}})"
            r"+\gamma_{\mathrm{HC}}\big)"
        ),
        "alpha_zero": (
            r"\alpha_k=0\ \mathrm{if}\ "
            r"\sigma(\log\alpha_k)\le"
            r"\frac{0.1}{1.2}\approx 0.083"
        ),
        "alpha_one": (
            r"\alpha_k=1\ \mathrm{if}\ "
            r"\sigma(\log\alpha_k)\ge"
            r"\frac{1.1}{1.2}\approx 0.917"
        ),
        "hc_thresh": (
            r"t=-\gamma_{\mathrm{HC}}/\zeta_{\mathrm{HC}}"
            r"=0.1/1.1\approx 0.0909"
        ),
    }

    # Fail fast if any LaTeX fails conversion
    for name, latex in eqs.items():
        latex_to_a14m(latex)

    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ----- 1 Title -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "MidlineSparseMoE-GP", top=2.1, size=36)
    _textbox(
        s,
        0.5,
        2.9,
        9.0,
        1.4,
        "Two-stage model for co-located multimodal regression\n"
        "Scientific target: modes of p(y|x), not expert count",
        size=18,
        color=SLATE,
        align=PP_ALIGN.CENTER,
    )
    _textbox(
        s,
        0.5,
        6.6,
        9.0,
        0.4,
        "Advisor overview  ·  equations = native PowerPoint Equation objects",
        size=12,
        color=GRAY,
        align=PP_ALIGN.CENTER,
    )

    # ----- 2 Two-stage diagram -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "Two-stage method at a glance")
    _rounded(s, 0.45, 1.4, 2.3, 1.55, NAVY, "Data\n{(xᵢ, yᵢ)}\ni = 1…n", size=15)
    _arrow(s, 2.85, 2.15, 3.2, 2.15)
    _rounded(s, 3.25, 1.25, 2.85, 1.85, TEAL, "Stage 1\nExact GP midline\nμ₀(x), σ₀(x)\n→ freeze", size=14)
    _arrow(s, 6.2, 2.15, 6.55, 2.15)
    _rounded(
        s, 6.6, 1.2, 2.95, 1.95, ORANGE,
        "Stage 2\nMLP π(x)+δ(x)\n+ Adam scalars α,γ\nAdam on lean L",
        size=12,
    )
    _arrow(s, 8.05, 3.25, 8.05, 3.7)
    _rounded(s, 5.7, 3.8, 3.85, 1.55, GREEN, "Predict p(y|x)\nReport K_modes\nvs K̂ / K_eff", size=14)
    _textbox(
        s,
        0.45,
        3.9,
        5.0,
        3.0,
        "Key idea\n"
        "• One shared GP for the midline (Stage-1 L-BFGS)\n"
        "• Experts = additive residuals (not separate GPs)\n"
        "• Champion: MLP gate π(x) + MLP offsets δ(x)\n"
        "• Adam scalars: α (Hard-Concrete), γ carve-out\n"
        "• Modes of the density are the scientific target",
        size=13,
        color=SLATE,
    )

    # ----- 3 Symbol glossary A -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "Symbol glossary (1/2) — data & Stage 1")
    _def_block(
        s,
        0.55,
        1.15,
        9.0,
        5.8,
        [
            ("x, y", "input and scalar response"),
            ("n", "number of training pairs"),
            ("(xᵢ, yᵢ)", "i-th training example"),
            ("μ₀(x)", "Stage-1 GP posterior mean (midline); frozen in Stage 2"),
            ("σ₀(x)", "Stage-1 GP posterior std (noise/ceiling)"),
            ("rᵢ = yᵢ − μ₀(xᵢ)", "residual after Stage 1"),
            ("ŝ", "sample std of residuals {rᵢ}"),
            ("c", "cap multiplier for σ₀ (champion p12 uses c on residual scale)"),
            ("σ₀^cap(x)", "min(σ₀(x), max(c ŝ, 0.05)) — capped GP std"),
            ("σ₀^mix(x)", "GP std used in Stage 2: σ₀^cap (p12) or full σ₀ (legacy)"),
            ("K_max", "overcomplete expert budget (e.g. 6)"),
            ("k = 1…K_max", "expert index"),
            ("ε", "tiny floor for numerical stability / soft thresholds"),
            ("T, t", "total Adam steps and current step"),
        ],
        size=14,
    )

    # ----- 4 Symbol glossary B -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "Symbol glossary (2/2) — Stage 2 & reporting")
    _def_block(
        s,
        0.55,
        1.1,
        9.0,
        6.0,
        [
            ("α_k ∈ [0,1]", "Hard-Concrete on/off switch for expert k"),
            ("π_k(x)", "mixture / gate; champion = MLP(x)→softmax (ConstantGate optional)"),
            ("π̄_k", "average gate mass = (1/n) Σᵢ π_k(xᵢ)"),
            ("δ_k(x)", "offset; champion = OffsetNetwork MLP; pinned π·α-weighted center"),
            ("γ_k ∈ (0,1)", "variance carve-out fraction (sigmoid of raw_gamma)"),
            ("μ_k(x)", "expert mean = μ₀(x) + α_k δ_k(x)"),
            ("Var_k(x)", "expert variance = [σ₀^mix]² (1 − α_k γ_k) + ε"),
            ("α̃_k", "soft usage score from α_k and π̄_k (Option E)"),
            ("τ", "temperature for soft usage α̃_k"),
            ("β_HC, γ_HC, ζ_HC", "HC defaults: β=2/3, stretch [γ,ζ]=[-0.1, 1.1] (not expert variance)"),
            ("λ_·", "scalar weights on each regularizer (λ_k may anneal in t)"),
            ("gap_ij", "standardized mean gap |μ̄ᵢ−μ̄ⱼ| / (σ̄ᵢ+σ̄ⱼ)"),
            ("K_modes", "# local maxima of p(y|x) — scientific metric"),
            ("K̂, K_eff", "hard active count; usage×separation effective count (diagnostic)"),
        ],
        size=13,
    )

    # ----- 5 Stage 1 -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "Stage 1 — Exact GP midline")
    _rounded(
        s,
        0.5,
        1.15,
        9.0,
        0.75,
        TEAL,
        "Fit one exact GPR on ALL training points  →  μ₀(x), σ₀(x)",
        size=16,
    )
    add_omml_equation(s, eqs["sigma_cap"], left=0.45, top=2.1, width=9.1, height=1.2, font_sz=1600)
    _def_block(
        s,
        0.55,
        3.5,
        9.0,
        1.3,
        [
            ("σ₀^cap", "optional ceiling so Stage 2 does not ride a fat noise band (p12)"),
            ("σ₀^mix", "denotes whichever ceiling Stage 2 uses (cap or full)"),
        ],
        size=13,
    )
    _bullet_block(
        s,
        0.55,
        5.0,
        9.0,
        2.0,
        [
            "Full domain — no partitioning data by expert.",
            "After Stage 1 the GP is frozen; Stage 2 never updates it.",
            "Residual quantiles seed initial expert offsets δ_k.",
        ],
        size=15,
    )

    # ----- 6 Stage 2 structure -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "Stage 2 — Mixture on the midline")
    add_omml_equation(s, eqs["mu_k"], left=0.45, top=1.05, width=9.1, height=0.95, font_sz=1800)
    add_omml_equation(s, eqs["var_k"], left=0.45, top=2.15, width=9.1, height=0.95, font_sz=1500)
    add_omml_equation(s, eqs["py_x"], left=0.45, top=3.25, width=9.1, height=0.95, font_sz=1500)
    _def_block(
        s,
        0.55,
        4.4,
        9.0,
        2.7,
        [
            ("α_k", "Hard-Concrete switch (Adam scalar); unused experts → 0"),
            ("π_k(x)", "mixture weight — champion GatingNetwork MLP"),
            ("δ_k(x)", "residual offset — champion OffsetNetwork MLP"),
            ("γ_k", "carve-out (Adam scalar): γ→1 sharpens; γ→0 keeps fat blob"),
            ("ε", "tiny variance floor"),
        ],
        size=13,
    )

    # ----- How α becomes 0 or 1 -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "How α_k becomes 0 or 1 — Hard-Concrete defaults")
    add_omml_equation(s, eqs["hc_defaults"], left=0.45, top=0.95, width=9.1, height=0.75, font_sz=1600)
    add_omml_equation(s, eqs["alpha_det"], left=0.45, top=1.8, width=9.1, height=0.9, font_sz=1400)
    add_omml_equation(s, eqs["alpha_zero"], left=0.45, top=2.8, width=9.1, height=0.75, font_sz=1500)
    add_omml_equation(s, eqs["alpha_one"], left=0.45, top=3.65, width=9.1, height=0.75, font_sz=1500)
    add_omml_equation(s, eqs["hc_thresh"], left=0.45, top=4.5, width=9.1, height=0.7, font_sz=1500)
    _bullet_block(
        s,
        0.55,
        5.35,
        9.0,
        1.9,
        [
            "log α_k is free (Adam); α_k is not itself a free [0,1] parameter.",
            "Stretch into [γ_HC, ζ_HC]=[-0.1, 1.1], then clip_[0,1] → exact 0/1 in the tails.",
            "Training uses a stochastic Concrete sample; eval uses the deterministic formula.",
        ],
        size=13,
    )

    # ----- Soft usage -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "Shared building block — soft usage α̃_k")
    add_omml_equation(s, eqs["tilde_alpha"], left=0.45, top=1.2, width=9.1, height=1.25, font_sz=2000)
    _def_block(
        s,
        0.55,
        2.7,
        9.0,
        2.2,
        [
            ("α̃_k ≈ 1", "expert is “on”: both α_k and π̄_k are above ε"),
            ("α̃_k ≈ 0", "expert is unused for sparsity / separation / γ prior"),
            ("Used in", "L_usage, L_sep, L_alive, L_ov, L_γ, L_rec, and K_eff"),
        ],
        size=14,
    )
    _bullet_block(
        s,
        0.55,
        5.1,
        9.0,
        2.0,
        [
            "Makes every regularizer continuous and usage-aware.",
            "Avoids hard discrete on/off decisions inside the Adam loop.",
        ],
        size=15,
    )

    # ----- Loss overview -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "Stage 2 loss — total objective")
    add_omml_equation(s, eqs["loss_total_1"], left=0.45, top=1.05, width=9.1, height=1.05, font_sz=1400)
    add_omml_equation(s, eqs["loss_total_2"], left=0.45, top=2.2, width=9.1, height=0.95, font_sz=1400)
    add_omml_equation(s, eqs["lambda_anneal"], left=0.45, top=3.3, width=9.1, height=1.0, font_sz=1600)
    _def_block(
        s,
        0.55,
        4.5,
        9.0,
        1.4,
        [
            ("NLL", "mixture negative log-likelihood (only term that fits data)"),
            ("λ_k, alive, λ_δ, λ_α", "lean champion fixed weights (+ adaptive merge)"),
            ("λ_sep/rec/ov/γ/orphan", "OFF (=0) in current lean champion"),
        ],
        size=13,
    )
    _bullet_block(
        s,
        0.55,
        6.0,
        9.0,
        1.2,
        [
            "No mixture-mean MSE — would collapse co-located modes toward μ₀.",
            "Adam updates Stage-2 MLP weights + α/γ scalars; Stage-1 GP stays frozen.",
        ],
        size=13,
    )

    # ----- Per-term slides -----
    terms = [
        (
            "NLL — mixture negative log-likelihood",
            "nll",
            [
                ("n", "training size"),
                ("π_k", "mixture weight for expert k"),
                ("μ_k, Var_k", "expert mean and variance at xᵢ"),
                ("N(·)", "Gaussian density"),
            ],
            [
                "Why: Directly trains the predictive density p(y|x).",
                "Justification: Correct likelihood for a Gaussian mixture (logsumexp).",
                "This is the only term that “wants” to explain the data.",
            ],
        ),
        (
            "L_usage — soft expert count (sparsity)",
            "usage",
            [
                ("α̃_k", "soft usage of slot k"),
                ("α_k, π̄_k", "switch and average gate mass"),
                ("ε, τ", "threshold and temperature for the soft step"),
                ("σ(·)", "logistic sigmoid"),
            ],
            [
                "Why: Overcomplete K_max tends to keep unused experts alive.",
                "Justification: Continuous L0-style pressure on used slots.",
                "Annealed λ_k: form modes before pruning (P4).",
            ],
        ),
        (
            "L_sep — separate offset curves",
            "sep",
            [
                ("δ_ℓi", "pinned offset of expert i at training row ℓ"),
                ("m_sep", "target separation margin on mean |δᵢ−δⱼ|"),
                ("α̃_i α̃_j", "only push pairs that are both in use"),
                ("softplus", "smooth hinge: softplus(z)=log(1+e^z)"),
            ],
            [
                "Why: Two experts can copy the same residual curve.",
                "Justification: Usage-weighted push so distinct arms stay distinct.",
                "Feeds K_eff = Σ α̃_k s_k (duplicates get low separation score).",
            ],
        ),
        (
            "L_HC — Hard-Concrete complexity",
            "hc",
            [
                ("log α_k", "unconstrained Hard-Concrete location (Adam learns this)"),
                ("β_HC = 2/3", "Concrete temperature (HardConcreteGate.beta)"),
                ("γ_HC = −0.1, ζ_HC = 1.1", "stretch limits (HardConcreteGate.gamma / .zeta)"),
                ("t = −γ_HC/ζ_HC ≈ 0.0909", "L0-surrogate threshold in L_HC"),
            ],
            [
                "Why: Push mass of α_k into the clipped tails → exact 0 or 1.",
                "Justification: Louizos et al. Concrete / Hard-Concrete L0 complexity.",
                "L_HC ≈ expected # of non-zero gates under the Concrete prior.",
            ],
        ),
        (
            "L_orphan — couple α and π",
            "orphan",
            [
                ("α_k", "switch for expert k"),
                ("π̄_k", "average mixture weight"),
                ("ε", "dead-mass threshold"),
            ],
            [
                "Why: Block α-on/π≈0 (dead experts) and π-on/α≈0 (dead mass).",
                "Justification: Bidirectional π–α coupling stabilizes A04 fits.",
                "Stops runaway offsets and dead-π dominance.",
            ],
        ),
        (
            "L_alive — keep at least some mass",
            "alive",
            [
                ("Σ_k α̃_k", "total soft usage"),
                ("0.8", "soft floor on total usage"),
                ("weight 0.5", "fixed coefficient in front of L_alive in code"),
            ],
            [
                "Why: Pure sparsity can kill all experts → midline only.",
                "Justification: Soft floor so multimodal data keeps ≥ ~1 used slot.",
                "Balances L_usage without a hard constraint.",
            ],
        ),
        (
            "L_ov — overlap / mean-gap",
            "ov",
            [
                ("μ̄_k, σ̄_k", "training-average expert mean and std"),
                ("gap_ij", "separation in units of combined widths"),
                ("m_ov", "soft target for the standardized gap"),
            ],
            [
                "Why: Means inside each other’s noise bands merge into one hump.",
                "Justification: Standardized separation discourages merged peaks.",
                "Does not prescribe a target K — only discourages overlap.",
            ],
        ),
        (
            "L_δ — offset L2",
            "delta",
            [
                ("δ", "all pinned offsets (over data / experts)"),
                ("λ_δ", "ridge weight (e.g. 0.02 in treated A04)"),
            ],
            [
                "Why: Offsets can explode (expert means diving far from data).",
                "Justification: Mild ridge keeps residual experts on midline scale.",
                "Paired with orphan coupling in A04 settings.",
            ],
        ),
        (
            "L_γ — carve-out prior (p12)",
            "gamma",
            [
                ("γ_k", "carve-out fraction for expert k"),
                ("α̃_k", "soft usage — prior acts on active slots"),
                ("(1−γ_k)²", "penalty for not carving (staying as wide as σ₀)"),
            ],
            [
                "Why: Without pressure, γ→0 and experts stay as wide as σ₀ (fat blob).",
                "Justification: Active experts should sharpen modes below the GP ceiling.",
                "Part of champion p12 with capped σ₀; free β-scale was muted.",
            ],
        ),
        (
            "L_rec — recovery margin (+rec)",
            "rec",
            [
                ("gap_ij", "same standardized gap as in L_ov"),
                ("m_rec = 1", "Behboodian-scale margin (|Δμ| ≳ σᵢ+σⱼ)"),
                ("λ_rec", "weight (0 by default; 0.15 in recovery probes)"),
            ],
            [
                "Why: 1D GMM modes appear only if means separate vs σ.",
                "Justification: Enforce gap ≥ 1 so the fit stays multimodal-capable.",
                "Certificate checks this post-fit; does not prove match to truth.",
            ],
        ),
    ]

    for title, eq_name, defs, bullets in terms:
        s = prs.slides.add_slide(blank)
        _add_title(s, title, size=22)
        add_omml_equation(
            s, eqs[eq_name], left=0.45, top=1.05, width=9.1, height=1.4, font_sz=1500
        )
        _textbox(s, 0.55, 2.6, 9.0, 0.35, "Variables", size=14, bold=True, color=NAVY)
        _def_block(s, 0.55, 2.95, 9.0, 1.7, defs, size=13)
        _textbox(s, 0.55, 4.7, 9.0, 0.35, "Why / justification", size=14, bold=True, color=NAVY)
        _bullet_block(s, 0.55, 5.1, 9.0, 2.1, bullets, size=13)

    # ----- Closing -----
    s = prs.slides.add_slide(blank)
    _add_title(s, "What to remember")
    add_omml_equation(s, eqs["kmodes"], left=0.45, top=1.15, width=9.1, height=1.15, font_sz=1600)
    _bullet_block(
        s,
        0.55,
        2.55,
        9.0,
        4.5,
        [
            "Stage 1: one exact GP midline on the full domain (then freeze).",
            "Stage 2: sparse residual mixture — fit NLL + structure regularizers.",
            "Scientific metric: K_modes (peaks of p(y|x)), not K̂ experts.",
            "Champion path: cap σ₀ + γ prior (p12); optional +rec for mean–σ separation.",
            "Co-located multimodality is the design target; x-split tasks need different bias.",
            "Click any equation in PowerPoint to edit it with the built-in equation tools.",
        ],
        size=15,
    )

    prs.save(OUT)
    print(f"Wrote raw OMML deck: {OUT}")
    print("Normalizing equations through PowerPoint…")
    _powerpoint_normalize(OUT)
    print(f"Normalized: {OUT}")
    return OUT


if __name__ == "__main__":
    build()

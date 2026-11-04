"""Slides: purpose of true p(y|x) vs local data KDE, with full formula derivations.

Native PowerPoint equations via LaTeX -> MathML -> MML2OMML (same pipeline as
``make_midline_sparse_slides.py``).

Run (gpplus2026):
  python make_true_kde_marginal_slides.py
"""

from __future__ import annotations

import shutil
import unicodedata
import zipfile
from copy import deepcopy
from pathlib import Path

import latex2mathml.converter
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent / "True_vs_DataKDE_marginals.pptx"
MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "foundation_compare" / "figures"

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
RED = RGBColor(0xC5, 0x30, 0x30)
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


def _add_title(slide, text, *, top=0.28, size=24):
    box = slide.shapes.add_textbox(Inches(0.4), Inches(top), Inches(9.2), Inches(0.5))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    _set_run(r, text, size=size, bold=True, color=NAVY)


def _textbox(slide, left, top, width, height, text, *, size=14, bold=False, color=SLATE):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    _set_run(r, text, size=size, bold=bold, color=color)
    return box


def _bullet_block(slide, left, top, width, height, lines, *, size=13):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(3)
        r = p.add_run()
        _set_run(r, "•  " + line, size=size, color=SLATE)
    return box


def _card(slide, left, top, width, height, fill=LIGHT):
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
    shape.line.width = Pt(1.0)
    return shape


def _bmp_safe(text: str) -> str:
    """Remap astral-plane chars that PowerPoint COM corrupts on resave.

    latex2mathml emits U+1D4xx script/blackboard letters for \\mathcal / \\mathbb;
    Office then writes lone UTF-16 surrogates into slide XML (invalid UTF-8).
    """
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if o > 0xFFFF or 0xD800 <= o <= 0xDFFF:
            name = unicodedata.name(ch, "")
            token = name.split()[-1] if name else ""
            out.append(token if len(token) == 1 and token.isalpha() else "?")
        else:
            out.append(ch)
    return "".join(out)


def latex_to_a14m(latex: str, *, font_sz: int = 1600) -> etree._Element:
    # Prefer BMP-safe macros before conversion (defense in depth with _bmp_safe).
    latex = (
        latex.replace(r"\mathcal{N}", r"\mathrm{N}")
        .replace(r"\mathbb{E}", r"\mathrm{E}")
    )
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
    for t_el in omath.xpath(".//m:t", namespaces={"m": M_NS}):
        if t_el.text:
            t_el.text = _bmp_safe(t_el.text)
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


def _pptx_is_readable(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith(".xml"):
                    continue
                data = zf.read(name)
                data.decode("utf-8")
                etree.fromstring(data)
        Presentation(str(path))
        return True
    except Exception as exc:
        print(f"PPTX validation failed for {path.name}: {exc}")
        return False


def add_omml_equation(
    slide,
    latex: str,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    font_sz: int = 1600,
    panel: bool = True,
):
    if panel:
        _card(slide, left, top, width, height)
    box = slide.shapes.add_textbox(
        Inches(left + 0.08),
        Inches(top + 0.1),
        Inches(width - 0.16),
        Inches(height - 0.16),
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


def _powerpoint_normalize(path: Path) -> None:
    try:
        import win32com.client
    except ImportError:
        print("win32com not available; skipping PowerPoint normalize pass")
        return
    backup = path.with_suffix(".pre_normalize.pptx")
    shutil.copy2(path, backup)
    app = win32com.client.Dispatch("PowerPoint.Application")
    try:
        app.Visible = 0
    except Exception:
        app.Visible = 1
    try:
        pres = app.Presentations.Open(str(path.resolve()), WithWindow=False)
        pres.SaveAs(str(path.resolve()), 24)
        pres.Close()
    finally:
        try:
            app.Quit()
        except Exception:
            pass
    if not _pptx_is_readable(path):
        print("Normalize corrupted PPTX; restoring pre-normalize copy")
        shutil.copy2(backup, path)
    else:
        backup.unlink(missing_ok=True)


def _try_add_picture(slide, path: Path, left, top, width):
    if path.is_file():
        slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))
        return True
    return False


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ------------------------------------------------------------------
    # 1. Title
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "True p(y|x) vs local data KDE", top=2.2, size=32)
    _textbox(
        s,
        0.8,
        3.0,
        8.4,
        1.2,
        "Why both appear on foundation / MidlineSparse marginal panels,\n"
        "and how every formula is derived from the synthetic DGPs.",
        size=16,
        color=GRAY,
    )
    _textbox(
        s,
        0.8,
        6.6,
        8.4,
        0.4,
        "Source: plot_foundation_marginals.py  ·  experiments_MOEGP",
        size=12,
        color=GRAY,
    )

    # ------------------------------------------------------------------
    # 2. Purpose
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Purpose: two reference curves on each marginal panel")
    _card(s, 0.4, 1.0, 4.4, 5.6)
    _textbox(s, 0.6, 1.15, 4.0, 0.4, "True p(y | x)  (red)", size=16, bold=True, color=RED)
    _bullet_block(
        s,
        0.6,
        1.7,
        4.0,
        4.5,
        [
            "Analytical density of the synthetic data-generating process (DGP).",
            "Gold standard for whether a model recovered multimodality / skew / heteroscedasticity.",
            "Does not depend on finite-sample noise or probe bandwidth.",
            "Answers: “What should p(y|x) look like in population?”",
        ],
        size=13,
    )
    _card(s, 5.2, 1.0, 4.4, 5.6)
    _textbox(s, 5.4, 1.15, 4.0, 0.4, "Data KDE (local x)  (gray dashed)", size=16, bold=True, color=GRAY)
    _bullet_block(
        s,
        5.4,
        1.7,
        4.0,
        4.5,
        [
            "Nonparametric estimate from train+test points near the probe x.",
            "Shows what the finite dataset actually supports at that x.",
            "Can disagree with truth when local n is small or modes are sparse.",
            "Answers: “What does this sample look like near x?”",
        ],
        size=13,
    )

    # ------------------------------------------------------------------
    # 3. Why both
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Why keep both (not just one)?")
    _bullet_block(
        s,
        0.5,
        1.0,
        9.0,
        5.8,
        [
            "Scoring rules (CRPS, NLL) compare models to observed y; they do not visualize density shape.",
            "True p(y|x) diagnoses model misspecification of the DGP (e.g. unimodal GP on bimodal A01).",
            "Local KDE diagnoses whether the sample is informative at that probe (especially co-located modes).",
            "If model ≈ KDE but ≠ truth → under-sampling / hard local regime, not necessarily a bad model.",
            "If model ≈ truth but ≠ KDE → model may extrapolate the DGP better than the local histogram.",
            "If model ≠ both → clear failure of the predictive density at that x.",
        ],
        size=15,
    )

    # ------------------------------------------------------------------
    # 4. Generic setup
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Setup: conditional density at a probe")
    _textbox(
        s,
        0.5,
        0.95,
        9.0,
        0.5,
        "All panels evaluate densities on a fine y-grid at a fixed probe x = x★.",
        size=14,
        color=GRAY,
    )
    add_omml_equation(
        s,
        r"p_{\mathrm{true}}(y\mid x^{\star}),\qquad"
        r"\hat p_{\mathrm{KDE}}(y\mid x^{\star}),\qquad"
        r"p_{\mathrm{model}}(y\mid x^{\star})",
        left=0.5,
        top=1.6,
        width=9.0,
        height=1.1,
        font_sz=1800,
    )
    _bullet_block(
        s,
        0.5,
        3.0,
        9.0,
        3.5,
        [
            "True: closed form from the DGP used to generate the toy (A01–A05).",
            "KDE: Gaussian kernel density of y-values whose x lie near x★.",
            "Model: Gaussian mixture predictive from MidlineSparse / GP / TabPFN / TabFM / …",
            "Implemented in plot_foundation_marginals.TRUE_PDF and empirical_local_pdf.",
        ],
        size=14,
    )

    # ------------------------------------------------------------------
    # 5. A01
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "A01 bimodal — derivation of true p(y|x)")
    _textbox(
        s,
        0.5,
        0.9,
        9.0,
        0.55,
        "At each x, pick mode m∈{1,2} with equal probability, then add Gaussian noise.",
        size=13,
        color=GRAY,
    )
    add_omml_equation(
        s,
        r"\mu_1(x)=2x+5,\qquad \mu_2(x)=-x+15,\qquad \sigma=2",
        left=0.5,
        top=1.5,
        width=9.0,
        height=0.85,
        font_sz=1700,
    )
    add_omml_equation(
        s,
        r"Y\mid X=x,\,M=m \;\sim\; \mathcal{N}\!\bigl(\mu_m(x),\sigma^{2}\bigr),"
        r"\quad \mathbb{P}(M=1)=\mathbb{P}(M=2)=\tfrac12",
        left=0.5,
        top=2.5,
        width=9.0,
        height=0.95,
        font_sz=1500,
    )
    add_omml_equation(
        s,
        r"p(y\mid x)=\sum_{m=1}^{2}\mathbb{P}(M=m)\,"
        r"\varphi\!\bigl(y;\mu_m(x),\sigma^{2}\bigr)"
        r"=\tfrac12\,\varphi(y;2x+5,4)+\tfrac12\,\varphi(y;-x+15,4)",
        left=0.5,
        top=3.6,
        width=9.0,
        height=1.15,
        font_sz=1400,
    )
    _textbox(
        s,
        0.5,
        5.0,
        9.0,
        1.8,
        "Here φ(y; μ, σ²) is the N(μ,σ²) density. Modes are co-located in x "
        "(both active everywhere); separation grows with x, so panels near "
        "x≈8 show clear bimodality while x≈2 looks nearly unimodal.",
        size=13,
        color=SLATE,
    )

    # ------------------------------------------------------------------
    # 6. Unimodal + A02
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Unimodal control & A02 step — true densities")
    _textbox(s, 0.5, 0.95, 9.0, 0.35, "Unimodal negative control", size=15, bold=True, color=TEAL)
    add_omml_equation(
        s,
        r"Y=0.5\,X+2+0.3\,\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,1)"
        r"\quad\Rightarrow\quad p(y\mid x)=\varphi(y;\,0.5x+2,\,0.09)",
        left=0.5,
        top=1.35,
        width=9.0,
        height=1.0,
        font_sz=1500,
    )
    _textbox(s, 0.5, 2.55, 9.0, 0.35, "A02 discontinuous step (separated regimes)", size=15, bold=True, color=TEAL)
    add_omml_equation(
        s,
        r"\mu(x)=\mathbf{1}\{x\ge 0.5\},\qquad"
        r"Y=\mu(X)+0.05\,\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,1)",
        left=0.5,
        top=2.95,
        width=9.0,
        height=0.95,
        font_sz=1500,
    )
    add_omml_equation(
        s,
        r"p(y\mid x)=\varphi\!\bigl(y;\,\mathbf{1}\{x\ge 0.5\},\,0.05^{2}\bigr)",
        left=0.5,
        top=4.05,
        width=9.0,
        height=0.85,
        font_sz=1600,
    )
    _textbox(
        s,
        0.5,
        5.15,
        9.0,
        1.6,
        "Unimodal: single Gaussian — models should not invent modes.\n"
        "A02: mean jumps at 0.5; locally still unimodal, but hard for smooth GPs near the kink.",
        size=13,
        color=SLATE,
    )

    # ------------------------------------------------------------------
    # 7. A04
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "A04 trimodal — unequal widths")
    _textbox(
        s,
        0.5,
        0.9,
        9.0,
        0.45,
        "Three co-located Gaussian modes with distinct slopes and noise levels.",
        size=13,
        color=GRAY,
    )
    add_omml_equation(
        s,
        r"\mu_0=2x+5,\;\sigma_0=0.8;\quad"
        r"\mu_1=-0.5x+12,\;\sigma_1=2;\quad"
        r"\mu_2=x-2,\;\sigma_2=4",
        left=0.4,
        top=1.45,
        width=9.2,
        height=0.95,
        font_sz=1400,
    )
    add_omml_equation(
        s,
        r"\mathbb{P}(M=m)=\tfrac13,\qquad"
        r"Y\mid X=x,M=m\sim\mathcal{N}(\mu_m(x),\sigma_m^{2})",
        left=0.5,
        top=2.55,
        width=9.0,
        height=0.9,
        font_sz=1500,
    )
    add_omml_equation(
        s,
        r"p(y\mid x)=\sum_{m=0}^{2}\tfrac13\,\varphi\!\bigl(y;\mu_m(x),\sigma_m^{2}\bigr)",
        left=0.5,
        top=3.6,
        width=9.0,
        height=0.9,
        font_sz=1700,
    )
    _bullet_block(
        s,
        0.5,
        4.7,
        9.0,
        2.2,
        [
            "Wide component (σ=4) can mask narrow peaks in finite samples → KDE may look unimodal.",
            "True density still has up to three modes; MidlineSparse is scored on recovering them.",
        ],
        size=14,
    )

    # ------------------------------------------------------------------
    # 8. A05 overview
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "A05 heteroscedastic skew — generative story")
    add_omml_equation(
        s,
        r"Y=\mu(X)+\varepsilon(X),\qquad \mu(x)=0.5x+2",
        left=0.5,
        top=1.0,
        width=9.0,
        height=0.85,
        font_sz=1800,
    )
    _textbox(
        s,
        0.5,
        2.0,
        9.0,
        0.4,
        "Noise ε(x) is a contaminated asymmetric Laplace (still unimodal at each x).",
        size=13,
        color=GRAY,
    )
    add_omml_equation(
        s,
        r"w=\frac{x-x_{\mathrm{lo}}}{x_{\mathrm{hi}}-x_{\mathrm{lo}}}\in[0,1],\quad"
        r"x_{\mathrm{lo}}=0,\;x_{\mathrm{hi}}=10",
        left=0.5,
        top=2.5,
        width=9.0,
        height=0.85,
        font_sz=1500,
    )
    add_omml_equation(
        s,
        r"\sigma(x)=\sigma_{\min}+(\sigma_{\max}-\sigma_{\min})\,w^{1.5},"
        r"\quad \sigma_{\min}=0.2,\;\sigma_{\max}=2",
        left=0.5,
        top=3.5,
        width=9.0,
        height=0.9,
        font_sz=1500,
    )
    add_omml_equation(
        s,
        r"\tau(x)=\tau_{\min}+(\tau_{\max}-\tau_{\min})\,w,"
        r"\quad \tau_{\min}=0.35,\;\tau_{\max}=0.75",
        left=0.5,
        top=4.55,
        width=9.0,
        height=0.9,
        font_sz=1500,
    )
    _textbox(
        s,
        0.5,
        5.65,
        9.0,
        1.2,
        "σ(x) grows nonlinearly (heteroscedasticity). τ(x) is the probability of the "
        "positive exponential side (skew). Contamination rate ε_c = 0.12, κ = 4.",
        size=13,
        color=SLATE,
    )

    # ------------------------------------------------------------------
    # 9. A05 AL density
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "A05 — asymmetric Laplace scales (Var ≈ σ²)")
    _textbox(
        s,
        0.5,
        0.9,
        9.0,
        0.55,
        "Two-sided exponential: P(+)=τ, scale b₊ on the positive side, b₋ on the negative.",
        size=13,
        color=GRAY,
    )
    add_omml_equation(
        s,
        r"b_{+}=\frac{\sigma}{\sqrt{2\tau}},\qquad"
        r"b_{-}=\frac{\sigma}{\sqrt{2(1-\tau)}}",
        left=0.5,
        top=1.55,
        width=9.0,
        height=0.95,
        font_sz=1800,
    )
    add_omml_equation(
        s,
        r"\mathrm{Var}(\varepsilon_{\mathrm{raw}})=2\bigl(\tau b_{+}^{2}+(1-\tau)b_{-}^{2}\bigr)=\sigma^{2}",
        left=0.5,
        top=2.7,
        width=9.0,
        height=0.9,
        font_sz=1600,
    )
    add_omml_equation(
        s,
        r"z\ge 0:\quad f_{\mathrm{raw}}(z)=\tau\,b_{+}^{-1}\,e^{-z/b_{+}}",
        left=0.5,
        top=3.75,
        width=9.0,
        height=0.75,
        font_sz=1500,
    )
    add_omml_equation(
        s,
        r"z<0:\quad f_{\mathrm{raw}}(z)=(1-\tau)\,b_{-}^{-1}\,e^{z/b_{-}}",
        left=0.5,
        top=4.6,
        width=9.0,
        height=0.75,
        font_sz=1500,
    )
    _textbox(
        s,
        0.5,
        5.55,
        9.0,
        1.4,
        "When τ=1/2, b₊=b₋=σ/√2 and the law is symmetric Laplace with variance σ². "
        "Away from 1/2 the density is skewed while variance still tracks σ².",
        size=13,
        color=SLATE,
    )

    # ------------------------------------------------------------------
    # 10. A05 centering + contamination
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "A05 — zero-mean centering and contamination mixture")
    add_omml_equation(
        s,
        r"\mathbb{E}[\varepsilon_{\mathrm{raw}}]=\tau b_{+}-(1-\tau)b_{-}"
        r"\;=\;\Delta(x)",
        left=0.5,
        top=1.0,
        width=9.0,
        height=0.9,
        font_sz=1700,
    )
    add_omml_equation(
        s,
        r"\varepsilon_{\mathrm{AL}}=\varepsilon_{\mathrm{raw}}-\Delta(x)"
        r"\quad\Rightarrow\quad \mathbb{E}[\varepsilon_{\mathrm{AL}}]=0",
        left=0.5,
        top=2.05,
        width=9.0,
        height=0.9,
        font_sz=1600,
    )
    _textbox(
        s,
        0.5,
        3.1,
        9.0,
        0.4,
        "Change of variables for the density of AL noise (Jacobian = 1):",
        size=13,
        color=GRAY,
    )
    add_omml_equation(
        s,
        r"f_{\mathrm{AL}}(e)=f_{\mathrm{raw}}(e+\Delta),\qquad"
        r"e=y-\mu(x)",
        left=0.5,
        top=3.55,
        width=9.0,
        height=0.85,
        font_sz=1600,
    )
    add_omml_equation(
        s,
        r"p(y\mid x)=(1-\varepsilon_c)\,f_{\mathrm{AL}}\!\bigl(y-\mu(x)\bigr)"
        r"+\,\varepsilon_c\,\varphi\!\bigl(y;\mu(x),(\kappa\sigma(x))^{2}\bigr)",
        left=0.4,
        top=4.55,
        width=9.2,
        height=1.05,
        font_sz=1400,
    )
    _textbox(
        s,
        0.5,
        5.85,
        9.0,
        1.1,
        "With ε_c=0.12, κ=4: mostly skewed core, occasional wide Gaussian outliers. "
        "Target remains unimodal — MidlineSparse should not invent modes.",
        size=13,
        color=SLATE,
    )

    # ------------------------------------------------------------------
    # 11. KDE definition
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Local data KDE — mathematical definition")
    _textbox(
        s,
        0.5,
        0.9,
        9.0,
        0.45,
        "Pool train and test: {(x_i, y_i)}. Fix probe x★ and bandwidth h_x.",
        size=13,
        color=GRAY,
    )
    add_omml_equation(
        s,
        r"\mathcal{N}(x^{\star};h_x)=\{\,i:\;|x_i-x^{\star}|\le h_x\,\}",
        left=0.5,
        top=1.45,
        width=9.0,
        height=0.85,
        font_sz=1700,
    )
    add_omml_equation(
        s,
        r"\hat p_{\mathrm{KDE}}(y\mid x^{\star})"
        r"=\frac{1}{|\mathcal{N}|\,h_y}\sum_{i\in\mathcal{N}}"
        r"K\!\left(\frac{y-y_i}{h_y}\right)",
        left=0.5,
        top=2.5,
        width=9.0,
        height=1.1,
        font_sz=1600,
    )
    add_omml_equation(
        s,
        r"K(u)=\frac{1}{\sqrt{2\pi}}\,e^{-u^{2}/2}"
        r"\quad\text{(Gaussian kernel; SciPy gaussian\_kde)}",
        left=0.5,
        top=3.8,
        width=9.0,
        height=1.0,
        font_sz=1500,
    )
    _textbox(
        s,
        0.5,
        5.05,
        9.0,
        1.8,
        "SciPy’s gaussian_kde chooses the y-bandwidth h_y by Scott/Silverman rules "
        "from the local y-sample. We only control the x-neighborhood that defines N.",
        size=13,
        color=SLATE,
    )

    # ------------------------------------------------------------------
    # 12. Bandwidth schedule
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Local x-bandwidth schedule (implementation)")
    add_omml_equation(
        s,
        r"h_x^{(0)}=\max\!\bigl(0.08\,(x_{\max}-x_{\min}),\,10^{-3}\bigr)",
        left=0.5,
        top=1.0,
        width=9.0,
        height=0.9,
        font_sz=1700,
    )
    _bullet_block(
        s,
        0.5,
        2.1,
        9.0,
        4.5,
        [
            "Collect N = {i : |x_i − x★| ≤ h_x}.",
            "If |N| < 12, set h_x ← 1.6 h_x and retry (up to 6 widenings).",
            "If still |N| < 5 after widening, omit the KDE curve (too few points).",
            "Otherwise fit gaussian_kde({y_i : i∈N}) and evaluate on the y-grid.",
            "Probes are interior linspace points (endpoints excluded) so neighborhoods are not half-empty.",
        ],
        size=14,
    )

    # ------------------------------------------------------------------
    # 13. Reading plots
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "How to read a marginal panel")
    _try = _try_add_picture(
        s,
        FIG_DIR / "a01_marginal_x8p33.png",
        0.35,
        0.95,
        5.6,
    )
    if not _try:
        _card(s, 0.35, 0.95, 5.6, 5.8)
        _textbox(
            s,
            0.55,
            3.2,
            5.2,
            1.0,
            "(Run compare_foundation_baselines to populate\n"
            "a01_marginal_x8p33.png here.)",
            size=13,
            color=GRAY,
        )
    _bullet_block(
        s,
        6.15,
        1.0,
        3.5,
        5.8,
        [
            "Red = population truth.",
            "Gray dashed = finite-sample local shape.",
            "Colored = one model’s predictive density.",
            "A01 at large x: truth is bimodal; unimodal models fail the red curve.",
            "KDE may be flatter if few points fall in each mode locally.",
        ],
        size=13,
    )

    # ------------------------------------------------------------------
    # 14. Summary table
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Summary: what each curve answers")
    _bullet_block(
        s,
        0.5,
        1.05,
        9.0,
        4.8,
        [
            "True p(y|x) — no sample; uses DGP. Answers: population conditional law.",
            "Local KDE — uses data near x★; no DGP. Answers: what this sample supports locally.",
            "Model density — uses fit to data; not given the closed form. Answers: what the predictor believes.",
            "MidlineSparse / exact GP train on the same draws as the KDE, but never see the analytic density.",
            "Disagreement pattern: model≈KDE≠truth → sparse local sample; model≈truth≠KDE → model generalizes; model≠both → predictive failure.",
        ],
        size=15,
    )
    _textbox(
        s,
        0.5,
        6.1,
        9.0,
        0.8,
        "Code: plot_foundation_marginals.TRUE_PDF  and  empirical_local_pdf",
        size=12,
        color=GRAY,
    )

    # ------------------------------------------------------------------
    # 15. Takeaways
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title(s, "Takeaways")
    _bullet_block(
        s,
        0.5,
        1.1,
        9.0,
        5.5,
        [
            "True p(y|x) is the DGP’s closed-form conditional density — the scientific target.",
            "Local KDE is a bandwidthed empirical check that the sample carries that structure.",
            "A01/A04: mixture of Gaussians with known mode weights and σ.",
            "A05: derive AL scales for Var=σ², center to zero mean, then contaminate.",
            "KDE: Gaussian kernel on y-values with |x_i−x★|≤h_x, h_x grown until enough points.",
            "Use disagreements among (truth, KDE, model) to separate model error from data sparsity.",
        ],
        size=15,
    )

    prs.save(str(OUT))
    print(f"Wrote {OUT}")
    if not _pptx_is_readable(OUT):
        raise RuntimeError(f"Wrote unreadable PPTX before normalize: {OUT}")
    _powerpoint_normalize(OUT)
    if not _pptx_is_readable(OUT):
        raise RuntimeError(f"PPTX unreadable after normalize: {OUT}")
    print(f"Normalized {OUT}")
    return OUT


if __name__ == "__main__":
    build()

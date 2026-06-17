# Overleaf: RFF Woodbury derivation

## Upload to Overleaf

1. Create a new blank project on [Overleaf](https://www.overleaf.com).
2. Upload [`rff_woodbury_derivation.tex`](rff_woodbury_derivation.tex) (or zip this folder).
3. Set the main document to `rff_woodbury_derivation.tex`.
4. Compile with **pdfLaTeX**.

## Contents

- **Orange boxes**: differences between your \(\Sigma^{-1}\) formula and the GPPlus docstring (layout, symbol names).
- **Green boxes**: identical objects (same \(\Sigma\), same \(M\), same cost, same MLL log-det).
- **Section 5**: where \((\sigma^2 I)^{-1}\) goes (division / matvecs) vs.\ what is Cholesky'd (\(M\) only).
- **Sections 6–7**: step-by-step \(\Sigma^{-1}b\) and MLL split (\(n\log\sigma^2\) + \(\log|M\)).
- Full Woodbury proof sketch and link \(\Phi = Z^\top\).

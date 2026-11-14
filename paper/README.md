# Paper draft (Overleaf)

LaTeX draft of the follow-up paper on multi-object 6D pose estimation on BOP. Structure and
style follow `docs/PAPER_FEAT.md` (same skeleton as the WACV 2026 FastPose-ViT paper); every number
comes from `docs/BOP.md` / `TASKS.md` §9. Still-missing items are listed in `docs/PAPER_TODO.md`.

- `main.tex`: the paper (title, section includes). `supplementary.tex`: the supplementary
  material, a **separate document** (CVF conferences take it as its own PDF), with sections S1-S8
  that the paper cites as "Supp. Sec. Sx". Both share `preamble.tex` (packages, macros,
  notation) and `references.bib`. On Overleaf, compile each one by choosing it as the main
  document (Menu -> Main document), or keep two projects. In the PDF, orange **Pending** marks results
  that are still training (T-LESS, augmentation study, bbox jitter 5 / 15 / 30 %), red **TODO**
  marks text or facts to write or check.
- `sections/*.tex`: abstract, introduction, related work, method, experiments, conclusion.
- `references.bib`: entries marked `% CHECK` were written from memory and must be verified.

**Overleaf:** upload this folder (or zip it) as a new project; it compiles with pdfLaTeX + BibTeX.
For the official format, add the WACV author-kit files (`wacv.sty`, `ieee_fullname.bst`) next to
`main.tex`: the preamble switches to the WACV review format automatically when `wacv.sty` is
present, and falls back to a plain two-column article otherwise.

Figures are in `figures/`, made from the results by `vis/paper_*.py` (teaser diagram, qualitative
results, failure cases, latency plot, YCB-V depth-bias histogram).

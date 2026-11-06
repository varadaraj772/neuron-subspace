# JMLR submission checklist

Everything is prepared. The remaining steps are the portal actions.

## Files

- `main.pdf` — the submission PDF (13 pages, JMLR `jmlr2e` style).
- `source.zip` — LaTeX source: `main.tex`, `jmlr2e.sty`, `references.bib`,
  `tables/`, `figures/`.
- `cover_letter.txt` — draft cover letter (add the repository URL).
- `metadata.txt` — title, author, email, affiliation, keywords, abstract.

## Portal steps

1. Log in to the JMLR submission system.
2. Enter the title, abstract, keywords, and author details from `metadata.txt`.
3. Upload `main.pdf` as the manuscript.
4. Upload `source.zip` as the source archive.
5. Paste or upload `cover_letter.txt` if the form asks for a cover letter.
6. Confirm originality (not published, not under review elsewhere) and the
   competing-interests declaration.
7. Optional: suggest reviewers.
8. Submit.

## Before you submit

- The paper and cover letter now cite
  `https://github.com/varadaraj772/neuron-subspace`.
- **The repository currently returns 404 to unauthenticated requests.** Make it
  public, or provide a public mirror, before submitting; JMLR readers will click
  the link.
- **Push the latest changes.** The local repository has uncommitted work and the
  pushed commit predates the paper and the 4B results:
  `paper/`, `src/make_tables.py`, `src/make_figures.py`, `results/Qwen3.5-4B/`,
  `results_hybrid/Qwen3.5-4B/`, plus modified `.gitignore`, `results/summary_all.csv`,
  `run_all.sh`; deleted `src/make_report.py` and `FINAL_REPORT.md`.
- Confirm the header fields: `Journal of Machine Learning Research (2026) 1-13`,
  `Submitted 11/26`. The volume/published fields are intentionally blank; JMLR
  fills them after acceptance.
- `Editor: Action Editor` is a placeholder; JMLR assigns the action editor.
- The qualitative appendix is redacted. Keep it that way.
- `main.pdf` is a build artifact and is not tracked by git; `source.zip` is
  self-contained.

## Rebuild (if you edit the paper)

```bash
cd paper
tectonic -X compile main.tex
cp main.pdf submission/main.pdf
cd .. && zip -q -r paper/submission/source.zip paper/main.tex paper/jmlr2e.sty \
  paper/references.bib paper/tables paper/figures
```

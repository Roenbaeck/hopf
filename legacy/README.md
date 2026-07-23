# Superseded exploratory material

This directory preserves the experiments and reference drafts that preceded
the current paper.  They are retained for provenance only.

The scripts in `experiments/` explored several proposed curvature arguments.
Some use an obsolete curvature sign convention, some test formulas that were
subsequently disproved, and some report numerical patterns that are not used
in the final theorem.  They are intentionally excluded from the root build
and should not be cited as verification of `hopf_seam.tex`.

The text files in `reference_material/` belong to the earlier, broader
"seam geometry" direction.  The current paper is self-contained and does not
depend on them.

For the supported verification path, use `../verify_hessian_tori.py` via
`make verify` from the repository root.

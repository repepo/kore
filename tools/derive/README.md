# tools/derive

NEW (2026-10-29). Fitting and code generation for the magnetic couplings of `bin/operators.py`; the rows themselves are
derived by `tests/magnetic/derive.py` (sympy). Guide with worked examples: `docs/derivations.md`.

| script | what it does |
| --- | --- |
| `fit_rows.py KIND N [--rows u,v,f,g] [--l LMIN:LMAX] [--m MMIN:MMAX] [--jobs J] [--cache FILE]` | derives the rows of a background field (KIND pol or tor, degree N 1 or 2) for a range of l, m with derive.py and fits each coefficient as the field's angular factor times a(l) + b(l) m², exactly (rational coefficients); one line per (row, column, Δl, monomial) |
| `gen_table.py FIT_LOG` | turns that output into the table entries of an operator (as `operators.induction_pol2`, `lorentz_pol2`) in kore's labels, and lists the labels with their column parity for `submatrices.py` |

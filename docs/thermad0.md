# ThermaD = 0 with the thermal wind (thermal_wind = 1, 2)

`ThermaD` is kore's thermal-diffusion coefficient. With `ThermaD = 0` the heat equation has no diffusion: entropy
perturbations are only carried by the flow, so the perturbations are adiabatic. kore writes the heat equation in a
special, lower-order form in that case, and in that form the thermal-wind terms of a differential rotation
(`thermal_wind = 1, 2`) could not be represented, so kore used to stop. This change gives exactly this case its own
radial row weight and basis, and leaves every other configuration bitwise unchanged. All operator, assembly, budget,
MPI and regression checks pass, and as ThermaD → 0 the eigenvalues of diffusive runs converge linearly to the
ThermaD = 0 ones wherever the resolution allows; the remaining limitations are listed under [Caveats](#caveats).

Results of the change, 2026-10-12. The checks are in `tests/heat_dr/`; the analysis scripts named below (`track.py`,
`coupling.py`, `dense.py`) are not part of the repository.

## Background: how kore stores the heat equation

This section introduces the few code concepts the rest of the page relies on. Broader notation is in the
[notation](heat-equation.md#notation-and-code-names) section of the companion page.

kore solves the generalized eigenproblem $A x = \lambda B x$, with $\lambda = \sigma + i\omega$ ($\sigma$ the growth
rate, $\omega$ the frequency). The unknowns are the radial Chebyshev coefficients of the poloidal and toroidal flow
($P$, $T$) and of the entropy perturbation $s$, for each spherical-harmonic degree $l$. The matrix rows come in
sections: `u` and `v` for the two momentum equations, and **section h** for the heat (entropy) equation.

- **Row weight.** Before projection, each equation is multiplied by a power of $r$ (and, for the heat equation, by a
  power of $p_0 = \rho_0 T_0$). This makes every coefficient a polynomial in $r$, so the matrices are banded.
- **Gegenbauer basis.** The rows of a section are projected onto a Gegenbauer polynomial family $C^{(k)}$. The $k$-th
  radial derivative of a Chebyshev series is exactly a $C^{(k)}$ series, so a section that contains $k$-th derivatives
  needs at least a $C^{(k)}$ basis. $C^{(0)}$ is the Chebyshev basis itself. The order per section is stored in
  `gebasis` in `bin/submatrices.py`; `gebasis[2]` is section h.
- **bc rows and `chop`.** For each $l$, the top rows of a section's block are replaced by boundary-condition (bc) rows.
  `chop` is the number of rows freed for them. Without diffusion the heat equation has no thermal bc, so section h
  has no bc rows.
- **Operator labels.** Each radial operator is named by a short code string, decoded by `utils.decode_label`. For
  example `h1tw20_D1` means: section `h`; digit `1`, which lowers the row's power of $r$ by one (`rpower = base - rx`,
  where the base is the row weight); profile `tw2` differentiated `0` times; `_D1`, the first radial derivative of the
  unknown. The same label therefore means a different operator when the row weight of its section changes.
- **`remroco`.** In the full sphere (`ricb = 0`, no inner core) the code removes rows and columns of the wrong radial
  parity. This requires each operator to have a definite parity in $r$.

The heat-equation rows exist in three variants. Here (eq) is the $p_0$-weighted heat equation of
[heat-equation.md](heat-equation.md) §0, and the weight is applied to it:

| case | row weight on (eq) | basis of section h | thermal bc rows |
| --- | --- | --- | --- |
| ThermaD > 0 | $r^2$ | $C^{(2)}$ | yes |
| ThermaD = 0, old ("old rows") | $r/p_0$ | $C^{(0)}$ | none |
| ThermaD = 0 with the thermal wind ("new rows", this change) | $r^2/p_0$ | $C^{(1)}$ | none |

The old rows are still used with ThermaD = 0 when `thermal_wind = 0` or `diff_rot = 0`.

**Thermal wind.** With differential rotation $\Omega = \Omega_0(1 + \Delta\Omega)$, the background entropy gains a
latitudinal part $S_{tw}$ (the thermal-wind entropy) fixed by thermal-wind balance. It is expanded in Legendre
polynomials, with coefficients $\sigma_n$:

$$
S_{tw}(r,\theta) = \sum_{n = 2, 4, 6} \sigma_n(r)\,P_n(\cos\theta).
$$

The parameter `thermal_wind` selects it: 0 none, 1 the linear thermal wind, 2 the exact one (adds $\sigma_6$),
3 a prescribed $\partial_\theta S_{tw}$. `Gaspard` ($G$) is the frame rotation rate in units of the time unit. The
differential rotation adds three kinds of terms to the heat equation ([heat-equation.md](heat-equation.md) §0–2):

$$
p_0\left[\, i\,m\,G\,\Delta\Omega\, s \;+\; u_r\,\frac{\partial S_{tw}}{\partial r} \;+\; \frac{u_\theta}{r}\,\frac{\partial S_{tw}}{\partial\theta} \right].
$$

- (a) is the advection of $s$ by the differential rotation. It is present whenever `diff_rot = 1`.
- (b) is the radial advection of $S_{tw}$.
- (c) and (c′) are the latitudinal advection of $S_{tw}$ by the poloidal ($P$) and toroidal ($T$) parts of $u_\theta$.

(b), (c) and (c′) exist only with `thermal_wind ≥ 1`. In kore's profile names, `tw2`, `tw4`, `tw6` hold $\sigma_n$ and
`tp2`, `tp4`, `tp6` hold $\sigma_n'$, each times $p_0$ except in the new ThermaD = 0 mode below.

## Why ThermaD = 0 needed a separate treatment

This section explains why the old ThermaD = 0 rows cannot hold the thermal-wind terms, and why the fix is to change the
weight and the basis rather than drop the $1/p_0$.

With `ThermaD = 0` kore writes the heat equation divided by $p_0$, because $p_0 \approx 10^{-7}$ near the surface made
the $p_0$-weighted rows ill-conditioned. It also multiplies it by $r$, projects it in a $C^{(0)}$ Gegenbauer basis, and
uses no thermal bc rows: `decode_label` gives section h `rpower = 1 - rx`, and `submatrices.py` sets `gebasis[2] = 0`.

Term (c) of [heat-equation.md](heat-equation.md) §2, the advection of the thermal-wind entropy by $u_\theta$, is,
before the row weight,

$$
p_0\,\sigma_n\,\frac{P + r P' + r\,(\ln\rho)'\,P}{r^2}\;P_n'(X)\,D ,
$$

where $X$ and $D$ are the angular coupling matrices of heat-equation.md §2. Two things break:

1. **Basis.** Its $P'$ part (`h1tw20_D1`, `h1tw40_D1`, `h1tw60_D1`) is a first derivative, which lands in $C^{(1)}$.
   In a $C^{(0)}$ section, `submatrices.py` builds `G[gbx][gbx-dx] = G[0][-1]`. Python's negative index returns
   `G0[-1] = 1` (the identity) instead of failing. The D1 output, a $C^{(1)}$ series, would then be read as Chebyshev
   coefficients. Silently wrong.
2. **Row weight.** With the $r$ weight, the $P$ part (`h2tw20_D0`) decodes to `rpower = 1 - 2 = -1`, a profile
   $\sigma_n/r$. In the full sphere $\sigma_2(0) \neq 0$ for the conical, Y20 and solar profiles ($\sigma_2 \propto r f/g$
   and $g \propto r$). So $\sigma_n/r$ is singular at $r = 0$ and has the wrong parity for `remroco`. At ricb > 0 it
   would be regular, but item 1 remains.

The old code stopped in `submatrices.py` before reaching either problem. Dropping the $1/p_0$ would fix item 2 but bring
back the conditioning problem. The fix therefore keeps the $1/p_0$ and, for this case only, changes the row weight from
$r$ to $r^2$ and the basis from $C^{(0)}$ to $C^{(1)}$.

## What changed, file by file

This section lists every file touched, so that a reader can find where the new mode is decided and which outputs it
affects.

The new mode is decided in one place, `bin/utils.py`:

```python
adiab_tw = thermal == 1 and ThermaD == 0 and diff_rot == 1 and thermal_wind >= 1
```

In this mode the section-h rows are $r^2\,(\mathrm{eq})/p_0$ in $C^{(1)}$, still without bc rows, and they reuse the
labels of both paths.

- `bin/utils.py`: `adiab_tw` (the only place that decides the mode). `decode_label` gives section h `rpower = 2 - rx`
  when `adiab_tw`. So:
    - `h0_D0` is $r^2 s$;
    - `h1dSd0_D0` is $r^2 u_r S'$ (as $r L P S'$);
    - `h0abu0_D0`/`h0aub0_D0` is the $s$ advection with the $r^2$ weight;
    - the thermal-wind labels of the ThermaD > 0 path (`h1tp20_D0`, `h2tw20_D0`, `h1tw20_D1`, `h1tw20lho1_D0`,
      `h1tw20_D0`, and the 4, 6 ones) decode as they do there.
- `bin/submatrices.py`:
    - `gebasis[2] = 1 if ut.adiab_tw else 0`;
    - the `sys.exit` for ThermaD = 0 with the thermal wind is removed;
    - `G[gbx][gbx-dx]` raises a `ValueError` (label, dx, basis) when dx > gbx;
    - `chop = 0` for section h when ThermaD = 0 (at ricb > 0, `chop = gbx = 1` would otherwise put one zero row on top
      of every section-h block);
    - comments.
- `bin/radial_profiles.py`: `ptw(r)` returns $p_0$, or 1 when `adiab_tw`. `tw2/4/6` and `tp2/4/6` use it, so in the new
  mode they are $\sigma_n$ and $\sigma_n'$ (bitwise the same as before when ThermaD > 0).
- `bin/operators.py`: comments only. `thermal_diffrot` already used the same (b), (c), (c′) labels for both paths, and
  (a) uses `h0abu0_D0`/`h0aub0_D0` whenever ThermaD = 0.
- `bin/params_default.py`: `thermal_wind` comment: ThermaD > 0 is no longer required.
- `bin/assemble.py`: unchanged. `bc_theta_spherical` returns nothing for ThermaD = 0, and section h is packed from the
  operators.
- `bin/ru_insane.py`, `tests/heat_dr/ebal.py`: `gb['h'] = 1` in the new mode; `chop['h'] = 0` for every ThermaD = 0
  run; entropy pairing weight `wgt['h'] = p₀` (rows $r^2\,(\mathrm{eq})/p_0$) instead of $p_0 r$. The pairing weight
  is what the checks multiply the rows by when they pair them with $s^*$ to form the entropy budget. `heat_check.py`,
  `asm_check.py`, `terms.py` and `sd_check.py` take these from `ebal.py`.
- `tests/heat_dr/asm_check.py`: docstring. In the new mode the reference run is the same run with
  `diff_rot_amplitude = 0`, because `diff_rot = 0` or `thermal_wind = 0` gives the old $r\,(\mathrm{eq})/p_0$ rows in
  $C^{(0)}$. A wrong reference trips the existing assertion, which now hints at that.
- `tests/heat_dr/run_all.sh`:
    - new-mode runs `k1 km0 km2b ky20wb kshell kg3 ksun ksun_a` with `thermal_wind` 1 and 2, and their `z_*`
      references (amplitude 0), in the heat_check and asm_check lists;
    - rigid configurations `K1` ($G = 1$) and `K13` ($G = 3$), whose baseline is rigid with amplitude 0 (the same
      heat rows);
    - `terms.py` and `sd_check.py` on `k1_tw2`.
- `tests/heat_dr/README.md`, `docs/heat-equation.md`: the asm_check reference; §0 row weights, §3 the implementation,
  §5 the submatrices row, §7 the open item updated.

Deviations from the proposed design:

- `chop = 0` for section h (one line in `submatrices.py`). It was needed: at ricb > 0, $C^{(1)}$ would otherwise
  reserve one zero row per $l$ for a bc that does not exist. Mutation M4 below shows what that does.
- asm_check and the rigid test need a reference with the same heat rows, so they use the `diff_rot_amplitude = 0` run
  instead of the `diff_rot = 0` one. No code beyond comments and the run_all lines.

Bitwise constraint: ThermaD > 0 runs, and ThermaD = 0 runs with `thermal_wind = 0` or `diff_rot = 0`, never enter the
new branches (`adiab_tw` is False; `ptw` returns `prf.pressure(r, 0)` as before; `chop` was already 0 for gbx = 0).
Confirmed in test f.

## Results

This section collects the tests: operator-level checks (a, b), eigenvalue identities and limits (c, d), energy
budgets (e), regression and MPI (f, g), and independent reviews (h).

Runs are the test defaults of `mk_run.py` plus `par.ThermaD = 0`, unless stated. The defaults are: ricb 0.35, m 1,
conical differential rotation with amplitude −0.25, Beyonce 10 (buoyancy factor), ViscosD 1e-3, stress-free walls,
'user def' profiles. "tw2" means `thermal_wind = 2`. `SUN` is the run_all.sh string (ricb 0, smoothed sun, N 120,
lmax 30) with ThermaD = 0. N is the number of radial Chebyshev modes, lmax the largest degree.

### a. Operator and assembly checks: heat_check.py and asm_check.py (pass < 1e-8)

These compare the code's matrices with an independent computation of the same physics.

- `heat_check.py` compares the section-h operator blocks with an $(r, \theta)$ quadrature of the heat terms for random
  fields.
- `asm_check.py` compares the heat rows of the assembled $A$ minus those of a reference run with the same quadrature.
  Here the reference is `z_*` (the same run, amplitude 0).

| run | heat_check tw1 | heat_check tw2 | asm_check tw1 | asm_check tw2 |
| --- | --- | --- | --- | --- |
| k1: conical m 1, N 96 | 3.0e-15 | 2.6e-15 | 4.5e-15 | 3.0e-15 |
| km0: m 0, lmax 21 | 1.1e-14 | 9.6e-15 | 9.1e-15 | 8.8e-15 |
| km2b: m 2, symm −1, lmax 21 | 3.3e-15 | 3.1e-15 | 3.4e-15 | 3.7e-15 |
| ky20wb: Y20-wall-bounded | 7.9e-15 | 7.9e-15 | 1.7e-14 | 1.9e-14 |
| kshell: shellular | 2.8e-15 | 2.3e-15 | 2.9e-15 | 1.9e-15 |
| kg3: Gaspard 3, Beyonce 90 | 2.6e-15 | 2.2e-15 | 4.0e-15 | 2.6e-15 |
| ksun: SUN, ricb 0, symm 1 | 5.5e-15 | 6.3e-15 | 9.1e-13 | 5.7e-13 |
| ksun_a: SUN, symm −1 | 5.9e-15 | 7.4e-15 | 2.3e-12 | 8.7e-13 |

All 32 are ok, and run_all.sh reproduces them.

**Limitation: these checks do not see the rows near the wall.** The numbers above are $p_0$-weighted: the rows
$r^2\,(\mathrm{eq})/p_0$ are paired with $p_0 s^*$, and $p_0 \approx 2.5\times10^{-6}$ at $r = 1$ for the test
profiles. Near the wall the new rows are less accurate than the ThermaD > 0 ones. Term (c) carries $\sigma_n (\ln\rho)'$
without $p_0$. For the 'user def' density,

$$
\rho \propto \left(1 - \mathrm{aux0}^2 r^2\right)^2,
\qquad
(\ln\rho)' = -\frac{4\,\mathrm{aux0}^2\, r}{1 - \mathrm{aux0}^2 r^2},
$$

which is nearly singular at $r = 1/\mathrm{aux0}$. Checked unweighted (`tests/heat_dr/row_check.py`, $P = 0$ at the
walls), the rows are right to:

- 4e-11 at ricb 0.35 (N 96), and 6e-11 for the smoothed sun (N 120);
- only 5e-8 – 3e-7 for the 'user def' profiles at ricb 0 (N 96; 3e-11 at N 160);
- 3e-3 with aux0 = 0.999 (N 96), where heat_check still reports 5e-15.

In the eigenvalues it is not the limiting error in the runs tried (full sphere, user def). There the eigenvalue change
between N 96 and N 240, $\lvert\Delta\lambda\rvert$, is 7e-5 to 5.5e-4 with `thermal_wind` 1, against 5e-5 to 3.6e-4
(N 96 vs 128) with `thermal_wind` 0; at aux0 = 0.999 both are 5e-2 to 1e-1 off at N 96. But the $p_0$-weighted checks
are blind to it.

A has no zero row and B no zero section-h row (k1: 0 of 2880, and 60 zero B rows, all momentum bc rows).

The adapted tools on the unchanged modes: heat_check s96 3.3e-13, k0 1.4e-15, sun 3.4e-11, g3_tw2 2.9e-13; asm_check
k0 1.4e-15; ru_insane PASS on all four.

### b. Block-by-block checks: ru_insane.py (N ≥ 96, tol 1e-6): all 16 runs PASS

`ru_insane.py` tests each matrix block against a direct evaluation on an $(r, \theta)$ grid, and checks that the
advection term (a) is anti-Hermitian, as it must be (it only shifts frequencies). The "worst" column is the largest
error over all its tests.

| run | heat advection vs grid | thermal wind vs grid | advection anti-Herm. | worst (all tests) |
| --- | --- | --- | --- | --- |
| k1, k1_tw2 | 2.6e-15 | 1.2e-14, 1.1e-14 | 2.0e-16 | 4.7e-08 |
| km0, km0_tw2 | 0 | 1.0e-14, 9.7e-15 | n/a (m 0) | 4.8e-09 |
| km2b, km2b_tw2 | 6.8e-15 | 1.1e-14, 1.1e-14 | 2.0e-16 | 3.7e-08 |
| ky20wb, ky20wb_tw2 | 1.5e-14 | 1.2e-14, 1.2e-14 | 1.2e-16 | 4.7e-08 |
| kshell, kshell_tw2 | 1.4e-14 | 6.0e-15, 4.9e-15 | 9.0e-17 | 4.7e-08 |
| kg3, kg3_tw2 | 2.7e-15 | 1.2e-14, 1.1e-14 | 2.2e-16 | 4.7e-08 |
| ksun, ksun_tw2 | 3.1e-15 | 1.7e-14, 1.1e-14 | 6.2e-16 | 6.9e-08 |
| ksun_a, ksun_a_tw2 | 8.0e-15 | 2.3e-14, 2.6e-14 | 8.0e-16 | 5.8e-08 |

The worst column is the viscous Hermitian test, which is the same with ThermaD > 0. The entropy rows match direct
quadratures:

- entropy block Hermitian ≤ 1.4e-15;
- buoyancy, advection and entropy rows vs quadrature ≤ 4e-15 at ricb 0.35 (buoyancy 2e-8 and 6e-8 for the sun, as
  with ThermaD > 0).

**Mutations.** To show the checks would catch the relevant mistakes, each mutation was applied to a copy of the code:

| mutation | heat_check | ru_insane |
| --- | --- | --- |
| M1: keep $p_0$ in tw/tp | 2.8e-2 FAIL | thermal wind 0.23, FAIL |
| M3: `gebasis[2] = 0` (old basis) | `submatrices.py` raises `ValueError` (below) | n/a |
| M4: no `chop = 0` (zero row per $l$ at ricb > 0) | 0.18 FAIL | entropy Hermitian 1.4, advection 2.4, FAIL |

The M3 message:

```
ValueError: h1tw20_D1: derivative order 1 needs at least the C^(1) basis, section h has C^(0)
```

### c. Rigid-offset identity (rigid_thermal.py, m 2, N 64, lmax 25, 16 modes, ThermaD 0, thermal_wind 1)

A rigid offset of the rotation rate must be equivalent to a rigid rotation at a different frame rate. This gives an
exact identity for the eigenvalues ([heat-equation.md](heat-equation.md) §5):

$$
\lambda(\Delta\Omega, G) = \lambda_0\big(G\,(1+\Delta\Omega)\big) - i\,m G\,\Delta\Omega .
$$

The table gives the error of this identity for two choices of the baseline $\lambda_0$. The first baseline is a rigid
run with amplitude 0 and frame rate Gaspard $G(1 + 0.1)$: it has the same heat rows as the tested run. The second is
`diff_rot = 0`, which uses the old $r\,(\mathrm{eq})/p_0$ rows in $C^{(0)}$: a different discretization of the same
problem.

| baseline | $G = 1$ | $G = 3$ |
| --- | --- | --- |
| rigid, amplitude 0, Gaspard $G(1 + 0.1)$ (run_all `K1`, `K13`) | 1.4e-9 (1.3e-9 in the run_all pieces) | 8.4e-12 (8.2e-12) |
| `diff_rot = 0` (old rows) | 8.0e-7 (N 64), 6.6e-9 (N 96), 7.8e-9 (N 128) | 2.5e-7 (N 64) |

- For a rigid offset $S_{tw} = 0$, so this tests the new rows and basis with term (a), not the thermal wind.
- The 1.4e-9 at $G = 1$ is set by the most damped modes ($\sigma \approx -0.97$), where csolve's residuals are ~7e-9
  (`csolve.py` is the test eigensolver).
- The second row is the old and new discretizations converging to each other with N.

### d. ThermaD → 0 continuity

This test asks whether the ThermaD = 0 eigenvalues are the limit of the diffusive ones. Diffusive runs use the
ThermaD > 0 path (with thermal bc rows), so the comparison is between two independent discretizations.

**The modes.** Five lightly damped, thermally coupled modes T1–T5 are taken from k1 at N 128 (tw1). `coupling.py`
measures how strongly each mode feels the thermal physics: its share of buoyancy in the momentum balance, and of the
diff-rot heat terms in the entropy balance,

$$
\text{buoyancy share} = \frac{\lvert B\rvert}{\lvert\lambda\rvert K},
\qquad
\text{diff-rot heat share} = \frac{\lvert H\rvert}{\lvert\lambda\rvert E},
$$

with $B$ the buoyancy term and $K$ the kinetic energy of the momentum balance, and $H$ the diff-rot heat terms and
$E$ the entropy "energy" of the entropy balance. The last column is the part of $H$ due to the thermal wind.

| mode | $\lambda$ (ThermaD = 0, N 128) | buoyancy share | diff-rot heat share | of which $u\cdot\nabla S_{tw}$ |
| --- | --- | --- | --- | --- |
| T1 | −0.01684560+0.49088938i | 0.045 | 0.49 | 0.10 |
| T2 | −0.07611871+0.44729603i | 0.11 | 0.37 | 0.066 |
| T3 | −0.12446846+0.39326248i | 0.22 | 0.37 | 0.051 |
| T4 | −0.11370717+1.76151161i | 0.29 | 0.12 | 0.0062 |
| T5 | −0.04118984+2.54022727i | 0.55 | 0.075 | 0.0041 |

$\lambda(\mathrm{ThermaD} = 0)$ changes by 6e-9 to 4e-8 between N 96 and 128 (tw1 and tw2), so it is converged in N.

**The distances.** Each entry in the three tables below is the distance

$$
\big\lvert\lambda(\mathrm{ThermaD}) - \lambda(\mathrm{ThermaD} = 0)\big\rvert ,
$$

computed by shift-invert (an iterative eigensolver targeted at each mode) with `track.py`; residuals ≤ 3e-11. The
column headings 1e-4 … 1e-10 are the values of ThermaD.

Fixed entropy at the walls (`bco_thermal = bci_thermal = 0`, as mk_run sets them), stress-free, `thermal_wind` 1:

| mode | N | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 96 | 4.9e-04 | 5.4e-05 | 5.4e-06 | 2.6e-07 | 8.8e-06 | 9.9e-06 |
| T1 | 128 | 4.9e-04 | 5.4e-05 | 5.4e-06 | 5.5e-07 | 2.4e-06 | 6.0e-06 |
| T1 | 192 | | | | | 5.7e-08 | |
| T1 | 256 | | | | | 5.4e-08 | 1.7e-06 |
| T2 | 96 | 1.3e-03 | 1.5e-04 | 1.5e-05 | 7.0e-06 | 3.2e-05 | 4.6e-05 |
| T2 | 128 | 1.3e-03 | 1.5e-04 | 1.5e-05 | 1.4e-06 | 1.1e-05 | 2.8e-05 |
| T3 | 96 | 3.4e-03 | 3.7e-04 | 3.6e-05 | 1.1e-05 | 1.1e-04 | 9.9e-05 |
| T3 | 128 | 3.4e-03 | 3.7e-04 | 3.6e-05 | 3.9e-06 | 2.1e-05 | 6.1e-05 |
| T4 | 96 | 4.2e-03 | 4.2e-04 | 4.2e-05 | 1.3e-05 | 1.2e-05 | 1.2e-05 |
| T4 | 128 | 4.2e-03 | 4.2e-04 | 4.2e-05 | 3.3e-06 | 7.5e-06 | 7.3e-06 |
| T5 | 96 | 1.9e-03 | 1.9e-04 | 1.9e-05 | 1.0e-05 | 1.0e-05 | 1.0e-05 |
| T5 | 128 | 1.9e-03 | 1.9e-04 | 1.9e-05 | 1.9e-06 | 6.2e-06 | 6.1e-06 |
| T5 | 192 | | | | | 2.2e-06 | |
| T5 | 256 | | | | | 1.6e-07 | 1.7e-06 |

The same with `thermal_wind` 2. Its ThermaD = 0 eigenvalues $\lambda(0)$ are:

- T1 −0.01701638+0.49013119i
- T2 −0.07696123+0.44671144i
- T3 −0.12435151+0.39357407i
- T4 −0.11360552+1.76197107i
- T5 −0.04089672+2.54083851i

| mode | N | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 96 / 128 | 4.9e-04 / 4.9e-04 | 5.4e-05 / 5.4e-05 | 5.5e-06 / 5.5e-06 | 2.4e-07 / 5.5e-07 | 6.6e-06 / 1.8e-06 | 7.6e-06 / 4.6e-06 |
| T2 | 96 / 128 | 1.3e-03 / 1.3e-03 | 1.5e-04 / 1.5e-04 | 1.5e-05 / 1.5e-05 | 6.4e-06 / 1.4e-06 | 2.1e-05 / 9.0e-06 | 3.9e-05 / 2.3e-05 |
| T3 | 96 / 128 | 3.5e-03 / 3.5e-03 | 3.7e-04 / 3.7e-04 | 3.7e-05 / 3.7e-05 | 1.0e-05 / 3.9e-06 | 9.4e-05 / 2.0e-05 | 8.8e-05 / 5.4e-05 |
| T4 | 96 / 128 | 4.1e-03 / 4.1e-03 | 4.2e-04 / 4.2e-04 | 4.2e-05 / 4.2e-05 | 1.2e-05 / 3.3e-06 | 1.0e-05 / 6.3e-06 | 9.8e-06 / 6.0e-06 |
| T5 | 96 / 128 | 1.9e-03 / 1.9e-03 | 1.9e-04 / 1.9e-04 | 1.9e-05 / 1.9e-05 | 8.7e-06 / 1.7e-06 | 8.4e-06 / 5.1e-06 | 8.3e-06 / 5.0e-06 |

Zero entropy flux at the walls (`bci_thermal = bco_thermal = 1`), stress-free, `thermal_wind` 1. At ThermaD = 0 the
thermal bc parameters do not enter: A is byte-identical to the fixed-entropy run.

| mode | N | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 96 / 128 | 5.6e-04 / 5.6e-04 | 6.0e-05 / 6.0e-05 | 6.0e-06 / 6.0e-06 | 6.0e-07 / 6.0e-07 | 6.3e-08 / 6.1e-08 | 4.2e-09 / 1.7e-09 |
| T2 | 96 / 128 | 1.6e-03 / 1.6e-03 | 1.9e-04 / 1.9e-04 | 1.8e-05 / 1.8e-05 | 1.8e-06 / 1.8e-06 | 2.1e-07 / 1.9e-07 | 2.5e-08 / 9.1e-09 |
| T3 | 96 / 128 | 3.3e-03 / 3.3e-03 | 3.7e-04 / 3.7e-04 | 3.6e-05 / 3.6e-05 | 3.6e-06 / 3.6e-06 | 3.6e-07 / 3.8e-07 | 1.8e-08 / 5.1e-09 |
| T4 | 96 / 128 | 4.1e-03 / 4.1e-03 | 4.2e-04 / 4.2e-04 | 4.2e-05 / 4.2e-05 | 4.2e-06 / 4.2e-06 | 4.2e-07 / 4.2e-07 | 5.9e-09 / 4.8e-09 |
| T5 | 96 / 128 | 1.9e-03 / 1.9e-03 | 1.9e-04 / 1.9e-04 | 1.9e-05 / 1.9e-05 | 1.9e-06 / 1.9e-06 | 1.9e-07 / 1.9e-07 | 2.7e-09 / 1.9e-09 |

Controls at N 96, fixed entropy, distances at ThermaD 1e-6, 1e-8, 1e-10:

- no-slip walls (bci = bco = 1, `thermal_wind` 1): T1 3.7e-6, 3.7e-8, 4.1e-10; T4 4.2e-5, 4.2e-7, 4.2e-9;
  T5 2.1e-5, 2.1e-7, 2.1e-9;
- `thermal_wind` 0 (old rows, unchanged code): T1 5.5e-6, 5.5e-8, 5.8e-10; T4 4.1e-5, 4.1e-7, 4.1e-9;
  T5 1.8e-5, 1.8e-7, 1.8e-9.

**Reading.** When N resolves the ThermaD > 0 run, the eigenvalues converge to the ThermaD = 0 ones linearly in ThermaD,
with the same slope for both bcs:

$$
\lambda(\mathrm{ThermaD}) - \lambda(0) \approx c\,\mathrm{ThermaD}, \qquad c = 5 \text{ to } 40 .
$$

That holds at every ThermaD with zero-flux or no-slip walls, and down to 1e-7 (N 128) and 1e-8 (N 256) with fixed
entropy and stress-free walls.

The exception is the fixed-entropy, stress-free case at small ThermaD. There the ThermaD = 0 entropy is not zero at
the wall: $u_\theta \neq 0$ there, and so is the source $u_\theta\,\partial_\theta S_{tw}/r$. A ThermaD > 0 run
must bring $s$ to zero at the wall, so it has a boundary layer of width

$$
\delta \sim \left(\frac{\mathrm{ThermaD}}{\lvert\lambda\rvert}\right)^{1/2},
$$

about 1e-4 at 1e-8 and 1e-5 at 1e-10.

- Below the resolved range its eigenvalue stalls 1e-6 to 1e-4 away, and the gap falls with N (T1 at 1e-10: 9.9e-6,
  6.0e-6, 1.7e-6 at N 96, 128, 256; T5 at 1e-8: 1.0e-5, 6.2e-6, 2.2e-6, 1.6e-7 at N 96, 128, 192, 256).
- The ThermaD = 0 runs have no layer and are converged at N 96.
- `thermal_wind = 0` has no such layer: $u_r = 0$ and $S' = 0$ at the walls make $s = 0$ there anyway.

**Box counts.** These check that no spurious eigenvalues appear near the tracked modes. Complete spectra at N 96 were
computed by dense QZ (the direct generalized eigensolver) after Ruiz equilibration (a row/column rescaling of $A$ and
$B$). They are accurate to ~5e-7 against shift-invert; the unequilibrated QZ was off by up to 1e-1 and gave fake
eigenvalues with $\sigma$ up to 49. The numbers are eigenvalues within ±0.05, ±0.1 and ±0.2 in $\sigma$ and $\omega$ of
each mode, written ThermaD 0 / ThermaD 1e-10:

| mode | tw1 | tw2 |
| --- | --- | --- |
| T1 | 1/1, 2/2, 4/4 | 1/1, 2/2, 4/4 |
| T2 | 1/1, 3/3, 74/72 | 1/1, 3/3, 76/75 |
| T3 | 1/1, 3/3, 266/259 | 1/1, 3/3, 264/258 |
| T4 | 1/1, 1/1, 1/1 | 1/1, 1/1, 1/1 |
| T5 | 1/1, 2/2, 2/2 | 1/1, 2/2, 2/2 |

**Whole spectrum**, N 96 tw1 (tw2). "The band" is a box around $\sigma = 0$ that holds the discretized continuous
spectrum of term (a) (see [Caveats](#caveats)):

$$
\text{band:}\quad \lvert\sigma\rvert < 0.03, \qquad \omega \in (-0.01,\ 0.26).
$$

| | ThermaD 0 | ThermaD 1e-10 |
| --- | --- | --- |
| finite eigenvalues | 2820 (2820) | 2800 (2800) |
| in the band | 790 (791) | 770 (771) |
| outside the band with $\sigma > -0.3$ | 130 (129) | 130 (129) |
| growing ($\sigma > 10^{-3}$, all inside the band) | 22 (19) | 22 (19) |

- The 20 extra ThermaD = 0 eigenvalues are the 20 $s$ coefficients that the ThermaD > 0 run spends on bc rows, and
  they all fall in the band.
- Outside the band the 130 (129) modes pair one to one: median distance 4.7e-6 (3.8e-6), maximum 1.2e-4 (1.0e-4), the
  latter for modes close to the band.
- Only the ±0.2 boxes of T2 and T3 reach the band, which is where their counts differ.
- No spurious mode outside the band.

### e. Balances

These check that the eigenvectors satisfy the energy and entropy budgets, with the new terms included.

- `sd_check.py` compares the powers computed by `spin_doctor` (kore's post-processing diagnostics) with the matrix
  pairings: dWdr for the flow, dWdr_thm for the entropy.
- resid1, resid2 (flow) and resid3 (entropy) are `spin_doctor`'s budget residuals.
- `terms.py` checks the entropy balance built from the matrix rows, whose residual is

$$
\mathrm{res} = \lambda E - A - D - H .
$$


Results:

- k1 (tw1, N 96, 8 modes near −0.05 + 1.2i):
    - sd_check `worst dWdr, dWdr_thm 1.9e-10 -> ok`;
    - resid3 with the new terms 3e-13 to 7e-15 (without them 7e-1 to 5e-3), resid1 ≤ 5.8e-7, resid2 ≤ 3.4e-8;
    - eig2kore + spin_doctor: $\mathrm{resid}_s$ ≤ 2.9e-13, $\mathrm{resid}_\sigma$ ≤ 5.8e-7;
    - terms.py entropy balance: res ≤ 3.6e-14.
- k1_tw2 (the same, and the run_all line): sd_check 1.9e-10 ok; terms.py res ≤ 1.0e-13; resid3 ≤ 2.8e-12.
- ksun_tw2 (ricb 0, N 120, 12 modes near −0.05 + 0.8i): terms.py res ≤ 1.7e-12, but sd_check FAIL (5.2).
    - Modes 0–4 and 7 ($\omega \approx 0.2494$–$0.2520$) sit at the edge of the advection continuum
      ($-i\,m G\,\Delta\Omega$, $\omega \in [0, 0.25]$) and are not resolved at any N.
    - The discrete modes near 0.51i are under-resolved at N 120: resid3 3–6e-2, dWdr_thm 2.7e-4 to 1.4e-3. At the
      same N the ThermaD = 1e-10 run gives dWdr_thm 4e-4 to 2e-3 and `thermal_wind` 0 gives 2e-5 to 3.5e-4.
    - At N 200 the new mode gives resid3 1.6e-8 to 2.7e-7 and dWdr_thm 1.9e-10 to 3.4e-8 for those modes
      (λ = −0.023069 + 0.515537i; ThermaD = 1e-10 at N 120: −0.023070 + 0.515536i).
    - So run_all's sd_check line uses k1_tw2.

### f. Bitwise regression against the code before this change (each built with its own mk_run.py)

This confirms that configurations outside the new mode produce exactly the same matrices as before.

A.npz and B.npz are byte-identical (`cmp`) for:

- s96, k0, sun, sun_a_tw2, g3_tw2, m0_tw2;
- sunk0_tw0 (SUN, ThermaD 0, `thermal_wind` 0);
- nodr_k0, nodr_sunk0 (SUN, ThermaD 0, `diff_rot` 0);
- bc1 (thermal bc 1);
- f10 and f10k (forcing 10, ThermaD 1e-3 and 0, `thermal_wind` 0; A and B_forced).

### g. MPI (mpiexec -n 1, 2, 3, 4)

This confirms that the parallel assembly gives the same matrices as a single rank.

A.npz and B.npz are byte-identical to the single-rank files, with the same pattern and 0 difference, for:

- mpi_k1 (lmax 24, tw1);
- mpi_ksun (smoothed sun, ricb 0, N 64, lmax 24, tw2);
- mpi_km0 (m 0, lmax 23, tw2).

### h. Independent reviews

The reviews covered the discretisation and the numerics.

**Discretisation: no defect.**

- B matches only the claimed combination ($C^{(1)}$, $r^2$, first row) to 1e-15; every other basis, row or weight is
  off by ≥ 0.19.
- 19 other configurations are byte-identical to the old code (every .mtx), MPI with 2 ranks too.
- Its one finding, that the $p_0$-weighted checks cannot see the rows near the wall, gave `row_check.py` (a).

**Numerics.**

- With $S_{tw} = 0$ the new rows reproduce the old-row spectrum (conical N 96: ≤ 4e-8; Y20-wall-bounded: 69
  eigenvalues, median 1e-9, max 5e-7).
- The smoothed sun with `thermal_wind` 2 converges in N: for the discrete modes, $\lambda$ changes by about $10^{-8}$
  to $2\times10^{-8}$ between N 160 and N 200.
- ThermaD = 1e-10 at N 200 is within 8e-9 of ThermaD = 0 there.
- The growing modes of the caveats (see [Caveats](#caveats)).

## Caveats

These are the limitations a user of ThermaD = 0 with the thermal wind should know about.

**Singular limit.** With fixed-entropy, stress-free walls, ThermaD → 0 at fixed N is a singular limit (d). The
ThermaD = 0 run is the well-resolved one. A "tiny ThermaD instead of 0" run, as the params comment suggests for the old
ThermaD = 0, is not equivalent below ThermaD ~ 1e-7 at N 128.

**Continuous spectrum and small-scale growing modes.** Term (a) without diffusion has a continuous spectrum,

$$
\lambda = -i\,m G\,\Delta\Omega(r, \theta).
$$

ThermaD = 0 with `thermal_wind = 0` has it too. The discretization shows it as hundreds of eigenvalues near
$\sigma = 0$. With the thermal wind some of them grow:

- 22 (19 with `thermal_wind` 2) with $\sigma \leq 0.011$ for k1 (N 96, lmax 20);
- 6 with $\sigma \leq 0.013$ for Y20-wall-bounded;
- 26 up to $\sigma = 0.10$ for the smoothed sun (ricb 0, N 160, lmax 30).

They are converged in N but not in lmax: their eigenfunctions peak at $l \approx \mathrm{lmax}$, and the sun has 43
and 57 of them, $\sigma$ up to 0.12, at lmax 40 and 50. The ThermaD = 1e-10 runs have the same ones. They are absent
with `thermal_wind = 0`, with $S_{tw} = 0$ in the new rows, and with ThermaD = 1e-3. So they are small-scale
instabilities of the diffusionless thermal wind, unresolved in $l$, not an artefact of the new rows.

**Eigensolver near the band.** ARPACK converges badly near it: csolve at 0 + 0.6i got 11/12 after 5000 iterations at
ThermaD 0 and at 1e-10, and some shift-inverts at T2 took tens of minutes. Use shifts away from that band.

**Resolution.** In the full sphere, ThermaD = 0 eigenfunctions need more N than ThermaD = 1e-3 ones (e: N 120 → 200
for the smoothed sun).

**Ruiz prescaling.** `solve.py` turns Ruiz prescaling off automatically for ThermaD = 0. That was tuned on the
$r\,(\mathrm{eq})/p_0$ rows and is not tested here for the $r^2$ rows (no PETSc/SLEPc in this container).

**Behaviour change.** A run with `thermal = 1`, `diff_rot = 1`, `ThermaD = 0` and the default `thermal_wind = 1` used
to stop and now runs. The old behaviour (advection of $s$ only) needs `thermal_wind = 0`, as before.

**Coverage.** The rigid test (c) does not exercise the thermal wind, since $S_{tw} = 0$ for a rigid offset. The
thermal-wind blocks are covered by a, b and d.

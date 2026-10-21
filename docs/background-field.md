# Background magnetic field

With `magnetic = 1`, Kore solves for a small magnetic field perturbation $\mathbf{b}$ on top of a steady, imposed background field $\mathbf{B}_0$. The background field is axisymmetric and poloidal:

$$
\mathbf{B}_0 = c_{norm}\, \nabla\times\nabla\times\left( h(r)\, Y_l^0(\theta)\, \mathbf{r} \right),
$$

where $\mathbf{r}$ is the position vector, $Y_l^0$ is a Schmidt semi-normalised spherical harmonic ($Y_1^0 = \cos\theta$), and $c_{norm}$ is set by `cnorm`. The radial component is $B_{0r} = c_{norm}\, l(l+1)\, h\, Y_l^0 / r$. The field enters the Lorentz force and the induction term of the linearised equations, the magnetic torques on the boundaries, and `spin_doctor`'s diagnostics.


## Field shapes

`B0` selects the radial function $h(r)$:

| `B0` | $l$ | $h(r)$ | Notes |
|---|---|---|---|
| `'axial'` | 1 | $r/2$ | Uniform field along the rotation axis, $\mathbf{B}_0 = c_{norm}\,\hat{\mathbf{z}}$. Imposed from outside the core. It has no cylindrical-radial component, so it supports no torsional waves. The default. |
| `'dipole'` | 1 | $1/(2r^2)$ | Point dipole at the centre, current-free in the fluid. Needs an inner core (`ricb > 0`). |
| `'G21 dipole'` | 1 | $r/6 - r^3/10$ | Dipole made by currents in the fluid (Gerick et al. 2021, GJI). |
| `'Luo_S1'` | 1 | $(5 - 3r^2)\, r$ | Luo & Jackson (2022, Proc. R. Soc. A) field S1. Same shape as `'G21 dipole'` (30 times larger). |
| `'Luo_S2'` | 2 | $(157 - 296 r^2 + 143 r^4)\, r^2$ | Luo & Jackson (2022) quadrupolar field S2. |
| `'FDM'` | `B0_l` | spherical Bessel functions, see below | Poloidal free magnetic decay mode (Zhang & Fearn 1994, 1995; Schmitt 2012). |

The free decay mode is $h = j_l(\beta r)$ in a full sphere (Gubbins & Roberts 1987), and $h = j_l(\beta r)\, y_{l-1}(\beta) - j_{l-1}(\beta)\, y_l(\beta r)$ in a shell (Zhang & Fearn 1995), with $j_l$ and $y_l$ the spherical Bessel functions. $\beta$ is the decay-mode wavenumber, found by a root search started from `beta` (default 3); the value found is written to `params.dat` as `beta_actual`. `B0_l` (default 1) is the degree of the mode.

The fields with odd $l$ are equatorially antisymmetric and `'Luo_S2'` is symmetric. The symmetry of $\mathbf{b}$ is that of the flow (`symm`) times that of $\mathbf{B}_0$.


## Normalisation

`cnorm` sets the amplitude $c_{norm}$, and with it what "a field of unit strength" means in the Lehnert and Elsasser numbers:

| `cnorm` | Meaning |
|---|---|
| a number | $c_{norm}$ itself (default 1). |
| `'rms_cmb'` | The rms of $B_{0r}$ over the CMB is 1. |
| `'mag_energy'` | $\int B_0^2\, dV = 1$ over the fluid shell, as in Luo & Jackson (2022). |
| `'Schmitt2012'` | $\int B_0^2\, dV = 4$ over the fluid shell, as in Schmitt (2012). |

The comments in `params_default.py` list numeric values that reproduce published normalisations:

| Value | Field | Equivalent to |
|---|---|---|
| 3.863752890 | FDM, $l = 1$, `ricb = 0.35` (Schmitt 2012, G101) | `'Schmitt2012'` |
| 4.067144 | FDM, $l = 1$, `ricb = 0.35` (Zhang & Fearn 1994) | not checked |
| $15\sqrt{21/(46\pi)}$ | `'G21 dipole'`, full sphere | `'Schmitt2012'` |
| 1.094357234 | FDM, $l = 1$, full sphere | `'Schmitt2012'` |
| 3.438024656 | FDM, $l = 1$, `ricb = 0.001` | `'Schmitt2012'` |
| 0.09530063707257978 | `'Luo_S1'`, full sphere | `'mag_energy'` |
| 0.6972166887783963 | `'Luo_S1'`, full sphere | volume rms of $B_{0s}$ = 1 |
| 0.005061567359972097 | `'Luo_S2'`, full sphere | `'mag_energy'` |
| 0.0158567582314039 | `'Luo_S2'`, full sphere | volume rms of $B_{0s}$ = 1 |

With an inner core, `'mag_energy'` normalises over the shell, so the field comes out stronger than in a full sphere. To keep exactly the currents of the full-sphere field in the fluid, use its numeric full-sphere value instead.


## The field inside the inner core: `B0_ic`

`'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` are defined for a full sphere. With an inner core (`ricb > 0`) they do not match a potential field at the ICB, so part of the electric current that makes them would have to flow inside the inner core, or in a current sheet on the ICB. Nothing keeps such a current steady in a solid inner core of finite conductivity: there is no flow there, hence no electromotive force, and an axisymmetric azimuthal current decays on the inner core's magnetic diffusion time.

| `B0_ic` | Effect |
|---|---|
| `'potential'` (default) | Removes that part of the current. $h$ becomes $h + \beta_{ic}\, r^{-(l+1)}$, with $\beta_{ic} = r_i^{l+1}\,(r_i h' - l h)/(2l+1)$ evaluated at the ICB. The added term carries no current and keeps the CMB matching, so the currents in the fluid are unchanged, and $\mathbf{B}_0$ continues into the inner core as the potential field $h(r_i)\,(r/r_i)^l$. This is the self-consistent steady field for an insulating inner core and for any finite inner-core conductivity. |
| `'full sphere'` | $h$ unchanged; its currents extend into the inner core. Self-consistent only for a perfectly conducting inner core, or as a snapshot of currents frozen in the inner core. |

`B0_ic` has no effect for a full sphere, nor for `'axial'` and `'FDM'` (they already match a potential field at the ICB) or `'dipole'` (a point source at the centre). For `'Luo_S2'` with `ricb = 0.35` the change is 10% of the field at the ICB and 1.7% at the CMB, and it lowers the frequency of the S2 torsional mode by about 1% (insulating inner core).


## A thin conducting wall at the CMB: `B0_cmb`

With a thin conducting layer at the base of the mantle (`mantle = 'TWA'`, see [Inner core and mantle](boundaries.md#thin-conducting-wall)), the background field must satisfy the wall's matching condition too. A steady $\mathbf{B}_0$ drives no current in the wall, so only the wall's permeability terms act on it, not its conductance `c1_cmb`. The condition at the CMB is

$$
\left(\frac{1}{\mu} + l\,c_{cmb}\right) (r h)' + (l + e)\,h = 0, \qquad e = \frac{c_{cmb}\,\mu\,l}{p}\left(\frac{l+1}{p} - 1\right),
$$

with $p$ = `muw_cmb`. For $p = 1$ (the default) and $\mu = 1$ it is the insulating condition, so `B0_cmb` changes nothing: a static field crosses a wall with the permeability of vacuum unchanged.

| `B0_cmb` | Effect |
|---|---|
| `'wall'` (default) | Adds $\alpha\, r^l$ to $h$, with $\alpha$ chosen to satisfy the condition. The added term carries no current and keeps the ICB matching of `B0_ic`, so the currents in the fluid are unchanged. For $l = 1$ it is a uniform axial field. |
| `'plain'` | $h$ unchanged. The field then meets the wall condition only for `mu = 1` and `c_cmb = 0` or `muw_cmb = 1`. |

`B0_cmb` has no effect with an insulating mantle, with `mu = 1` and `c_cmb = 0` or `muw_cmb = 1`, or for `'axial'`. With the former condition (`muw_cmb = np.inf`, a highly permeable wall), for the S2 torsional mode with `c1_cmb = 0.015` it raises the frequency by 0.27% at `c_cmb = 0.01` and by 5.8% at `c_cmb = 0.3`.

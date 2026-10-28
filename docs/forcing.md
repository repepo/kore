# Convective forcing

The inertial modes of the Sun are observed with finite amplitudes because the convection keeps exciting them while it
also damps them. Forcing 11 adds this excitation to kore's linear equations as a random body force: the fluctuating
part of the Reynolds stress of the convective eddies. Its strength is tied to kore's viscosity, which in a convection
zone stands for the same eddies, so that a mode receives from the eddies the energy that their viscosity takes away.
kore returns the power spectrum of the kinetic energy of the forced flow at the frequencies you choose, and from it the
energy that each mode reaches. Where the force comes from and why its strength follows from the viscosity is explained
on [Reynolds-stress forcing](reynolds-stress.md). This page shows how to run it and how to read the result.

## What a run computes

The force is random, so a single solution means nothing: only its statistics do. For each forcing frequency $w$, kore
solves

$$
(A - i\thinspace w\thinspace B)\thinspace\mathbf x_j = \mathbf b_j, \qquad j = 1, \dots, K,
$$

where $A$ and $B$ are the matrices of the eigenproblem ([Numerical method](numerical-method.md#eigenproblem)) and the
$K$ = `rsf_nsamp` right-hand sides $\mathbf b_j$ are independent realisations of the random force. The mean of the
kinetic energies of the $K$ solutions is the power spectral density of the kinetic energy at $w$. It is an estimate,
whose standard error goes as $1/\sqrt K$. The matrix is factored once per frequency, and each realisation then costs
only a back-substitution, so the cost of a run is set by the number of frequencies.

The flow goes as $e^{iwt}$, so that $w$ is the frequency $-\omega$ of the [Anelastic equations](anelastic.md#non-dimensionalization)
in kore's units: negative $w$ is prograde, as for `forcing_frequency`. A mode of eigenvalue $\lambda$ resonates at
$w = \mathrm{Im}\thinspace\lambda$. Near a damped mode that is well separated from the others, the spectrum is a
Lorentzian centred there, with a half-width at half maximum of $\lvert\mathrm{Re}\thinspace\lambda\rvert$, and the
area under it is the kinetic energy of the mode.

## Example: a Rossby mode

This example forces the sectoral Rossby mode $l = m = 2$ of an incompressible fluid in a uniformly rotating spherical
shell, and recovers its resonance. It takes less than a minute on four cores.

### 1. Write the parameter file

Make a run directory as in [Getting started](getting-started.md), and write in `bin/parameters.py`

```python
from params_default import default_params

par = default_params()

par.timescale = 'rotation'
par.Ek = 1e-5
par.set_scales()                  # Gaspard = 1, ViscosD = Ek

# Incompressible fluid in a uniformly rotating shell, stress-free walls
par.structure = 'Boussinesq'
par.thermal = 0
par.ricb = 0.35
par.bci = 0
par.bco = 0

# m = 2; the sectoral Rossby mode is equatorially antisymmetric
par.m = 2
par.symm = -1

par.N = 96
par.lmax = 49
par.ncpus = 4

# Random Reynolds-stress forcing, 16 realisations
par.forcing = 11
par.forcing_frequency = 0.0
par.rsf_nsamp = 16
par.rsf_taper = 0.2               # the default, 0.04, needs N of about 256 here
par.petsc_opts['mat_mumps_icntl_14'] = 100

# Target of the eigenvalue solve of step 3
par.rtau = 0.0
par.itau = 0.667
par.nev = 4
```

The viscosity is uniform, so all of it counts as turbulent: the turbulent viscosity is $\nu_t$ = `ViscosD` =
$10^{-5}$. The strength of the forcing is a turbulent temperature $\Theta$, the energy of one eddy. With the default
law, $\Theta = 3\rho_0\nu_t^2\ell_c$, and with the default eddy size $\ell_c = 0.25$ it is $7.5\times10^{-11}$ here,
in kore's units of energy. `rsf_taper` sets the width over which the force goes to zero at the walls. Keep
`forcing_frequency = 0`: the scan of step 4 sets the frequencies, and $A$ is then the matrix of the eigenproblem,
which step 3 uses.

### 2. Build the matrices

```shell
./bin/submatrices.py 4
mpiexec -n 4 ./bin/assemble.py > log_asm
```

`assemble.py` writes `A.npz` and `B.npz`, and the $K$ right-hand sides as the columns of `B_forced.npz`. `log_asm`
describes the forcing:

```
 Random Reynolds-stress forcing (forcing = 11): 16 realisation(s), seed 0
 20 noise centres on r = -0.1803 .. 1.4990, l_c = 0.25 .. 0.25, nu_t floor 0 ViscosD
 max Theta nu_t = 7.500e-16 at r = 1.000 (theta = nu, scale 1)
 Chebyshev tail of the rows (max over the last 10% / max): 3.5e-07
```

The last line checks that `N` resolves the force. Above $10^{-6}$ it ends with "WARNING: under-resolved, raise N or
l_c": then raise `N`, or widen the sharpest edge of the force (`rsf_taper`, or `rsf_rcut_width` below). With the
default `rsf_taper` this line reads 0.13 at N = 96.

### 3. Find the mode

The eigenvalue problem can reuse the same matrices. In a subdirectory:

```shell
mkdir eig
cp -r bin eig/
ln -s $PWD/A.npz $PWD/B.npz eig/
echo "par.forcing = 0" >> eig/bin/parameters.py
cd eig
mpiexec -n 4 ./bin/solve.py
./bin/spin_doctor.py 4
cd ..
```

The first eigenvalue is the Rossby mode, $\lambda = -6.934\times10^{-5} + 0.66666674\thinspace i$, with residσ about
$10^{-12}$. Its frequency is close to the inviscid value $\omega = -2/(m+1)$, retrograde.

### 4. Scan the frequencies

Add the frequencies to `bin/parameters.py`, 21 points over five half-widths on each side of the resonance:

```python
import numpy as np
w0, s = 0.66666674, 6.93e-5       # Im λ and |Re λ| of the mode
par.rsf_frequencies = w0 + s*np.linspace(-5, 5, 21)
```

and solve:

```shell
mpiexec -n 4 ./bin/solve.py
```

`solve.py` prints one line per frequency as it goes, for example

```
w = 6.666667400e-01   mean KE 4.442305e-07 +- 1.1e-07   (0.7 s)
```

and appends the same numbers to `rsf_spectrum.dat`, and the kinetic energy of each realisation to `rsf_ke.dat`. A scan
writes no fields.

### 5. Read the spectrum

`rsf_spectrum.dat` has one row per frequency: $w$, the mean kinetic energy and its standard error.

```
# w  mean_KE  stderr   (forcing = 11, nsamp 16, seed 0, rsf_split None)
6.663202400000e-01 1.711741196503e-08 4.052316136428e-09
...
6.665974400000e-01 2.222626974507e-07 5.266390856613e-08
6.666320900000e-01 3.554873668819e-07 8.423321121690e-08
6.666667400000e-01 4.442304926635e-07 1.052621556910e-07
6.667013900000e-01 3.554458399913e-07 8.422364035422e-08
6.667360400000e-01 2.222297651836e-07 5.265642773220e-08
...
```

The peak is at $w = \mathrm{Im}\thinspace\lambda$, and the spectrum falls to half the peak at
$w = \mathrm{Im}\thinspace\lambda \pm \lvert\mathrm{Re}\thinspace\lambda\rvert$: the Lorentzian of the mode. The energy
of the mode is the height of the Lorentzian times $\lvert\mathrm{Re}\thinspace\lambda\rvert/2$, here
$4.44\times10^{-7}\times6.93\times10^{-5}/2 = 1.5\times10^{-11}$. In the limit of eddies much smaller than the mode
and of a force that reaches the walls, the mode would reach $\Theta$, since all of its damping is turbulent
([Reynolds-stress forcing](reynolds-stress.md#energy-of-a-mode)). The finite eddy size and the wide taper lower
it here to a fifth of $\Theta$.

The standard error is a quarter of the mean, as expected for 16 realisations. The spectrum is nevertheless smooth,
because every frequency uses the same realisations: the shape of a spectrum is much better determined than its level.

## Setting up a run

### Requirements

- `ViscosD` > 0: the force is the partner of the viscous force.
- In a full sphere (`ricb = 0`), `N` must be even, and `rpower_u` ≥ 5, `rpower_v` ≥ 3 (the defaults).
- `forcing_amplitude_cmb` is not used: the power scales with `rsf_scale` instead.
- The heat equation is not forced: the right-hand side is zero in its rows, also with `thermal = 1`.

`assemble.py` stops with a message if one of the first two is not met.

### Where the force acts

The force follows the turbulent part of the viscosity,
$\nu_t$ = `ViscosD` $(\hat\nu - \nu_f)$, clipped at 0, with $\hat\nu$ the viscosity profile of
[Background model](background.md#diffusivity) and $\nu_f$ the part that is not turbulent. By default $\nu_f$ is `visc0`
for the diffusivities `'step'` and `'solar_rz'` when `visc0` < 1, the viscosity of the radiative zone, and 0 otherwise.
Where $\nu_t = 0$ there is no force at all.

- With a `'user'` viscosity profile, set `rsf_nu_floor` to its radiative-zone value, or the radiative zone is forced
  too.
- To keep the force out of the tail of a viscosity step, cut it below a radius with `rsf_rcut`. The force is zero below
  `rsf_rcut` and full above `rsf_rcut` + `rsf_rcut_width`, joined smoothly. Do not raise `rsf_nu_floor` for this: the
  clipped $\nu_t$ then has a corner that the Chebyshev expansion cannot resolve.
- The edges need radial resolution. In a full sphere, an edge near $r = 0.7$ of width 0.04 needs N of about 1280, and
  one of width 0.08 about 900. The tail line of `log_asm` tells you when an edge is under-resolved.
- The force goes to zero over `rsf_taper` below the outer wall and above an inner core, and over `rsf_taper_in` above
  $r = 0$ in a full sphere, so that it needs no traction at the walls.
- `rsf_split` = a radius adds the kinetic energy below that radius to the output, for example the energy in a radiative
  zone.

### Strength of the force

The variance of the force is proportional to $\Theta\nu_t$. `rsf_theta` sets the law for $\Theta$:

| `rsf_theta` | $\Theta$ | depends on $\nu$ |
| --- | --- | --- |
| `'nu'` (default) | $3\rho_0\nu_t^2\ell_c$: eddies of size $\ell_c$ whose mixing-length viscosity $u_c\ell_c/3$ is $\nu_t$ | as $\nu_t^2$ |
| `'mlt'` | $\tfrac13\rho_0u_c^2\ell_c^3$, with the mixing-length velocity $u_c = (\alpha F_c/10\rho_0)^{1/3}$ of the convective flux $F_c$, $\alpha$ = `rsf_alpha` | no |
| a number | that constant | no |

- `rsf_scale` multiplies $\Theta$, and with it the whole spectrum.
- `'mlt'` needs `rsf_lum`, the luminosity $L_\star$ in kore's units, $\rho_u L^5/\tau^3$, with $\rho_u$ the unit of
  density of the structure (the central density for the Model S structures) and $L$, $\tau$ the units of length and
  time: $7.0\times10^{-7}$ for the Sun with a Model S structure, `r_out` = 0.985 and $\tau = 1/\Omega$,
  $\Omega/2\pi$ = 456 nHz. It takes the convective flux as $L_\star/(4\pi r^2)$ times $\nu_t/\max\nu_t$, and needs
  `visc_pow = 0`.
- The eddy size $\ell_c$ is a smooth minimum of `rsf_alpha` $H_p$ ($H_p$ the pressure scale height) and `rsf_lc_max`,
  or the constant `rsf_lc`. With `rsf_angular = 1` (the default) the eddies also have a finite horizontal size, which
  weakens the force at degrees $l \gtrsim r/\ell_c$.
- **The Sun.** The observed amplitudes of the solar Rossby modes (Löptien et al. 2018) correspond to
  $\Theta_{\rm eff} \approx (0.6\text{–}3)\times10^{34}$ erg, $(0.3\text{–}1.6)\times10^{-11}$ in kore's units with
  the solar values above. That is about $10\thinspace(10^{12}\thinspace\mathrm{cm^2\thinspace s^{-1}}/\nu_t)^2$ times
  the `'nu'` law, or $1/20$ of the `'mlt'` law, for eddies much smaller than the modes. With the default eddy size and
  angular factor the same modes are forced a few times less, so `rsf_scale` must be larger by as much. Treat these as
  starting values for `rsf_scale`.

### Frequencies

- Around a mode: about 20 points over $\mathrm{Im}\thinspace\lambda \pm 5\thinspace\lvert\mathrm{Re}\thinspace\lambda\rvert$,
  and one at $\mathrm{Im}\thinspace\lambda$. An eigenvalue solve on the same matrices (step 3) gives the modes.
- Between modes, a uniform grid. A mode whose half-width is smaller than the grid step can fall between two points:
  add a point at each resonance you care about.
- `rsf_frequencies` takes any list or array, for example `np.loadtxt('freqs.txt')` for a file with one frequency per
  line.
- For $m = 0$ the spectrum is even in $w$: scan $w > 0$ only.

### Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `rsf_nsamp` | 1 | number of realisations; 16 to 30 is usual |
| `rsf_seed` | 0 | realisation $j$ uses `numpy.random.default_rng([rsf_seed, j])` |
| `rsf_theta` | `'nu'` | the law of $\Theta$ (above) |
| `rsf_scale` | 1.0 | multiplies $\Theta$ |
| `rsf_lum` | None | luminosity in kore's units, for `'mlt'` |
| `rsf_alpha`, `rsf_lc_max` | 2.0, 0.25 | the eddy size, a smooth minimum of `rsf_alpha` $H_p$ and `rsf_lc_max` |
| `rsf_lc` | None | a constant eddy size instead |
| `rsf_angular` | 1 | 1: eddies of finite horizontal size; 0: white in angle |
| `rsf_nu_floor` | None | the part $\nu_f$ of $\hat\nu$ that is not turbulent; None: `visc0` for `'step'` and `'solar_rz'` with `visc0` < 1, else 0 |
| `rsf_taper` | 0.04 | width over which the force goes to zero at the walls |
| `rsf_taper_in` | 0.2 | the same above $r = 0$ in a full sphere |
| `rsf_rcut`, `rsf_rcut_width` | None, 0.04 | no force below `rsf_rcut`, full force above `rsf_rcut` + `rsf_rcut_width` |
| `rsf_frequencies` | None | the frequencies of the scan, in kore's units (negative prograde) |
| `rsf_split` | None | a radius: also report the kinetic energy below it |

## Reading the output

| file | columns |
| --- | --- |
| `rsf_spectrum.dat` | $w$, the mean kinetic energy and its standard error; with `rsf_split`, then the mean kinetic energy below `rsf_split` and its standard error |
| `rsf_ke.dat` | $w$, the kinetic energy of each realisation; with `rsf_split`, then the kinetic energy below `rsf_split` of each realisation |

Each scan first writes a line starting with `#`, which `np.loadtxt` skips.

- **Power spectral density.** The mean kinetic energy at $w$ is a two-sided power spectral density, in kore's units:
  for $m \neq 0$, $\int(\text{mean KE})\thinspace dw/2\pi$ is the mean kinetic energy of the forced flow. For $m = 0$ the
  flow is real, the spectrum is even in $w$, and that integral over all $w$ is twice the energy: integrate over $w > 0$
  only.
- **One symmetry class.** The force drives the two equatorial symmetries independently, and a run solves one of them
  (`symm`). The energy at one $m$ is the sum of a `symm = 1` and a `symm = -1` run.
- **Noise.** The standard error goes as $1/\sqrt K$, $K$ = `rsf_nsamp`: about a quarter of the mean with 16
  realisations near a mode, 18 % with 30. Realisation $j$ is the same at every frequency, in every scan and for any
  `rsf_nsamp`, so the shape of a spectrum is less noisy than its level, and scans of the same configuration can be
  joined.
- **Energy of a mode.** Fit a Lorentzian plus a smooth background to the peak: the energy of the mode is the height
  of the Lorentzian times $\lvert\mathrm{Re}\thinspace\lambda\rvert/2$. Its half-width should be
  $\lvert\mathrm{Re}\thinspace\lambda\rvert$. Other modes add a background to the raw peak.
- **Non-normal operators.** With a differential rotation the modes are not orthogonal, and the forced response can
  differ from the sum of the modes' Lorentzians, or exceed the energy that the fluctuation–dissipation relation gives a
  single mode. The forced solve includes this.
- **Unstable modes** have no stationary response: the spectrum near a growing mode means nothing. Check the eigenvalues
  near the frequencies you scan.

Without `rsf_frequencies`, `solve.py` solves at `forcing_frequency` only and writes the fields of the forced flow,
one column per realisation, as for the eigenvectors of an eigenvalue run ([Output files](output.md)).

## Large runs

- **Several scans in parallel.** Each scan needs only `bin/` and the matrices. Copy `bin/` into a new directory, link
  `A.npz`, `B.npz` and `B_forced.npz` into it, append `par.rsf_frequencies = ...` to its parameter file, and run
  `solve.py` there. Split the list of frequencies between such directories and run them at the same time.
- **Restarts.** A row is appended as soon as its frequency is solved, so a scan can be read while it runs, and a
  stopped scan only needs the remaining frequencies. Rerun in the same directory only with the same `B_forced.npz`:
  with another number of realisations the rows differ in length and `np.loadtxt` cannot read the file.
- **Failed frequencies.** A frequency whose realisations are all NaN or Inf prints "Solver crashed at w = …, got
  nan's!", and the scan goes on with the next one. The cause is a MUMPS factorisation that ran out of workspace: set
  `par.petsc_opts['mat_mumps_icntl_14'] = 100` from the start, as in the example, and rerun any frequency that still
  fails with 200 ([Solver](solver.md#bad-solutions)). A forced run reads the `mat_` options, an eigenvalue run the `st_` options:
  use `st_mat_mumps_icntl_14` for the eigenvalue solve of step 3.
- **Cost.** Each frequency costs one LU factorisation of a matrix of the size of the eigenproblem's, with the same
  memory, plus one back-substitution per realisation, which is much cheaper than the factorisation.
- **Several forcings at once.** $A$ and $B$ do not depend on the force, so several configurations of the forcing (other
  `rsf_theta`, `rsf_rcut`, …) can share the factorisations: put their right-hand sides side by side as columns of
  `B_forced.npz`. `rsf_spectrum.dat` then averages over all columns, so read `rsf_ke.dat` by groups of columns.

The checks of the implementation are in `tests/rsf/`, with a README.

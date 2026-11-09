# Convective forcing

The inertial modes of the Sun are observed with finite amplitudes because convection keeps exciting them while it also
damps them. Forcing 11 adds this excitation to kore's linear equations as a random force: the fluctuating part of the
Reynolds stress of the convective eddies. Its strength is tied to kore's viscosity, which in a convection zone
represents the same eddies, so that the energy a mode receives from the eddies balances the energy their viscosity
removes. kore returns the power spectrum of the kinetic energy of the forced flow at the frequencies you choose, and from
it you get the energy of each mode. The physics behind the force is on [Reynolds-stress forcing](reynolds-stress.md).
This page shows how to run it and how to read the result.

## What a run computes

The force is random, so only the statistics of the solution are meaningful. For each forcing frequency $w$, kore solves
the forced problem for `rsf_nsamp` independent random forces and averages their kinetic energies. This average is the
power spectral density of the kinetic energy at $w$; its uncertainty decreases as one over the square root of the
number of samples. The matrix is factored once per frequency and reused for all samples, so the cost of a run is set
mainly by the number of frequencies.

Frequencies follow the convention of `forcing_frequency`: negative $w$ is prograde. A mode with eigenvalue $\lambda$
resonates at $w = \mathrm{Im}\thinspace\lambda$. Near a well-isolated mode the spectrum is a Lorentzian peak centred
there, whose half-width is the mode's damping rate $\lvert\mathrm{Re}\thinspace\lambda\rvert$, and whose area is the
kinetic energy of the mode.

## Example: a Rossby mode

This example forces the sectoral Rossby mode $l = m = 2$ of an incompressible fluid in a uniformly rotating spherical
shell, and recovers its resonance. It takes less than a minute.

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

# Random Reynolds-stress forcing, 16 samples
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

The viscosity is uniform, so all of it counts as turbulent and drives the force. `rsf_taper` sets the width over
which the force goes to zero at the walls. Keep `forcing_frequency = 0`: the scan of step 4 sets the frequencies, and
the same matrices are reused for the eigenvalue solve of step 3.

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

The last line checks that `N` resolves the force. If it ends with "WARNING: under-resolved, raise N or l_c", raise `N`
or widen the sharpest edge of the force (`rsf_taper`, or `rsf_rcut_width` below).

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

The first eigenvalue is the Rossby mode, with a frequency close to the inviscid value $2/(m+1) = 2/3$ and a damping
rate of about $7\times10^{-5}$.

### 4. Scan the frequencies

Add the frequencies to `bin/parameters.py`, 21 points spanning five half-widths on each side of the resonance:

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

and appends the same numbers to `rsf_spectrum.dat`, and the kinetic energy of each sample to `rsf_ke.dat`. A scan
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

The peak is at the frequency of the mode, and the spectrum falls to half the peak one damping rate away on either side:
the Lorentzian of the mode. The energy of the mode is the peak height times half the damping rate, here about
$1.5\times10^{-11}$ in kore's units.

The standard error is about a quarter of the mean, as expected for 16 samples. The spectrum is still smooth, because
every frequency uses the same random samples: the shape of a spectrum is much better determined than its level.

## Setting up a run

### Requirements

- `ViscosD` > 0: the force is the partner of the viscous force.
- In a full sphere (`ricb = 0`), `N` must be even, and `rpower_u` ≥ 5, `rpower_v` ≥ 3 (the defaults).
- `rsf_taper` > 0, and `rsf_taper_in` > 0 in a full sphere: the force must vanish at the walls (below). With
  `rsf_rcut`, `ricb` ≤ `rsf_rcut` < 1 − `rsf_rcut_width` and `rsf_rcut_width` > 0.
- `forcing_amplitude_cmb` is not used: the power scales with `rsf_scale` instead.
- Only the momentum equation is forced, not the entropy or composition equations. Without thermal or compositional
  diffusion (`ThermaD = 0` or `CompoD = 0`), some combinations of frequency and differential rotation have no
  well-behaved solution; kore stops in those cases ([ThermaD = 0](background.md#thermad-0)).
- A steady run (a frequency 0) needs walls that stay in place, `bco` and `bci` 0 or 1.
- With `magnetic = 1` the perturbation of the field is solved with the flow, and a scan needs the inner core
  `'conducting, Chebys'` rather than `'conducting, Bessel'` ([Magnetic field](magnetic.md#forced-runs)).
- With the topographic drag (`topo_drag = 'rekier'`) a scan needs `topo_omega` ([Drag at the outer wall](drag.md#drag-of-the-topography-topo_drag-rekier)).
- Like every forced run it needs a spherical body (`centrifugal = 0`); [Combining features](compatibility.md#forced-runs)
  shows what else it combines with.

kore stops with a message where a requirement is not met.

### Where the force acts

The force acts only where the viscosity is turbulent: the part of the viscosity profile
([Background model](background.md#diffusivity)) above a floor that represents the non-turbulent viscosity. With the
diffusivities `'step'` and `'solar_rz'`, the floor is by default `visc0`, the viscosity of the radiative zone, so the
radiative zone is not forced. Otherwise the floor is 0.

- With a `'user'` viscosity profile, set `rsf_nu_floor` to its radiative-zone value, or the radiative zone is forced
  too.
- To keep the force out of the tail of a viscosity step, cut it below a radius with `rsf_rcut`. The force is zero below
  `rsf_rcut` and full above `rsf_rcut` + `rsf_rcut_width`, joined smoothly. Do not raise `rsf_nu_floor` for this: the
  clipped $\nu_t$ then has a corner that the Chebyshev expansion cannot resolve.
- Sharp edges need high radial resolution (in a full sphere, an edge of width 0.04 can need `N` above 1000). The
  last line of `log_asm` tells you when an edge is under-resolved.
- The force goes to zero over `rsf_taper` below the outer wall and above an inner core, and over `rsf_taper_in` above
  $r = 0$ in a full sphere, so that it needs no traction at the walls.
- `rsf_split` = a radius adds the kinetic energy below that radius to the output, for example the energy in a radiative
  zone.

### Strength of the force

The strength of the force is set by a "turbulent temperature" $\Theta$, the kinetic energy of one eddy.
In the ideal case of eddies much smaller than the mode, a mode whose damping is entirely turbulent reaches an energy
$\Theta$. `rsf_theta` chooses how $\Theta$ is computed:

| `rsf_theta` | $\Theta$ |
| --- | --- |
| `'nu'` (default) | from the turbulent viscosity and the eddy size, by mixing-length theory |
| `'mlt'` | from the convective flux of the star, by mixing-length theory; needs `rsf_lum` and `visc_pow = 0` |
| a number | that constant |

- `rsf_scale` multiplies $\Theta$, and with it the whole spectrum.
- `rsf_lum`, for `'mlt'`, is the luminosity in kore's units. For the Sun with a Model S structure, `r_out` = 0.985 and a
  frame rotating at 456 nHz, it is $7.0\times10^{-7}$.
- The eddy size is the smaller of `rsf_alpha` pressure scale heights and `rsf_lc_max`, or the constant `rsf_lc`. With
  `rsf_angular = 1` (the default) the eddies also have a finite horizontal size, which weakens the force at high
  harmonic degrees.
- **The Sun.** The observed amplitudes of the solar Rossby modes (Löptien et al. 2018) correspond to $\Theta$ of about
  $10^{-11}$ in kore's units, with the solar values above. Use this to choose a starting value of `rsf_scale`.

### Frequencies

- Around a mode: about 20 points spanning five damping rates on each side of its frequency, including one at the
  frequency itself. An eigenvalue solve on the same matrices (step 3) gives the modes.
- Between modes, a uniform grid. A mode whose half-width is smaller than the grid step can fall between two points:
  add a point at each resonance you care about.
- `rsf_frequencies` takes any list or array, for example `np.loadtxt('freqs.txt')` for a file with one frequency per
  line.
- For $m = 0$ the spectrum is even in $w$: scan $w > 0$ only.

### Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `rsf_nsamp` | 1 | number of samples; 16 to 30 is usual |
| `rsf_seed` | 0 | sample $j$ uses `numpy.random.default_rng([rsf_seed, j])` |
| `rsf_theta` | `'nu'` | the law of $\Theta$ (above) |
| `rsf_scale` | 1.0 | multiplies $\Theta$ |
| `rsf_lum` | None | luminosity in kore's units, for `'mlt'` |
| `rsf_alpha`, `rsf_lc_max` | 2.0, 0.25 | the eddy size: the smaller of `rsf_alpha` pressure scale heights and `rsf_lc_max` |
| `rsf_lc` | None | a constant eddy size instead |
| `rsf_angular` | 1 | 1: eddies of finite horizontal size; 0: white in angle |
| `rsf_nu_floor` | None | the non-turbulent part of the viscosity profile; None: `visc0` for `'step'` and `'solar_rz'`, else 0 |
| `rsf_taper` | 0.04 | width over which the force goes to zero at the walls |
| `rsf_taper_in` | 0.2 | the same above $r = 0$ in a full sphere |
| `rsf_rcut`, `rsf_rcut_width` | None, 0.04 | no force below `rsf_rcut`, full force above `rsf_rcut` + `rsf_rcut_width` |
| `rsf_frequencies` | None | the frequencies of the scan, in kore's units (negative prograde) |
| `rsf_split` | None | a radius: also report the kinetic energy below it |

## Reading the output

| file | columns |
| --- | --- |
| `rsf_spectrum.dat` | $w$, the mean kinetic energy and its standard error; with `rsf_split`, then the mean kinetic energy below `rsf_split` and its standard error |
| `rsf_ke.dat` | $w$, the kinetic energy of each sample; with `rsf_split`, then the kinetic energy below `rsf_split` of each sample |

Each scan first writes a line starting with `#`, which `np.loadtxt` skips.

- **Power spectral density.** The mean kinetic energy is a two-sided power spectral density: integrating it over
  $w$ (divided by $2\pi$) gives the mean kinetic energy of the forced flow. For $m = 0$ the spectrum is symmetric in
  $w$; integrate over positive $w$ only.
- **One symmetry class.** The force drives the two equatorial symmetries independently, and a run solves one of them
  (`symm`). The energy at one $m$ is the sum of a `symm = 1` and a `symm = -1` run.
- **Noise.** The standard error is about a quarter of the mean with 16 samples, and decreases as one over the square
  root of `rsf_nsamp`. Each sample is the same at every frequency and in every scan, so the shape of a spectrum is less
  noisy than its level, and scans of the same configuration can be joined.
- **Energy of a mode.** Fit a Lorentzian plus a smooth background to the peak: the energy of the mode is the height
  of the Lorentzian times half the damping rate. The fitted half-width should match the damping rate.
- **Differential rotation.** The modes are then not independent of each other, and the forced response can differ
  from the sum of the individual peaks. The forced solve includes this automatically.
- **Unstable modes** have no stationary response: the spectrum near a growing mode means nothing. Check the eigenvalues
  near the frequencies you scan.

Without `rsf_frequencies`, `solve.py` solves at `forcing_frequency` only and writes the fields of the forced flow,
one column per sample, as for the eigenvectors of an eigenvalue run ([Output files](output.md)).

## Large runs

- **Several scans in parallel.** Each scan needs only `bin/` and the matrices. Copy `bin/` into a new directory, link
  `A.npz`, `B.npz` and `B_forced.npz` into it, append `par.rsf_frequencies = ...` to its parameter file, and run
  `solve.py` there. Split the list of frequencies between such directories and run them at the same time.
- **Restarts.** A row is appended as soon as its frequency is solved, so a scan can be read while it runs, and a
  stopped scan only needs the remaining frequencies. Rerun in the same directory only with the same `B_forced.npz`:
  with another number of samples the rows differ in length and `np.loadtxt` cannot read the file.
- **Failed frequencies.** A frequency whose samples are all NaN or Inf prints "Solver crashed at w = …, got
  nan's!", and the scan goes on with the next one. The cause is a MUMPS factorisation that ran out of workspace: set
  `par.petsc_opts['mat_mumps_icntl_14'] = 100` from the start, as in the example, and rerun any frequency that still
  fails with 200 ([Solver](solver.md#bad-solutions)). A forced run reads the `mat_` options, an eigenvalue run the `st_` options:
  use `st_mat_mumps_icntl_14` for the eigenvalue solve of step 3.
- **Cost.** Each frequency costs about as much as one eigenvalue solve (one matrix factorization), with the same
  memory. Extra samples are cheap.
- **Several forcings at once.** $A$ and $B$ do not depend on the force, so several configurations of the forcing (other
  `rsf_theta`, `rsf_rcut`, …) can share the factorisations: put their right-hand sides side by side as columns of
  `B_forced.npz`. `rsf_spectrum.dat` then averages over all columns, so read `rsf_ke.dat` by groups of columns.

The checks of the implementation are in `tests/rsf/`, with a README.

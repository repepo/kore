# Examples and tests

The test cases in `tests/` reproduce published results and serve as worked examples. Each folder holds a complete parameter file, which can be copied to `bin/parameters.py` of a run folder to run the case by hand.


## Running the tests

```Shell
cd tests
pytest . -v -s
```

Each test copies `bin/` into its case folder, installs the case's parameter file as `bin/parameters.py`, runs Kore there with 2 MPI processes, compares the result with the reference and cleans up. All the tests together take a few minutes. They also run on GitHub Actions at every push to the `conductive_IC` branch, with the Ubuntu PETSc and SLEPc packages (complex scalars). Without pytest, the test methods can be called directly from python.


## The default torsional mode

The parameters in `bin/parameters.py` set up a torsional Alfvén mode: an $l = 1$ free-decay-mode field, $\Lambda = 10$, $Pm = 0.1$, $E = 10^{-7}$, `ricb = 0.35`, no-slip boundaries, an insulating inner core and a thin conducting wall at the CMB (`c_cmb = 0.01`, `c1_cmb = 0.015`). The mode closest to the target is $\lambda = -0.00464013 + 0.01150398\,i$ in rotation units, with Tor/Pol = 156 and Mag/Kin = 1.01. The [Quick start](quickstart.md) runs it step by step.


## Spin-over mode

`tests/spinover`, run by `test_spinover.py`. A hydrodynamic problem: $m = 1$, antisymmetric, $E = 10^{-3}$, `ricb = 0.35`, no-slip boundaries, $N = 68$, $l_{max} = 64$, without pre-scaling. The least damped of the 5 eigenvalues found around the default target ($i$) is the spin-over mode,

$$
\lambda = -0.1019947097 + 1.0057336901\,i,
$$

which must be matched to $10^{-8}$.


## Torsional modes of a quadrupolar field

`tests/torsional_s2`, run by `test_torsional_s2.py`. The first two torsional modes of the quadrupolar background field S2 of Luo & Jackson (2022, Proc. R. Soc. A 478, 20210982, table 1): an inviscid full sphere (`Ek = 0`, `ricb = 0`), insulating boundary, $Le = 10^{-4}$, Lundquist number $2\times10^4$, Alfvén time scale, `cnorm = 'mag_energy'`, $N = 128$, $l_{max} = 127$:

```python
par.ricb      = 0
par.Ek        = 0
par.magnetic  = 1
par.B0        = 'Luo_S2'
par.cnorm     = 'mag_energy'  # unit magnetic energy as in Luo & Jackson 2022
Lu            = 2e4
par.Le        = 1e-4
par.Em        = par.Le/Lu
par.timescale = 'Alfven'
par.N         = 128           # set manually (Ncheb is for viscous runs)
par.rtau      = -0.0142       # mode 2
par.itau      = 1.9095
```

The matrices are built once and solved twice, around each mode, since a mode far from the target is less accurate. Mode 1 uses a target given on the command line:

```Shell
mpiexec -n 2 ./bin/solve_nopp.py -eps_target -0.0066+1.0336i
```

| Mode | Kore ($N = 128$) | Luo & Jackson (2022) |
|---|---|---|
| 1 | $-0.0065950164 + 1.0335952716\,i$ | $-0.0065952461 + 1.0335959942\,i$ |
| 2 | $-0.0141843686 + 1.9095212618\,i$ | $-0.0141833345 + 1.9095123626\,i$ |

The test requires Kore's values to $10^{-7}$ and the published ones to $2\times10^{-6}$ (mode 1) and $10^{-5}$ (mode 2), relative to $|\lambda|$.


## Onset of convection

`test_convection_bouss.py` finds the critical Rayleigh number at which the least stable mode stops decaying, with `tests/find_Rac.py`: a root search on the growth rate as a function of the Rayleigh number based on the gap width, `Ra_gap` = `Ra`$\,(1 - r_i)^3$. Both cases use $Pr = 1$, $m = 9$, no-slip boundaries, `which_eigenpairs = 'TR'` with a target on the real axis, and `-eps_balance twoside`.

| Case | Setup | $Ra_c$ (gap) | Drift frequency $\omega_c$ |
|---|---|---|---|
| Jones, Soward & Mussa (2000), `tests/jones2000` | Full sphere, internal heating, fixed heat flux, $Ta = 10^9$ ($E = 6.325\times10^{-5}$), $N = 118$ | $4.66986\times10^{6}$ | $-1.93444\times10^{-2}$ |
| Dormy et al. (2004), `tests/dormy2004` | `ricb = 0.35`, differential heating, fixed temperatures, $E = 2\times10^{-5}$, $N = 74$, `g = 2` | $1.65404\times10^{6}$ | $-1.10162\times10^{-2}$ |

`find_Rac.py` can also be run by hand from a case folder that has `bin/` with the case's parameter file: `../find_Rac.py Ramin m [ncpus]`. The parameter file must contain a line `Ra_gap = ...`, which the script rewrites before each solve.


## Thermo-compositional convection

`tests/silva2019`, run by `test_thermocompositional.py`. With equal Prandtl and Schmidt numbers, temperature and composition combine into a single co-density, and the onset of convection depends only on the sum of the thermal and compositional Rayleigh numbers (Silva, Mather & Simitev 2019, GAFD, eq. 18): with $R_t = Ra\cos\alpha$ and $R_c = Ra\sin\alpha$, $Ra_c(\alpha)\,(\cos\alpha + \sin\alpha)$ is the same for every $\alpha$. The test finds $Ra_c$ for $\alpha$ = 0 (temperature only), $\pi/2$ (composition only) and $\pi/4$ (both), with $Pr = Sc = 1$, Coriolis number $10^4$, `ricb = 0.35`, $m = 7$, stress-free boundaries and fixed temperature and composition, and checks that this sum and the drift frequency agree to $5\times10^{-5}$.


## No rotation

`tests/norotation`, run by `test_norotation.py`. With `rotation = 0` and the viscous time scale, the toroidal flow in a full sphere decays freely, with eigenvalues $\lambda = -k^2$, where $k$ is the smallest root of $j_l(k) = 0$ for a no-slip boundary or of $k\,j_l'(k) = j_l(k)$ for a stress-free one. The test checks $l$ = 2 and 3, both boundary conditions and $m$ = 0, 1 and 2, to $10^{-10}$; the decay rate must not depend on $m$.

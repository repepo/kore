# Background flow

With `bgflow = 1` the fluid carries a steady, axisymmetric background flow, and kore solves the waves and instabilities
of the fluid on top of it. A differential rotation (`diff_rot = 1`, [Background model](background.md#rotation)) is
the special case of a flow along the azimuth. The flow that turning walls drive also turns in the meridional planes
(up along the axis and back near the walls, or the reverse), and its rotation has a shape that no rotation profile of
`diff_rot` holds. That flow, written by kore from its own run of turning walls ([Spherical Couette
flow](couette-flow.md)), is the main use; a flow given as expressions of $r$ is the other.

## What it adds

The background flow carries the perturbations and is distorted by them. kore adds both effects to each equation it
solves: to the momentum equation the advection of the flow by the background and of the background by the flow, to the
heat and composition equations the advection of the entropy and the composition, and to the induction equation the
field that the background flow induces from the perturbed field. The background may also carry its own entropy and
composition, the ones that the flow brought about in the run that made it; the flow of the perturbation then carries
those too.

The background is

$$
\mathbf U = \mathrm{Ro}\thinspace\mathrm{Gaspard}\sum_n\left[\rho_0^{-1}\nabla\times\nabla\times(\rho_0 P_n Y_n\thinspace\mathbf r)
+ \nabla\times(T_n Y_n\thinspace\mathbf r)\right],
$$

with $Y_n$ the Legendre polynomial of degree $n$ in $\cos\theta$ and $\mathrm{Ro}$ = `bgflow_Ro`, the rate of the flow
in units of the rotation rate of the frame (with `Gaspard = 0`, a frame at rest, in units of $1/\tau$). It carries no mass across a sphere:
$\nabla\cdot(\rho_0\mathbf U) = 0$. Its entropy and composition are $\mathrm{Ro}\sum_n\sigma_n Y_n$ and
$\mathrm{Ro}\sum_n\gamma_n Y_n$, used with `thermal = 1` and `compositional = 1`. The radial functions $P_n$, $T_n$,
$\sigma_n$, $\gamma_n$ (per unit Ro) come from a file or from expressions. The terms are in
[Background flow equations](bgflow-equations.md).

It works with any `structure`, in a shell or a full sphere, with or without the diffusion of the entropy and of the
composition, without viscosity in a full sphere, with a differential rotation, a magnetic field, conducting solids,
the forcings, and in eigenvalue and forced problems. The background must

- be **symmetric about the equator**, as kore's perturbations are (`symm`): $P_n$, $\sigma_n$, $\gamma_n$ of even
  $n$ and $T_n$ of odd $n$;
- **not cross the walls**: $P_n = 0$ there;
- **turn rigidly at a conducting solid** (a conducting inner core, a thin conducting layer, a resolved mantle): $P_n$,
  $P_n'$ and every $T_n$ but $T_1$ vanish at its wall, and the solid turns with the fluid there, at
  $\mathrm{Ro}\thinspace\mathrm{Gaspard}\thinspace T_1/r$, so that its field is carried along.

With a background field (`magnetic = 1`), `submatrices.py` prints how far the field is from steady under the
background flow, $\lvert\nabla\times(\mathbf U\times\mathbf B_0)\rvert\thinspace r$ relative to
$\lvert\mathbf U\rvert\lvert\mathbf B_0\rvert$, and warns above $10^{-7}$ (it stops with `ferraro_policy = 'refuse'`):
the linear problem assumes that something maintains the background. A uniform axial field under a rotation about the
axis is steady; most other pairs are not. kore stops with `centrifugal = 1`, `rotdyn = 1` and `topo_drag = 'rekier'`.

## A background from turning walls

A forcing-12 run of kore gives the steady flow of walls that turn about the axis at their own rates. In its folder,

```shell
python3 bin/bgflow.py 12
```

writes `bgflow.npz` with the degrees $n \le 12$ of that flow, per unit Ro: the flow of the linear equations, which is
proportional to the rates. Ro is the rate of the inner wall over `Gaspard` (of the outer wall when the inner one is at
rest), and the other wall turns at the same fraction of it as in the run. Copy the file to the folder of the run that
uses it, with `bgflow = 1` and the Ro you want: any Ro gives the linear flow at that rate. The degrees of the run that
wrote it go up to its `lmax` and fall off quickly: in the example below the degrees above 12 move the eigenvalues by
about $10^{-4}$, those above 6 by $5\times10^{-3}$. Thinner layers, at lower Ek, need more.

The linear flow holds for small Ro. At larger Ro the flow carries itself along: it moves its own momentum, which the
linear equations leave out. With

```shell
python3 bin/bgflow.py 12 nonlinear
```

kore solves the full steady equations of the axisymmetric flow at the Ro of the run (`forcing_amplitude_icb` over
`Gaspard`) by Newton's method, starting from the linear flow and raising Ro in steps of 0.15. The file then holds that
Ro, and a run that reads it needs `bgflow_Ro` at that value: the flow at another Ro is not this one scaled. Each step
prints how far the iteration still moves, and it converges in a few iterations, the error squaring at each. The method
is on [Background flow equations](bgflow-equations.md#newtons-method).

The file also holds the density of the run (the poloidal flow is that density's), the `ricb`, and which wall turned.
kore stops when a run reads it with another `ricb`, another density, or (for a nonlinear file) another `bgflow_Ro`.
With entropy or composition (`thermal`, `compositional` = 1), the file holds the parts of the entropy and of the
composition that the flow brought about, $\sigma_n$ and $\gamma_n$. `bgflow.py` does not write the flow of a run with a
background field: the steady state of such a run has a field that the flow induces from $\mathbf B_0$, which the file
would leave out. The files that the `conductive_IC` branch writes are read as they are.

## Example: waves on the flow of an inner core that turns

The flow of [Spherical Couette flow](couette-flow.md#example-an-inner-core-that-turns) at Ek $10^{-3}$, nonlinear at Ro
= 1 (the inner core turning at twice the rate of the frame), and the waves of $m = 2$ on it. First the flow: write in
`bin/parameters.py`

```python
from params_default import default_params

par = default_params()

par.timescale = 'rotation'
par.Ek = 1e-3
par.set_scales()

par.structure = 'Boussinesq'
par.ricb = 0.35
par.bci = 1
par.bco = 1

par.m = 0
par.symm = 1
par.N = 48
par.lmax = 31
par.ncpus = 2

par.forcing = 12
par.forcing_frequency = 0.0
par.forcing_amplitude_icb = 1.0
par.forcing_amplitude_cmb = 0.0
```

and run

```shell
./bin/submatrices.py 2
mpiexec -n 2 ./bin/assemble.py
mpiexec -n 2 ./bin/solve.py
python3 bin/bgflow.py 12 nonlinear
```

Then, in a new folder with that `bgflow.npz`, the same parameters with

```python
par.m = 2
par.lmax = 29
par.forcing = 0
par.itau = 1.0

par.bgflow = 1
par.bgflow_Ro = 1.0
```

and run `submatrices.py`, `assemble.py`, `solve.py` and `spin_doctor.py`. The eigenvalue nearest $1i$ is
$-0.1818 + 1.1437i$ on the flow, and $-0.2687 + 0.9803i$ without it. The nonlinear flow differs from the linear flow at
Ro = 1 by 6%, and the linear flow gives $-0.1811 + 1.1491i$: at this rate the nonlinear part moves the eigenvalues by
about 0.005. The eigenvalues hold to 8 digits at N 64 and `lmax` 41, and the `conductive_IC` branch gives the same ones
from its own matrices (`tests/bgflow`). The whole example takes about two minutes on two cores, most of it in Newton's
method.

## A background from expressions

With `bgflow_profiles` the background is given as numpy expressions of $r$ (and `ricb`, with the functions of
`diff_rot_profiles`), one per radial function and degree, per unit Ro:

```python
par.bgflow = 1
par.bgflow_Ro = 0.3
par.bgflow_profiles = {'P2': '0.2*(r-ricb)**2*(1-r)**2',   # a meridional cell in each hemisphere
                       'T1': '0.5*r',                      # a rotation
                       'T3': '(r-ricb)*(1-r)*r',            # and its shear
                       'S2': '0.05*(r-ricb)*(1-r)'}         # an entropy of degree 2 (thermal = 1)
```

The keys are the kind (`P`, `T`, `S` for $\sigma_n$, `C` for $\gamma_n$) and the degree. `bgflow_file` is then not
read. This $P_2$ vanishes at both walls with its derivative, and $T_3$ vanishes there, so the flow is a rigid rotation
at each wall and the walls may be conducting. In a full sphere an expression is evaluated at $\lvert r\rvert$ with the
parity $(-1)^n$ of its degree; a smooth flow goes as $r^n$ times a series in $r^2$, as
`'P2': 'r**2*(1-r**2)**2'` ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)).

## Resolution

A background flow can make the waves much harder to resolve than the same waves in a fluid at rest, in two ways.

- **The flow strains the waves.** In a full sphere a meridional circulation of degree 2 is a straining flow at the
  centre, which presses the waves into thin layers there, where the Chebyshev points are furthest apart. The budgets
  and irreg₀ then fail, and they converge as N grows: with the circulation $P_2 = 2r^2(1-r^2)^2$ at Ro = 0.1 and Ek
  $10^{-3}$, residσ is $3\times10^{-1}$ at N 64, $3\times10^{-4}$ at N 96, $10^{-8}$ at N 128 and $10^{-12}$ at N
  160.
- **The flow carries what nothing smooths.** Without the diffusion of the entropy or of the composition, or without
  viscosity, the background carries the perturbation along its streamlines with nothing to limit how fine it gets.
  The frequencies of that transport, $m$ times the angular velocities of the flow and the rates of its meridional
  circulation, form a continuous band, and a target in that band gives modes that do not converge. Target outside it,
  and keep the modes that stay put when N and lmax are raised.

As with any run, compare two resolutions; `spin_doctor`'s residuals tell when the budget, and so the mode, is resolved
([Numerical method](numerical-method.md#checks)).

## Output

`spin_doctor.py` adds the power of the background's terms to the budgets: in `flow.dat` the power of its force on the
flow, Wbg, its imaginary part and its enstrophy term (columns 46–48), in `thermal.dat` and `composition.dat` the work
of its advection (column 8). They enter residσ, residω, residens, resid𝑠, resid𝑐 and resid𝐄. The power Wbg is what
the perturbation draws from the shear of the background: a growing mode takes its energy from there. The background
puts no torque in the fluid, so resid𝐋 holds as without it, axial and equatorial. `params.dat` gets `bgflow` and
`bgflow_Ro` (columns 155–156), and after the `#` the expressions of the background or the name of its file with the
first 12 hexadecimal digits of its sha256 ([Output files](output.md)).

## Parameters

| parameter | meaning |
| --- | --- |
| `bgflow` | 1 for a background flow |
| `bgflow_Ro` | its rate Ro, in units of the frame's rotation rate (of $1/\tau$ with `Gaspard = 0`); a nonlinear file holds at its own Ro only |
| `bgflow_file` | the file of the background, `'bgflow.npz'` by default, read when `bgflow_profiles` is empty |
| `bgflow_profiles` | the background as expressions, `{'P2': '...', 'T1': '...', ...}`, per unit Ro; no `#` in them (they are written to `params.dat`) |

`tests/bgflow` checks the background against the `conductive_IC` branch, its terms against a differential rotation, a
change of frame and a computation on a grid, and the budgets ([Modifying kore](checking.md)).

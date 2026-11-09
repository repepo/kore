# Background flow

With `bgflow = 1` the fluid carries a steady, axisymmetric background flow, and kore computes the waves and
instabilities on top of it. Differential rotation (`diff_rot = 1`, [Background model](background.md#rotation)) is the
special case of a purely azimuthal flow. A background flow can also circulate in the meridional plane (up along the
axis and back near the walls, or the reverse), and its rotation can have shapes that `diff_rot` cannot describe.

The main use is the flow driven by turning walls, computed by kore itself in a [Spherical Couette flow](couette-flow.md)
run. You can also give the flow as expressions of the radius.

## What it adds

The background flow carries the perturbations and is distorted by them. kore adds both effects to the momentum, heat,
composition and induction equations. The background may also have its own entropy and composition, which the
perturbation flow then carries too.

The background flow is written as

$$
\mathbf U = \mathrm{Ro}\thinspace\mathrm{Gaspard}\sum_n\left[\rho_0^{-1}\nabla\times\nabla\times(\rho_0 P_n Y_n\thinspace\mathbf r)
+ \nabla\times(T_n Y_n\thinspace\mathbf r)\right],
$$

where $Y_n$ is the Legendre polynomial of degree $n$ in $\cos\theta$, $P_n(r)$ and $T_n(r)$ are the poloidal and
toroidal radial functions, and Ro = `bgflow_Ro` is the strength of the flow relative to the rotation rate of the frame.
The equations are in [Background flow equations](bgflow-equations.md).

The background flow must:

- be **symmetric about the equator**;
- **not cross the walls**;
- **turn rigidly at a conducting solid** (a conducting inner core, thin conducting layer or resolved mantle), so that
  the solid can turn with the fluid.

It works with any structure, in a shell or a full sphere, with differential rotation, a magnetic field, conducting
solids, the constant drag at the outer wall, the forcings and the free decay. It does not work with the centrifugal
deformation, the rotating solids or the topographic drag.

With a magnetic field, the background field is in general not steady under the background flow. `submatrices.py`
prints how far it is from steady and warns when the mismatch is significant: the linear problem assumes that something
maintains the background.

## A background from turning walls

Run a [Spherical Couette flow](couette-flow.md) (forcing 12), then, in its folder,

```shell
python3 bin/bgflow.py 12
```

This writes `bgflow.npz`, holding the flow up to Legendre degree 12, normalized to Ro = 1. Copy it to the folder of
the run that uses it, and set `bgflow = 1` and the Ro you want. A maximum degree of about 12 is usually enough; thinner
boundary layers, at lower Ekman numbers, need more.

This *linear* flow is accurate only for small Ro. At larger Ro the flow carries its own momentum, which the linear
equations leave out. With

```shell
python3 bin/bgflow.py 12 nonlinear
```

kore solves the full steady equations of the axisymmetric flow at the Ro of the run (`forcing_amplitude_icb` over
`Gaspard`) by Newton's method ([Background flow equations](bgflow-equations.md#newtons-method)). The file is then valid
only at that Ro, and a run that reads it must set `bgflow_Ro` to that value.

The file also records the density, `ricb` and which wall turned, and kore stops if a run reads it with incompatible
parameters. With `thermal = 1` or `compositional = 1` it also holds the background entropy and composition that the
flow produced. `bgflow.py` does not support runs with a magnetic field.

## Example: waves on the flow of an inner core that turns

The flow of [Spherical Couette flow](couette-flow.md#example-an-inner-core-that-turns) at Ek $10^{-3}$, nonlinear at
Ro = 1 (the inner core turning at twice the rate of the frame), and the waves of $m = 2$ on it. First the flow: write in
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

Then, in a new folder with that `bgflow.npz`, use the same parameters with

```python
par.m = 2
par.lmax = 29
par.forcing = 0
par.itau = 1.0

par.bgflow = 1
par.bgflow_Ro = 1.0
```

and run `submatrices.py`, `assemble.py`, `solve.py` and `spin_doctor.py`. The eigenvalue nearest $1i$ is
$-0.1818 + 1.1437i$ with the flow, compared with $-0.2687 + 0.9803i$ without it. The example takes a couple of
minutes, most of it in Newton's method.

## A background from expressions

With `bgflow_profiles` the background is given as numpy expressions of `r` (and `ricb`), one per radial function and
degree, normalized to Ro = 1:

```python
par.bgflow = 1
par.bgflow_Ro = 0.3
par.bgflow_profiles = {'P2': '0.2*(r-ricb)**2*(1-r)**2',   # a meridional cell in each hemisphere
                       'T1': '0.5*r',                      # a rotation
                       'T3': '(r-ricb)*(1-r)*r',            # and its shear
                       'S2': '0.05*(r-ricb)*(1-r)'}         # an entropy of degree 2 (thermal = 1)
```

Each key is a letter and a degree: `P` and `T` for the poloidal and toroidal flow, `S` for the entropy, `C` for the
composition. Poloidal, entropy and composition terms take even degrees, toroidal terms odd degrees. In this example the
flow does not cross the walls and turns rigidly there, so the walls may be conducting. In a full sphere the
expressions must be smooth at the centre ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)).

## Resolution

A background flow can make the waves much harder to resolve than in a fluid at rest:

- **The flow strains the waves.** A meridional circulation can squeeze the waves into thin layers, for instance at
  the centre of a full sphere, and the radial resolution `N` must be raised until the `spin_doctor` residuals are small.
- **Without diffusion, the flow creates ever finer structure.** Without thermal or compositional diffusion, or without
  viscosity, the background carries the perturbation along its streamlines with nothing to smooth it. This gives a band
  of frequencies where modes do not converge. Put the target outside that band, and keep only the modes that stay
  put when `N` and `lmax` are raised. For forced runs without diffusion kore stops in such cases
  ([Combining features](compatibility.md#forced-runs)).

As with any run, compare two resolutions and check `spin_doctor`'s residuals
([Numerical method](numerical-method.md#checks)).

## Output

`spin_doctor.py` adds the power exchanged with the background flow to the budgets: in `flow.dat` the power that the
perturbation draws from the background, Wbg (columns 46–48), and in `thermal.dat` and `composition.dat` the work of its
advection (column 8). A growing mode takes its energy from there. `params.dat` records `bgflow` and `bgflow_Ro`, and the
expressions or the name of the file ([Output files](output.md)).

## Parameters

| parameter | meaning |
| --- | --- |
| `bgflow` | 1 for a background flow |
| `bgflow_Ro` | its strength Ro, relative to the rotation rate of the frame; a nonlinear file is valid only at its own Ro |
| `bgflow_file` | the file of the background, `'bgflow.npz'` by default, read when `bgflow_profiles` is empty |
| `bgflow_profiles` | the background as expressions, `{'P2': '...', 'T1': '...', ...}`, normalized to Ro = 1; no `#` in them |

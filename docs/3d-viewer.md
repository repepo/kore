# 3D viewer

The [3D viewer](viewer/index.html) shows a kore solution in your web browser. You can rotate and zoom it, cut it with a
sphere, a meridional plane and the equatorial plane, pick a field and a component, and play the oscillation in time.
It runs entirely in the browser: nothing to install, and solution files never leave your computer unless you put them
online yourself.

## Quick start

In the run folder, after `solve.py` (and `spin_doctor.py` to see which solution you want):

```
python3 koreviz/export_web.py --solnum 3
```

This writes `kore_solution.json`. Open the [viewer](viewer/index.html) and load the file with **Open a file…**, or drag
it onto the view.

## Writing the file

`koreviz/export_web.py` reads `bin/parameters.py` and the solution files of the folder it runs in, and needs only numpy
and scipy. It exports every field the run solves for: the velocity `u`, the magnetic field `b`, the temperature (or
entropy) and the composition. With a conducting inner core (`innercore = 'conducting, Chebys'`) the field inside the
core is included too.

- `--solnum k`: the solution to export, the row index printed by `spin_doctor.py` (default 0).
- `--fields u,b,t,c`: which fields to export, among velocity, magnetic field, temperature and composition (default
  every field of the run).
- `--nr`, `--ntheta`: the radial and latitudinal points of the grid (default N, at most 200, and about 1.6 lmax).
- `--nr_ic`: the radial points inside a conducting inner core (default 48).
- `--title`: the title shown by the viewer (default the folder name).
- `--out`: the file to write (default `kore_solution.json`).

A file holds about 1 MB per vector field at lmax 120. For thin structures (shear layers, boundary layers) raise
`--ntheta` and `--nr`; for a light file to share, lower them.

Since kore solves one azimuthal wavenumber $m$ at a time, every component is

$$
f(r,\theta,\phi,t) = \mathrm{Re}\left[F(r,\theta)\, e^{i(m\phi + \omega t)}\right],
$$

with the growth or decay $e^{\sigma t}$ left out. The file holds the complex amplitude $F$ of each component on an
$(r,\theta)$ grid, and the viewer computes every longitude, every time and the cylindrical components $s$ and $z$ from
it exactly. The scalars, Schmidt normalisation and phase convention are those of `koreviz/kmode.py`, so a phase of $90°$
in the viewer is the solution multiplied by $i$.

## Using the viewer

- **Field and component.** For a vector field: $r$, $\theta$, $\phi$, cylindrical $s$ and $z$, or the magnitude.
  For the temperature and the composition: the value.
- **Time.** Play runs one period every few seconds; the slider sets the phase $\omega t$ by hand. Every field of a
  solution shares the same phase, so switching fields shows them at the same instant.
- **Spherical surface.** A sphere of any radius, with a wedge cut away to look inside and an adjustable opacity.
- **Meridional plane** at any longitude, the **equatorial plane**, and the **inner-core surface**.
- **Colour.** The scale is symmetric about zero (from zero for the magnitude) and set by the largest value in the whole
  volume; **Scale to visible cuts** uses the cuts shown instead, and **Saturation** brings out weak features.
- **Mouse.** Drag to rotate, scroll or pinch to zoom, right-drag to pan. The value under the pointer is shown in the
  corner with its $r$, colatitude and longitude.

The header shows the eigenvalue $\lambda = \sigma + i\omega$, the period $2\pi/\omega$ in the run's time unit, and the
main parameters.

## Sharing a solution

The viewer is part of this site, so a solution file put anywhere online can be opened with a link:

```
https://repepo.github.io/kore/viewer/?data=https://example.org/my_mode.json
```

The server holding the file has to allow requests from other sites (GitHub Pages and most file hosts do). Files placed
in `docs/viewer/` of the repository are published with the site and can be given by name, `?data=my_mode.json`. A file
`docs/viewer/samples.json` listing some of them,

```
[{"file": "attractor.json", "name": "Inertial-wave attractor, Ek 1e-7"},
 {"file": "my_mode.json", "name": "My mode"}]
```

adds a menu of examples to the viewer, which then opens on the first one. Each file adds a few megabytes to the
repository, so keep the examples few and their grids coarse.

The viewer also works offline from a copy of the repository: open `docs/viewer/index.html` in a browser and load the
file with **Open a file…** (the browser needs the internet once to fetch the three.js graphics library).

## Limitations

- Only the solution is shown, not the background state (the imposed field $B_0$, the background temperature or
  composition, a differential rotation).
- Nothing is drawn outside the outer boundary (no potential field, no resolved mantle), and inside an insulating or
  `'conducting, Bessel'` inner core only a grey sphere is drawn.

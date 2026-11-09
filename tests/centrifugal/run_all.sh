#!/bin/bash
# Checks of the centrifugal deformation (centrifugal = 1; docs/centrifugal-equations.md, Checks). NEW (2026-11-05).
# usage: tests/centrifugal/run_all.sh [WORKDIR]   (default tests/centrifugal/runs)
# Needs numpy, scipy, sympy and mpi4py; the budgets also petsc4py and slepc4py (SKIP_SOLVE=1 skips them). With OLD_KORE set to
# a copy of the commit before the change (git archive <commit> | tar -x -C $OLD_KORE), also tests/diff_rot/regress.sh, which
# fails in the runs that a later commit changed on purpose: against a checkout older than phase 6c of the conductive_IC port
# (2026-11-07), m_gen, m_fer and m_fdr (the push of the poloidal current sheet at their stress-free walls) and m_wnd (by 8e-13
# of the row scale: the round-off of the wound field's profiles, now cut).
# NEW (2026-11-09): the checks are independent jobs, NPAR at a time (default: the number of processors, and at most one job
# per 3 GB of available memory; a solve takes 2 ranks, shape_check.py up to 11 processes, derive_mag.py up to 16), the
# longest first. Each job runs in a shell with -e and pipefail (a script that fails in a pipe fails its check) and writes
# WORKDIR/logs/NAME.log; once all have finished the logs are printed in the order below, each with its time, then the
# slowest jobs. 55 minutes of processor time in all (a container with 4 processors, 2026-11-09: 25 minutes with NPAR = 4,
# the longest jobs derive_mag 22 min, spheroid_exact 19, eig 16, bm_gen 9, shape 8, each shared with the others).
# Each check prints its verdict; the script ends with 'centrifugal checks ok' (exit 0) or lists the ones that failed (exit 1).
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"; W=$PWD
if [ -z "$NPAR" ]; then   # (the processors this process may use)
  NPAR=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 2)
  MEMG=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo 2> /dev/null)
  [ -n "$MEMG" ] && [ $((MEMG/3)) -lt "$NPAR" ] && NPAR=$((MEMG/3))
fi
[ "$NPAR" -ge 1 ] 2> /dev/null || NPAR=1
# Open MPI binds the ranks of every mpiexec to the same first cores: runs side by side would share two cores. With more
# ranks than cores, a rank that waits gives its core away instead of polling
export OMPI_MCA_hwloc_base_binding_policy=${OMPI_MCA_hwloc_base_binding_policy:-none}
export OMPI_MCA_mpi_yield_when_idle=${OMPI_MCA_mpi_yield_when_idle:-1}
# and with jobs side by side, one thread per process: numpy's BLAS would otherwise start a thread per core in each
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
L=$W/logs; rm -rf "$L"; mkdir -p "$L"

# the longest jobs, started first and in this order
FIRST="derive_mag spheroid_exact eig bm_gen shape derive_diff bm_fs bm_wu spheroid bud_wu_dr bms_fs"
first() { case " $FIRST " in *" $1 "*) return 0;; esac; return 1; }
# on INT or TERM, every process the jobs started (a kill of the jobs alone would leave their children running)
ptree() { local p; echo "$1"; for p in $(pgrep -P "$1"); do ptree "$p"; done; }
trap 'kill $(for j in $(jobs -p); do ptree $j; done) 2> /dev/null; exit 130' INT TERM
ORDER=()
say() { [ "$PASS" = order ] && ORDER+=("#$1"); return 0; }
job() {   # job NAME CMD ARGS..: CMD ARGS in the background in a shell with -e and pipefail, at most NPAR jobs at a time;
          # its output in logs/NAME.log, its exit code and seconds in logs/NAME.rc
  local n=$1; shift
  case $PASS in
    order) ORDER+=("$n"); return 0;;
    rest)  first "$n" && return 0;;
    *)     [ "$n" = "$PASS" ] || return 0;;   # PASS: a name of FIRST
  esac
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 0.2; done
  ( s=$SECONDS; ( set -e -o pipefail; "$@" ) > "$L/$n.log" 2>&1; echo "$? $((SECONDS - s))" > "$L/$n.rc" ) &
}
last() { "$@" | tail -1; }
mk() {    # mk NAME CFG [solve]: builds the run NAME (its output in NAME.mk.log, shown if it fails)
  python3 "$T/mk_run.py" "$@" > "$1.mk.log" 2>&1 || { tail -20 "$1.mk.log"; echo "$1: mk_run.py failed"; return 1; }
}
ru() { mk "$1" "$2"; (cd "$1" && python3 bin/ru_insane.py | grep 'centrifugal\|PASS\|FAIL' | sed "s/^/$1: /"); }
bud() {   # bud NAME CFG [ARGS of budget_check.py]
  local d=$1 c=$2; shift 2
  mk "$d" "$c" solve; (cd "$d" && python3 "$T/budget_check.py" "$@" | sed "s/^/$d: /")
}
bm() {    # bm NAME CFG: built at f = 0.005 and 0.01 (@F@ in CFG), budget_check.py on the pair
  local d=$1 c=$2 F
  for F in 0.005 0.01; do mk ${d}_$F "${c//@F@/$F}" solve; done
  (cd ${d}_0.01 && python3 "$T/budget_check.py" 1e-8 ../${d}_0.005 | sed "s/^/$d: /")
}

WU="par.structure = 'Wu 2005'; par.r_out = 0.99; par.thermal = 1; par.stratification = 'top hat'; par.aux1 = 0.5; par.aux2 = 0.03; par.aux3 = 0.85; par.aux4 = 0.03; par.aux5 = 2.5; par.Beyonce = 10."
CL="par.centrifugal_shape = 'clairaut'; par.centrifugal_f = None; par.centrifugal_q = 0.05"
US="par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = 'p*(1 + r**2 - 0.5*r**4)'; par.centrifugal_param = 0.02"
BT="par.thermal = 1; par.stratification = 'linear'"
VI="par.ViscosD = 1e-3; par.ThermaD = 1e-3"
DR="par.diff_rot = 1; par.diff_rot_type = 'conical'; par.diff_rot_ampl = -0.25; par.thermal_wind = 1"
SP="par.centrifugal_shape = 'spheroid'; par.centrifugal_f = 0.3"     # NEW (2026-11-06): the exact spheroid
BUD="$WU; $VI; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 40; par.ncpus = 2; par.nev = 4; par.itau = 0.3"
SPB="$BT; par.ViscosD = 1e-2; par.ThermaD = 1e-2; par.Beyonce = 2.; $SP; par.ncpus = 2; par.nev = 4; par.itau = 0.5"   # NEW (2026-11-06)
MAG="par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 'mag_energy'; par.Ek = 1e-3; par.Em = 1e-3; par.Le = np.sqrt(1e-3); par.set_scales(); par.Gaspard = 1.; par.Beyonce = 1.; par.ricb = 0.35; par.N = 96; par.lmax = 24; par.ncpus = 2; par.nev = 4; par.itau = 1.0; par.prescale = None"
UP="par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = 'p*(1 + r**2)/2'; par.centrifugal_param = @F@"
GEN="par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.05*r**2', 2: '-0.1*r**2', 4: '-0.03*r**4'}; par.lmax = 32; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(1 - r**2)'}; par.cnorm_tor = 'mag_energy'"
SPM="$MAG; $SP; par.lmax = 40"
GEN0="par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(1 - r**2)'}; par.cnorm_tor = 'mag_energy'"

checks() {
say "--- derive.py: the closed forms of bin/centrifugal.py from X = α + ξ, w = J F⁻¹ u (sympy)"
job derive last python3 "$T/derive.py"
say "--- algebra_check.py: the angular couplings (angular.py's algebra) against a direct projection of the closed forms"
job algebra last python3 "$T/algebra_check.py"
say "--- derive_mag.py: the magnetic terms of the deformed body from b = F b̃/J (sympy), and kore's couplings against them"
job derive_mag last python3 "$T/derive_mag.py"
# NEW (2026-11-06, phase 2)
say "--- derive_diff.py: the viscous force, the diffusion, the stress-free and fixed-flux walls and the differential rotation
    of the deformed body (sympy), and kore's couplings and wall rows against them"
job derive_diff last python3 "$T/derive_diff.py"
say "--- bc_check.py: the insulating-wall rows on the deformed wall against the exact matching of a potential field"
job bc last python3 "$T/bc_check.py"
say "--- clairaut_check.py: the 'clairaut' flattening (Maclaurin 5/4, polytrope 7.5/π², Wu 2005 against Clairaut from ρ, the Sun)"
job clairaut python3 "$T/clairaut_check.py"
say "--- eig_check.py: f = 0, the spin-over 1 + f, the relabelling test, homogeneous 'clairaut', the monopole, the ideal fluid,
    f = 0 with a field, the relabelling test with the axial and the Malkus field"
job eig python3 "$T/eig_check.py"
# NEW (2026-11-06, phase 2)
say "--- spheroid_modes.py: the inertial modes of a spheroid, exact (polynomial method), against kore's dω/df (73 modes)"
job spheroid python3 "$T/spheroid_modes.py"
say "--- shape_check.py: damped modes when the walls are deformed (no-slip, stress-free, heat), exactly at Ek 1e-3"
job shape python3 "$T/shape_check.py" $(( NPAR < 11 ? NPAR : 11 ))
# NEW (2026-11-06, the exact spheroid)
say "--- spheroid_check.py: the exact spheroid ('spheroid'): its terms from the map, kore's couplings and wall rows against
    their projections, the inviscid, viscous and heat modes against exact solutions, the rigid rotation; and (step 2)
    with a field the insulating walls on exact potential fields, the free decay against an exact solution, the inner
    cores, and the first-order code parting as f²"
job spheroid_exact python3 "$T/spheroid_check.py"

say "--- ru_insane.py: the metric blocks against (r, θ) quadratures, Hermitian and positive, in shells and full spheres"
job ru_wu_m1   ru ru_wu_m1   "$WU; $VI; $CL; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 24"
job ru_wu_m0   ru ru_wu_m0   "$WU; $VI; $CL; par.ricb = 0.35; par.m = 0; par.symm = -1; par.N = 96; par.lmax = 25"
job ru_wu_dr   ru ru_wu_dr   "$WU; $VI; $CL; $DR; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 24"
job ru_bq_m2   ru ru_bq_m2   "$BT; $VI; $US; par.ricb = 0.35; par.m = 2; par.symm = -1; par.N = 48; par.lmax = 21"
job ru_bq_fs1  ru ru_bq_fs1  "$BT; $VI; $US; par.N = 48; par.rpower_v = 4"
job ru_bq_fs2  ru ru_bq_fs2  "$BT; $VI; $US; par.m = 2; par.symm = 1; par.N = 48; par.lmax = 21; par.rpower_v = 4"
job ru_bq_td0  ru ru_bq_td0  "$BT; par.ViscosD = 1e-3; par.ThermaD = 0; $US; par.symm = 1; par.N = 48; par.rpower_v = 4"
job ru_sph_m1  ru ru_sph_m1  "$BT; $VI; $SP; par.ricb = 0.35; par.symm = 1; par.N = 48; par.lmax = 40"
job ru_sph_fs2 ru ru_sph_fs2 "$BT; $VI; $SP; par.m = 2; par.symm = 1; par.N = 40; par.lmax = 41; par.rpower_v = 4"

if [ -n "$OLD_KORE" ]; then   # (builds and assembles only: it runs with SKIP_SOLVE too)
  say "--- regress.sh: centrifugal = 0 leaves the matrices of the 26 regression runs unchanged"
  job regress regress
fi

[ -n "$SKIP_SOLVE" ] && return 0
say "--- budget_check.py: spin_doctor's budgets with the metric weights of KE, the Coriolis power and TE (solve.py, 2 ranks)"
job bud_wu_sphere bud bud_wu_sphere "$BUD; par.centrifugal = 0"
job bud_wu_cl     bud bud_wu_cl     "$BUD; $CL"
job bud_wu_dr     bud bud_wu_dr     "$BUD; $CL; $DR"
job bud_bq_fs     bud bud_bq_fs     "$BT; $VI; $US; par.m = 2; par.symm = 1; par.N = 64; par.lmax = 41; par.ncpus = 2; par.nev = 4; par.itau = 0.5; par.rpower_v = 4"
job bud_sph_ns    bud bud_sph_ns    "$SPB; par.ricb = 0.35; par.symm = 1; par.N = 48; par.lmax = 40"
job bud_sph_sf    bud bud_sph_sf    "$SPB; par.ricb = 0.35; par.bco = 0; par.bci = 0; par.bco_thermal = 1; par.m = 2; par.symm = -1; par.N = 48; par.lmax = 41"
job bud_sph_fs    bud bud_sph_fs    "$SPB; par.m = 2; par.symm = 1; par.N = 40; par.lmax = 41; par.rpower_v = 4"
# NEW (2026-11-05, phase 1b): with a field, at f and f/2 (@F@): residσ, residω, resid𝐛 to round-off, resid𝐄 of second order
say "--- budget_check.py with a field: residσ, residω and resid𝐛 to round-off, resid𝐄 of second order in f (solve.py, 2 ranks)"
job bm_uniform bm bm_uniform "$MAG; par.centrifugal_f = @F@"
job bm_cic     bm bm_cic     "$MAG; $UP; par.innercore = 'conducting, Chebys'"
job bm_fs      bm bm_fs      "$MAG; $UP; par.ricb = 0; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6; par.rpower_v = 4"
job bm_tor     bm bm_tor     "$MAG; $UP; par.B0 = None; par.B0_tor = 'Malkus'; par.cnorm_tor = 'mag_energy'"
job bm_gen     bm bm_gen     "$MAG; $UP; $GEN"
# NEW (2026-11-06, step 2): the exact spheroid with a field (f = 0.3, -0.2 for the conducting inner core and the m = 2 case):
# every budget to round-off, the energy budget resid𝐄 too (an insulating and a conducting inner core, a full sphere with
# Luo_S2, the Malkus field, general poloidal and toroidal fields)
say "--- budget_check.py in the exact spheroid with a field: residσ, residω, resid𝐛 and resid𝐄 to round-off (solve.py, 2 ranks)"
job bms_ins bud bms_ins "$SPM" 1e-9 exact
job bms_cic bud bms_cic "$SPM; par.innercore = 'conducting, Chebys'; par.sigma_i2o = 0.3; par.centrifugal_f = -0.2" 1e-9 exact
job bms_fs  bud bms_fs  "$SPM; par.ricb = 0; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6; par.rpower_v = 4; par.lmax = 48" 1e-9 exact
job bms_tor bud bms_tor "$SPM; par.B0 = None; par.B0_tor = 'Malkus'; par.cnorm_tor = 'mag_energy'" 1e-9 exact
job bms_gen bud bms_gen "$SPM; $GEN0; par.centrifugal_f = -0.2; par.m = 2; par.symm = -1; par.lmax = 41" 1e-9 exact
# the anelastic 'Wu 2005' with a field needs N 128 for its budgets (the sphere's are 2e-6 at N 96): closure only
job bm_wu bud bm_wu "$MAG; $WU; par.thermal = 0; par.Beyonce = 1.; $CL; par.N = 128" 1e-7
}
regress() { bash "$T/../diff_rot/regress.sh" "$OLD_KORE" "$W/regress" | grep -v ': ok$'; }

T0=$SECONDS
PASS=order; checks
for PASS in $FIRST; do checks; done
PASS=rest; checks
wait

failed=()
status() { local rc t; read rc t < "$L/$1.rc" 2>/dev/null || rc=missing; [ "$rc" = 0 ] && ! grep -q "FAIL" "$L/$1.log"; }
for x in "${ORDER[@]}"; do
  case $x in
    '#'*) echo "${x#?}";;
    *)    cat "$L/$x.log"; status "$x" || { echo "$x: FAILED (exit $(cut -d' ' -f1 "$L/$x.rc" 2>/dev/null))"; failed+=("$x"); }
          echo "    ($x: $(cut -d' ' -f2 "$L/$x.rc" 2>/dev/null) s)";;
  esac
done
echo
echo "slowest jobs: $(for f in "$L"/*.rc; do echo "$(cut -d' ' -f2 "$f") $(basename "$f" .rc)"; done | sort -rn | head -8 | awk '{printf "%s%s %d s", (NR > 1 ? ", " : ""), $2, $1}')"
if [ ${#failed[@]} -eq 0 ]; then echo "centrifugal checks ok ($((SECONDS - T0)) s with NPAR = $NPAR)"
else echo "centrifugal checks FAILED: ${failed[*]} ($((SECONDS - T0)) s with NPAR = $NPAR)"; exit 1; fi

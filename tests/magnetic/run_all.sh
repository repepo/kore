#!/bin/bash
# Checks of the magnetic field with a variable density (docs/magnetic-equations.md, Checks). usage: tests/magnetic/run_all.sh [WORKDIR]
# (default tests/magnetic/runs; needs numpy, scipy, sympy, mpi4py, petsc4py, slepc4py). SKIP_DERIVE=1 skips the sympy checks.
# NEW (2026-11-09): the checks are independent jobs, NPAR at a time (default: the number of processors, and at most one job
# per 3 GB of available memory; a solve takes 2 ranks, malkus_check.py 4), the longest first; the sympy checks run in parts
# (--part k/K). Each job writes WORKDIR/logs/NAME.log, and once all have finished the logs are printed in the order below,
# then the slowest jobs. Two hours of jobs in all: 31 minutes with NPAR = 4. Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"; W=$PWD
if [ -z "$NPAR" ]; then   # (the processors this process may use)
  NPAR=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 2)
  MEMG=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo 2> /dev/null)   # (a solve at N 192 takes up to 2.4 GB a rank)
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
T0=$SECONDS

# the longest jobs (measured 2026-11-09, NPAR = 4), started first and in this order; the parts of a sympy check go by its name
FIRST="derive_check algebra_check rc2_comb_m2 bgen bfer bfdr malkus bdr_pol rcf_inv_s2 rc_f2_m2 bfer_tor rc_s2_m1 rcf_ax_m1 rigid rc_ax_m1 rcf_mal_m2"
first() { case " $FIRST " in *" ${1%%.*} "*) return 0;; esac; return 1; }
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
    *)     [ "${n%%.*}" = "$PASS" ] || return 0;;   # PASS: a name of FIRST
  esac
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 0.2; done
  ( s=$SECONDS; ( set -e -o pipefail; "$@" ) > "$L/$n.log" 2>&1; echo "$? $((SECONDS - s))" > "$L/$n.rc" ) &
}
parts() { # parts NAME K CMD ARGS..: K jobs NAME.k, each CMD ARGS --part k/K, reported as one line
  local n=$1 K=$2 k; shift 2
  [ "$PASS" = order ] && { ORDER+=("=$n $K"); return 0; }
  for ((k = 0; k < K; k++)); do job "$n.$k" "$@" --part "$k/$K"; done
}
ru()  { python3 "$T/mk_run.py" "$1" "$2"; (cd "$1" && python3 bin/ru_insane.py | grep "magnetic\|PASS\|FAIL" | sed "s/^/$1: /"); }
row() { python3 "$T/mk_run.py" "$1" "$2"; (cd "$1" && python3 "$T/row_check.py" | tail -1 | sed "s/^/$1: /"); }
bud() { python3 "$T/mk_run.py" "$1" "$2" solve; (cd "$1" && python3 "$T/budget_check.py" | tail -1 | sed "s/^/$1: /"); }
conf_fs() {   # NEW (2026-11-03): 'confined' with r1 > 0 in a full sphere (h at |r| with its parity): no parity warnings, and its budgets
  python3 "$T/mk_run.py" bconf_fs "$1" solve
  if grep -q "assumed parity" bconf_fs/log_sub; then echo "bconf_fs: parity warnings -> FAIL"; return 1; fi
  echo "bconf_fs: no parity warnings -> ok"
  (cd bconf_fs && python3 "$T/budget_check.py" | tail -1 | sed "s/^/bconf_fs: /")
}
rigid() { python3 "$T/rigid_check.py" | tail -7; }

checks() {
say "--- derive_check.py: the rows from the definitions (sympy) against conductive_IC's operators and the closed forms (SKIP_DERIVE=1 skips it)"
[ -z "$SKIP_DERIVE" ] && parts derive_check.py "$NPAR" python3 "$T/derive_check.py"
say "--- ru_insane.py: operator symmetries and the Lorentz/induction energy identity (B0 'axial', no-slip, insulating walls)"
job ru_uni    ru ru_uni    "par.structure = 'Boussinesq'"
job ru_wu_m1  ru ru_wu_m1  ""
job ru_wu_m0  ru ru_wu_m0  "par.m = 0; par.symm = 1; par.lmax = 23"
job ru_wu_m2  ru ru_wu_m2  "par.m = 2; par.symm = 1; par.lmax = 25"
job ru_wu_cic ru ru_wu_cic "par.B0 = 'FDM'; par.innercore = 'conducting, Chebys'"
say "--- row_check.py: every magnetic row block against derive.py ('Wu 2005' needs N = 192 for round-off: the pole of (ln rho0)' sits at r = 1/r_out)"
job rc_ax_m1  row rc_ax_m1  "par.N = 192"
job rc_fdm_m0 row rc_fdm_m0 "par.N = 192; par.B0 = 'FDM'; par.m = 0; par.symm = 1; par.lmax = 23; par.tol_B0 = 1e-15"   # the FDM series uncut: 1e-9 with tol_B0 = 1e-8
say "--- budget_check.py: spin_doctor's budgets with variable rho0 (FDM, conducting inner core, Ek 1e-3, N 192)"
job bud_m1 bud bud_m1 "par.N = 192; par.lmax = 48; par.B0 = 'FDM'; par.cnorm = 'Schmitt2012'; par.innercore = 'conducting, Chebys'"
job bud_m0 bud bud_m0 "par.N = 192; par.lmax = 47; par.B0 = 'FDM'; par.cnorm = 'Schmitt2012'; par.innercore = 'conducting, Chebys'; par.m = 0; par.symm = 1; par.itau = 0.3"
say "--- toroidal B0 (B0_tor 'Malkus' or 'wall', degree 1 or 2, alone or with 'axial'): row checks, budgets ('Wu 2005', N 192) and the Malkus (1967) theory"
job rc_mal_m1   row rc_mal_m1   "par.N = 192; par.B0 = None; par.B0_tor = 'Malkus'"
job rc_wall_m0  row rc_wall_m0  "par.N = 192; par.B0 = None; par.B0_tor = 'wall'; par.m = 0; par.symm = 1; par.lmax = 23"
job rc2_mal_m1  row rc2_mal_m1  "par.N = 192; par.B0 = None; par.B0_tor = 'Malkus'; par.B0_tor_l = 2"
job rc2_comb_m2 row rc2_comb_m2 "par.N = 192; par.B0_tor = 'Malkus'; par.B0_tor_l = 2; par.m = 2; par.symm = 1; par.lmax = 25"
job tb_mal_m1   bud tb_mal_m1   "par.N = 192; par.lmax = 48; par.B0 = None; par.B0_tor = 'Malkus'; par.cnorm_tor = 'mag_energy'; par.innercore = 'conducting, Chebys'"
job tb_wall_m0  bud tb_wall_m0  "par.N = 192; par.lmax = 47; par.B0 = None; par.B0_tor = 'wall'; par.cnorm_tor = 'mag_energy'; par.m = 0; par.symm = 1; par.itau = 0.3"
job tb2_wall_m1 bud tb2_wall_m1 "par.N = 192; par.lmax = 48; par.B0 = None; par.B0_tor = 'wall'; par.B0_tor_l = 2; par.cnorm_tor = 'mag_energy'; par.innercore = 'conducting, Chebys'"
job malkus python3 "$T/malkus_check.py"
say "--- NEW (2026-10-28): full sphere (ricb = 0) and the degree-2 poloidal B0 (Luo_S2, FDM with B0_l = 2): row checks and budgets"
FS="par.ricb = 0; par.innercore = 'insulator'"
job rcf_ax_m1  row rcf_ax_m1  "par.N = 288; $FS"   # (the pole of (ln rho0)' at r = 1/r_out is nearer to the full-sphere domain [-1, 1])
job rcf_s2_m0  row rcf_s2_m0  "par.N = 288; $FS; par.B0 = 'Luo_S2'; par.m = 0; par.symm = -1; par.lmax = 23"
job rcf_mal_m2 row rcf_mal_m2 "par.N = 288; $FS; par.B0 = None; par.B0_tor = 'Malkus'; par.m = 2; par.symm = 1; par.lmax = 25"
job rcf_inv_s2 row rcf_inv_s2 "par.N = 96; $FS; par.structure = 'Boussinesq'; par.Ek = 0; par.ViscosD = 0; par.bci = 0; par.bco = 0; par.B0 = 'Luo_S2'; par.m = 1; par.symm = 1; par.lmax = 24"
job rc_s2_m1   row rc_s2_m1   "par.N = 192; par.B0 = 'Luo_S2'; par.tol_B0 = 1e-15"   # (2e-8 with the default tol_B0: the B0_ic r^-3 term)
job rc_f2_m2   row rc_f2_m2   "par.N = 192; par.B0 = 'FDM'; par.B0_l = 2; par.m = 2; par.symm = -1; par.lmax = 25; par.tol_B0 = 1e-15"
job bf_s2_m1 bud bf_s2_m1 "par.N = 192; par.lmax = 48; $FS; par.B0 = 'Luo_S2'"
job bf_ax_m0 bud bf_ax_m0 "par.N = 96; par.lmax = 47; $FS; par.structure = 'Boussinesq'; par.m = 0; par.symm = 1; par.itau = 0.3"   # ('Wu 2005': 6e-7 at N 192, the Hartmann layer of a uniform field)
job b_s2_cic bud b_s2_cic "par.N = 192; par.lmax = 48; par.B0 = 'Luo_S2'; par.innercore = 'conducting, Chebys'"
say "--- NEW (2026-10-30): the magnetic field with a differential rotation, the confined, general and ferraro fields, the coupling algebra"
if [ -z "$SKIP_DERIVE" ]; then
  parts dr_induction_check.py $(( (NPAR + 7)/8 )) python3 "$T/dr_induction_check.py" --quick   # ∇×(U0×b) of diff_rot.py against derive.py (sympy)
  parts algebra_check.py $(( (NPAR + 1)/2 )) python3 "$T/algebra_check.py" --quick             # mag_algebra.py against derive.py, degrees 1 and 3
fi
job rigid rigid                                         # ΔΩ = c is a rigid rotation: λ = λ(Gaspard 1 + c) - i m c exactly
DR="par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.05*r**2', 2: '-0.1*r**2', 4: '-0.03*r**4'}; par.lmax = 32"
job bdr_tor  bud bdr_tor  "par.N = 96; $DR; par.B0 = None; par.B0_tor = 'Malkus'"
job bdr_pol  bud bdr_pol  "par.N = 192; $DR; par.B0 = 'axial'"
job bdr_fs   bud bdr_fs   "par.N = 192; $DR; $FS; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6"
job bconf    bud bconf    "par.N = 128; par.structure = 'Boussinesq'; par.B0 = 'confined'; par.B0_l = 2; par.B0_conf_r1 = 0.4; par.B0_conf_r2 = 0.8; par.symm = 1; par.itau = 0.6; par.lmax = 32"
job bconf_fs conf_fs "par.N = 192; $FS; par.structure = 'Boussinesq'; par.B0 = 'confined'; par.B0_l = 2; par.B0_conf_r1 = 0.3; par.B0_conf_r2 = 0.8; par.symm = 1; par.itau = 0.6; par.lmax = 32"
job bgen     bud bgen     "par.N = 192; $DR; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(1 - r**2)'}; par.cnorm_tor = 'mag_energy'"
job bfer     bud bfer     "par.N = 192; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 0.2; par.diff_rot_profiles = {0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', 4: '-(8/35)*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = 'w'; par.B0_ferraro_nmax = 7; par.lmax = 32"
job bfdr     bud bfdr     "par.N = 192; par.diff_rot = 1; par.diff_rot_type = 'ferraro'; par.diff_rot_F = 'psi + psi**2'; par.diff_rot_ampl = 0.05; par.diff_rot_nmax = 8; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.lmax = 32"   # ΔΩ = F(ψ) from the field
job bfer_tor bud bfer_tor "par.N = 192; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 0.2; par.diff_rot_profiles = {0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', 4: '-(8/35)*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = 'w'; par.B0_ferraro_nmax = 7; par.B0_tor = 'Malkus'; par.B0_tor_l = 2; par.cnorm_tor = 'mag_energy'; par.lmax = 32"   # with a toroidal part (degree 2: the same symmetry)
say "--- NEW (2026-10-30): B0_wound, the toroidal field wound by the differential rotation and balanced by diffusion"
job wound python3 "$T/wound_check.py"
job bwnd bud bwnd "par.N = 96; par.structure = 'Boussinesq'; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.05*r**2', 2: '-0.1*r**2', 4: '-0.03*r**4'}; par.B0_wound = 1; par.B0_tor = 'Malkus'; par.B0_tor_l = 2; par.cnorm_tor = 'mag_energy'; par.lmax = 32"   # max |B_phi| ~ 0.8 max |B_p|
say "--- continuity.py: 'Wu 2005' with r_out = sqrt(eps) -> 'Boussinesq' as eps -> 0"
job continuity python3 "$T/continuity.py" 1e-4 1e-3 1e-2 1e-1
}

PASS=order; checks
for PASS in $FIRST; do checks; done
PASS=rest; checks
wait

fail=0
status() { local rc t; read rc t < "$L/$1.rc" 2>/dev/null || rc=missing; [ "$rc" = 0 ] && ! grep -q "FAIL" "$L/$1.log"; }
for x in "${ORDER[@]}"; do
  case $x in
    '#'*) echo "${x#?}";;
    '='*) set -- ${x#?}; ok=1; n=0; what=
          for ((k = 0; k < $2; k++)); do
            if ! status "$1.$k"; then ok=0; cat "$L/$1.$k.log"; echo "$1 part $k/$2: FAILED"; fi
            line=$(sed -n 's/^.* part [0-9]*\/[0-9]*: \([0-9]*\) \([^,]*\),.*/\1 \2/p' "$L/$1.$k.log" | tail -1)
            [ -n "$line" ] && { n=$((n + ${line%% *})); what=${line#* }; }
          done
          echo "$1: $n $what ($2 parts), $([ $ok = 1 ] && echo ok || echo FAIL)"; [ $ok = 1 ] || fail=1;;
    *)    cat "$L/$x.log"; status "$x" || { echo "$x: FAILED (exit $(cut -d' ' -f1 "$L/$x.rc" 2>/dev/null))"; fail=1; };;
  esac
done
echo
echo "slowest jobs: $(for f in "$L"/*.rc; do echo "$(cut -d' ' -f2 "$f") $(basename "$f" .rc)"; done | sort -rn | head -8 | awk '{printf "%s%s %d s", (NR > 1 ? ", " : ""), $2, $1}')"
echo "tests/magnetic: $((SECONDS - T0)) s with NPAR = $NPAR, $([ $fail = 0 ] && echo "all checks passed" || echo FAILED)"
exit $fail

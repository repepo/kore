#!/bin/bash
# Rebuilds the test runs of the differential-rotation heat-equation terms and runs every check (docs/differential_rotation_details.md §9).
# usage: tests/heat_dr/run_all.sh [WORKDIR]   (default: tests/heat_dr/runs; needs numpy, scipy, sympy, mpi4py; ~10 minutes
#        serial; NEW 2026-11-09: the runs are built side by side, NPAR at a time, see below)
set -e -o pipefail  # a crash in a 'check | tail' or 'check | sed' pipeline stops the script
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
# NEW (2026-11-09): the runs (one rank each) are built side by side, NPAR at a time (default: the processors, at most one
# per 2 GB of available memory); bwait waits for them, prints their lines in the order they were started, and stops if a
# build failed. The checks then run in order, as before.
if [ -z "$NPAR" ]; then   # (the processors this process may use)
  NPAR=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 2)
  MEMG=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo 2> /dev/null)
  [ -n "$MEMG" ] && [ $((MEMG/2)) -lt "$NPAR" ] && NPAR=$((MEMG/2))
fi
[ "$NPAR" -ge 1 ] 2> /dev/null || NPAR=1
# and with jobs side by side, one thread per process: numpy's BLAS would otherwise start a thread per core in each
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# on INT or TERM, every process the builds started (a kill of the jobs alone would leave their children running)
ptree() { local p; echo "$1"; for p in $(pgrep -P "$1"); do ptree "$p"; done; }
trap 'kill $(for j in $(jobs -p); do ptree $j; done) 2> /dev/null || true; exit 130' INT TERM
BUILT=()
mk() {   # mk NAME CFG [quiet]: mk_run.py NAME CFG in the background, its output in NAME.mk (printed by bwait unless quiet)
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 0.2; done
  ( rc=0; python3 "$T/mk_run.py" "$1" "$2" > "$1.mk" 2>&1 || rc=$?; echo $rc > "$1.mkrc" ) &
  BUILT+=("$1:${3:-}")
}
bwait() {
  wait
  local e d q bad=0
  for e in "${BUILT[@]}"; do
    d=${e%%:*}; q=${e#*:}
    if [ "$(cat "$d.mkrc" 2>/dev/null)" != 0 ]; then bad=1; cat "$d.mk"; elif [ -z "$q" ]; then cat "$d.mk"; fi
    rm -f "$d.mk" "$d.mkrc"
  done
  BUILT=()
  [ $bad = 0 ] || { echo "tests/heat_dr: a build failed"; exit 1; }
}
SUN="par.ricb=0; par.structure='8-term fit'; par.stratification='odd top hat'; par.diffusivity='step'; par.N=120; par.lmax=30; par.m=1; par.symm=1; par.r_out=0.99; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=2.6; par.Beyonce=1e3; par.ViscosD=1e-3; par.ThermaD=1e-3; par.visc0=1.0; par.therm0=1.0; par.visc_pow=0.0; par.therm_pow=0.0; par.diff_rot_type='conical'; par.diff_rot_ampl=-0.25; par.bco=1; par.bco_thermal=0"
# 'solar' rotation in the full sphere: its tachocline tanh needs N ~ 400 for round-off (1e-2 at N = 120, 2e-8 at N = 240)
SOL="$SUN; par.N=400; par.lmax=14; par.diff_rot_type='solar'; par.diff_rot_ampl=1"
echo "--- sympy: closed forms of sigma2, sigma4, sigma6 (thermal_wind = 1 and 2) against the thermal-wind balance"
python3 "$T/tw_sympy.py" | grep -v '^ok '   # (its FAIL lines, if any, and the count)
echo "--- building the runs (conical rotation and the 'Wu 2005' structure unless stated; mk_run.py has the defaults; wall-bounded and shellular exercise f', f'')"
mk s96      'par.N=96'
mk nodr96   'par.N=96; par.diff_rot=0'
mk s160     'par.N=160'
mk tw0      'par.N=96; par.thermal_wind=0'
mk m0       'par.N=96; par.m=0; par.lmax=21'
mk nodr_m0  'par.N=96; par.m=0; par.lmax=21; par.diff_rot=0'
mk m2b      'par.N=96; par.m=2; par.symm=-1; par.lmax=21'
mk nodr_m2b 'par.N=96; par.m=2; par.symm=-1; par.lmax=21; par.diff_rot=0'
mk y20wb      "par.N=96; par.diff_rot_type='Y20-wall-bounded'"
mk nodr_y20wb "par.N=96; par.diff_rot_type='Y20-wall-bounded'; par.diff_rot=0"
mk shell      "par.N=96; par.diff_rot_type='shellular'; par.diff_rot_ampl=1; par.diff_rot_param=-0.25"
mk nodr_shell "par.N=96; par.diff_rot_type='shellular'; par.diff_rot_ampl=1; par.diff_rot_param=-0.25; par.diff_rot=0"
mk sun      "$SUN; par.diff_rot=1"
mk nodr_sun "$SUN; par.diff_rot=0"
mk sun_a      "$SUN; par.symm=-1; par.diff_rot=1"
mk nodr_sun_a "$SUN; par.symm=-1; par.diff_rot=0"
# exact thermal wind (thermal_wind = 2), Gaspard != 1, ThermaD = 0 (advection of s only), 'solar' in the full sphere
mk s96_tw2   'par.N=96; par.thermal_wind=2'
mk m0_tw2    'par.N=96; par.m=0; par.lmax=21; par.thermal_wind=2'
mk m2b_tw2   'par.N=96; par.m=2; par.symm=-1; par.lmax=21; par.thermal_wind=2'
mk y20wb_tw2 "par.N=96; par.diff_rot_type='Y20-wall-bounded'; par.thermal_wind=2"
mk shell_tw2 "par.N=96; par.diff_rot_type='shellular'; par.diff_rot_ampl=1; par.diff_rot_param=-0.25; par.thermal_wind=2"
mk sun_tw2   "$SUN; par.thermal_wind=2"
mk sun_a_tw2 "$SUN; par.symm=-1; par.thermal_wind=2"
mk sun_m0a_tw2      "$SUN; par.m=0; par.symm=-1; par.lmax=31; par.thermal_wind=2"
mk nodr_sun_m0a_tw2 "$SUN; par.m=0; par.symm=-1; par.lmax=31; par.diff_rot=0"
mk sun_m2_tw2       "$SUN; par.m=2; par.lmax=31; par.thermal_wind=2"
mk nodr_sun_m2_tw2  "$SUN; par.m=2; par.lmax=31; par.diff_rot=0"
mk sol_tw2      "$SOL; par.thermal_wind=2"
mk nodr_sol_tw2 "$SOL; par.diff_rot=0"
mk g3       'par.N=96; par.Gaspard=3; par.Beyonce=90'
mk g3_tw2   'par.N=96; par.Gaspard=3; par.Beyonce=90; par.thermal_wind=2'
mk nodr_g3  'par.N=96; par.Gaspard=3; par.Beyonce=90; par.diff_rot=0'
mk k0       'par.N=96; par.ThermaD=0; par.thermal_wind=0; par.Gaspard=3'
mk nodr_k0  'par.N=96; par.ThermaD=0; par.diff_rot=0; par.Gaspard=3'
# ThermaD = 0 with the thermal wind (thermal_wind 1 and 2): heat rows r² (eq)/p in C^(1) (utils.adiab_tw). diff_rot = 0 or
# thermal_wind = 0 gives the r (eq)/p rows in C^(0), so the reference of asm_check is the same run with diff_rot_ampl = 0 (z_*)
KTW="k1 km0 km2b ky20wb kshell kg3 ksun ksun_a"
for c in "k1:par.N=96" "km0:par.N=96; par.m=0; par.lmax=21" "km2b:par.N=96; par.m=2; par.symm=-1; par.lmax=21" \
         "ky20wb:par.N=96; par.diff_rot_type='Y20-wall-bounded'" "kshell:par.N=96; par.diff_rot_type='shellular'; par.diff_rot_ampl=1; par.diff_rot_param=-0.25" \
         "kg3:par.N=96; par.Gaspard=3; par.Beyonce=90" "ksun:$SUN" "ksun_a:$SUN; par.symm=-1"; do
    d=${c%%:*}; p="${c#*:}; par.ThermaD=0"
    mk $d "$p"; mk ${d}_tw2 "$p; par.thermal_wind=2"; mk z_$d "$p; par.diff_rot_ampl=0"
done
# thermal_wind = 3: prescribed dS_tw/dtheta = dSdtheta sin 2 theta (Bekki et al. 2024), with conical, uniform ('rigid', amplitude 0)
# and full-sphere rotation, and with ThermaD = 0
TW3="par.thermal_wind=3; par.dSdtheta=-0.05"
mk s96_tw3 "par.N=96; $TW3"
mk m0_tw3  "par.N=96; par.m=0; par.lmax=21; $TW3"
mk rig_tw3 "par.N=96; par.diff_rot_type='rigid'; par.diff_rot_ampl=0; $TW3"
mk sun_tw3 "$SUN; $TW3"
mk k1_tw3  "par.N=96; par.ThermaD=0; $TW3"
# the runs of the limit thermal_wind 2 -> 1, of the rigid offset and of the parity warnings (sections below)
for ty in conical Y20-wall-bounded; do for a in 0.1 0.01 0.001; do for w in 1 2; do
    mk lim_${ty}_${a}_$w "par.N=64; par.diff_rot_type='$ty'; par.diff_rot_ampl=$a; par.thermal_wind=$w" quiet
done; done; done
R='par.N=64; par.lmax=25; par.m=2'
V="$R; par.ViscosD=1.; par.ThermaD=1.; par.Beyonce=1e7"   # 'viscous'-like units: G = 1/E = 1e3
K="$R; par.ThermaD=0; par.thermal_wind=0"                  # no thermal diffusion: advection of s only
K1="$R; par.ThermaD=0; par.thermal_wind=1"                 # no thermal diffusion, heat rows r² (eq)/p in C^(1) (S_tw = 0 here)
DR="par.diff_rot=1; par.diff_rot_type='rigid'; par.diff_rot_ampl=0.1"
RIGID=("1:1:$R" "3:3:$R" "V:1e3:$V" "K:3:$K" "K1:1:$K1" "K13:3:$K1")
for c in "${RIGID[@]}"; do
    n=${c%%:*}; r=${c#*:}; G=${r%%:*}; base=${r#*:}
    nodr="par.diff_rot=0"; [ ${n:0:2} = K1 ] && nodr="$DR; par.diff_rot_ampl=0"   # K1: the same heat rows in the baseline
    mk rbase$n "$base; $nodr; par.Gaspard=$G*(1+0.1)" quiet
    mk rfix$n  "$base; $DR; par.Gaspard=$G" quiet
done
mk cyl0 "$SUN; par.diff_rot_type='cylindrical'" quiet
bwait
echo "--- heat_check: operator blocks vs the (r, theta) quadrature of p0 s* [i m G dOmega s + u.grad S_tw]"
for d in s96 s160 tw0 m0 m2b y20wb shell sun sun_a s96_tw2 m0_tw2 m2b_tw2 y20wb_tw2 shell_tw2 sun_tw2 sun_a_tw2 sun_m0a_tw2 sun_m2_tw2 sol_tw2 g3 g3_tw2 k0; do
    (cd $d && python3 "$T/heat_check.py" 3 | tail -1 | sed "s/^/$d: /"); done
for k in $KTW; do for d in $k ${k}_tw2; do (cd $d && python3 "$T/heat_check.py" 3 | tail -1 | sed "s/^/$d: /"); done; done
for d in s96_tw3 m0_tw3 rig_tw3 sun_tw3 k1_tw3; do (cd $d && python3 "$T/heat_check.py" 3 | tail -1 | sed "s/^/$d: /"); done
echo "--- asm_check: heat rows of the assembled A(diff_rot=1) - A(diff_rot=0) vs the same quadrature"
for p in s96:nodr96 s96_tw2:nodr96 m0_tw2:nodr_m0 m2b_tw2:nodr_m2b y20wb_tw2:nodr_y20wb shell_tw2:nodr_shell sun_tw2:nodr_sun sun_a_tw2:nodr_sun_a \
         sun_m0a_tw2:nodr_sun_m0a_tw2 sun_m2_tw2:nodr_sun_m2_tw2 sol_tw2:nodr_sol_tw2 g3:nodr_g3 g3_tw2:nodr_g3 k0:nodr_k0; do
    d=${p%%:*}; b=${p##*:}; (cd $d && python3 "$T/asm_check.py" ../$b/A.npz 3 | tail -1 | sed "s/^/$d: /"); done
for d in m0 m2b y20wb shell sun sun_a; do (cd $d && python3 "$T/asm_check.py" ../nodr_$d/A.npz 3 | tail -1 | sed "s/^/$d: /"); done
for k in $KTW; do for d in $k ${k}_tw2; do (cd $d && python3 "$T/asm_check.py" ../z_$k/A.npz 3 | tail -1 | sed "s/^/$d: /"); done; done
for p in s96_tw3:nodr96 m0_tw3:nodr_m0 rig_tw3:nodr96 sun_tw3:nodr_sun k1_tw3:z_k1; do
    d=${p%%:*}; b=${p##*:}; (cd $d && python3 "$T/asm_check.py" ../$b/A.npz 3 | tail -1 | sed "s/^/$d: /"); done
echo "--- row_check: section-h rows of the ThermaD = 0 runs, unweighted, vs (r, theta) physics (C^(1) basis and r^2 rows with the thermal wind)"
# heat_check/asm_check/ru_insane pair these rows with p0 s* (p0 ~ 1e-6 at the wall) and cannot see their near-surface part,
# where sigma_n (ln rho)' (term (c), no p0 in this mode) converges slowest
for d in k0 k1 k1_tw2 km0_tw2 km2b_tw2 kg3_tw2 ksun ksun_a_tw2 k1_tw3; do (cd $d && python3 "$T/row_check.py" 1e-8 1 | tail -1 | sed "s/^/$d: /"); done
echo "--- thermal_wind = 2 -> 1 as the amplitude -> 0: max|H2 - H1|/max|H1| / amplitude must tend to a constant"
for ty in conical Y20-wall-bounded; do
    args=""
    for a in 0.1 0.01 0.001; do
        for w in 1 2; do (cd lim_${ty}_${a}_$w && { python3 "$T/heat_check.py" 3 > heat_check.out || { tail -1 heat_check.out; exit 1; }; }); done
        args="$args $a lim_${ty}_${a}_1 lim_${ty}_${a}_2"
    done
    echo "$ty:"; python3 "$T/tw_limit.py" $args
done
echo "--- rigid-offset identity, any Gaspard G: lambda(rigid dO, G) = lambda(diff_rot = 0, G (1 + dO)) - i m G dO (ViscosD, ThermaD, Beyonce unchanged)"
for c in "${RIGID[@]}"; do   # (the runs built above)
    n=${c%%:*}; r=${c#*:}; G=${r%%:*}
    s0=$(python3 -c "print(1.1*$G)"); s1=$(python3 -c "print(0.9*$G)")
    (cd rbase$n && python3 "$T/csolve.py" 0 $s0 16 > csolve.out)
    (cd rfix$n  && python3 "$T/csolve.py" 0 $s1 16 > csolve.out)
    python3 "$T/rigid_thermal.py" rbase$n rfix$n 0.1 2 $G | tail -1 | sed "s/^/G = $G ($n): /"
done
echo "--- solar structure ('8-term fit'), ricb = 0: 12 modes near -0.05 + 0.8i, momentum and entropy balances (terms.py)"
(cd sun && python3 "$T/csolve.py" -0.05 0.8 12 > csolve.out && python3 "$T/terms.py" > terms.out && grep -A 13 "^entropy" terms.out)
(cd sun_tw2 && python3 "$T/csolve.py" -0.05 0.8 12 > csolve.out && python3 "$T/terms.py" > terms.out && echo "thermal_wind = 2:" && grep -A 13 "^entropy" terms.out)
# ThermaD = 0 with the exact thermal wind, ricb = 0.35 (shift away from the advection continuum -i m dOmega, omega in [0, 0.25])
(cd k1_tw2 && python3 "$T/csolve.py" -0.05 1.2 8 > csolve.out && python3 "$T/terms.py" > terms.out && echo "ThermaD = 0, thermal_wind = 2 (k1_tw2):" && grep -A 9 "^entropy" terms.out)
echo "--- parity WARNINGs of submatrices.py (ricb = 0): 'solar' and 'cylindrical' (S_tw = 0 exactly) must print none"
for d in sun sun_tw2 sol_tw2 cyl0; do echo "$d: $(grep -c WARNING $d/log_sub || true) warnings"; done
n=$(cat sol_tw2/log_sub cyl0/log_sub | grep -c WARNING || true)   # NEW (2026-11-09): a verdict, as the checks give
echo "'solar' and 'cylindrical': $n WARNINGs, $([ "$n" = 0 ] && echo ok || echo FAIL)"; [ "$n" = 0 ]
echo "--- spin_doctor balances with and without the differential-rotation terms, and Wdr, Wdr_thm vs ebal's R, H (sd_check.py)"
(cd sun && python3 "$T/sd_check.py" | sed "s/^/sun: /")
(cd rfix1 && python3 "$T/sd_check.py" | sed "s/^/rfix1: /")
(cd g3_tw2 && python3 "$T/csolve.py" -0.05 2.0 6 > csolve.out && python3 "$T/sd_check.py" | sed "s/^/g3_tw2: /")   # Gaspard 3, exact thermal wind
for d in nodr96 s96 tw0 y20wb shell; do (cd $d && python3 "$T/csolve.py" -0.05 0.8 8 > csolve.out && python3 "$T/sd_check.py" | sed "s/^/$d: /"); done
(cd k1_tw2 && python3 "$T/sd_check.py" | sed "s/^/k1_tw2: /")   # ThermaD = 0, eig.npz of the terms.py step
(cd s96_tw3 && python3 "$T/csolve.py" -0.05 0.8 8 > csolve.out && python3 "$T/sd_check.py" | sed "s/^/s96_tw3: /")   # prescribed S_tw

#!/bin/bash
# Builds the test runs of the random Reynolds-stress forcing (forcing = 11, bin/reynolds_stress.py) and runs every check.
# usage: tests/rsf/run_all.sh [WORKDIR]   (default: tests/rsf/runs; needs numpy, scipy, mpi4py and, for the solves,
#        petsc4py/slepc4py with MUMPS; MPIEXEC overrides 'mpiexec' and must start ranks of mpi4py's MPI; about half an
#        hour on one core)
# A failed build stops the script; a failed check is reported and the others still run. Exit code 1 if any check failed.
set -e -o pipefail
T=$(cd "$(dirname "$0")" && pwd)
H=$T/../heat_dr   # mk_run.py: run directory from the repo's bin/ with the test defaults (user-def profiles, ricb 0.35, m 1)
MPI=${MPIEXEC:-mpiexec}
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
fail=0
chk() { if ! eval "$1"; then echo "  ^ FAILED: $1"; fail=1; fi; }
# hydro, uniform rotation, constant Theta, nu_t = nu, l_c 0.05; taper 0.1 (its erfc width 0.0125 is resolved from N ~ 96)
RSF="par.forcing=11; par.thermal=0; par.diff_rot=0; par.rsf_theta=1.0; par.rsf_nu_floor=0; par.rsf_lc=0.05; par.rsf_taper=0.1; par.rsf_nsamp=2"
POW="par.rpower_u=6; par.rpower_v=4; par.rhopower_u=2; par.rhopower_v=2"
# a solar-like full sphere: 'smoothed sun' profiles, nu_t from a convection-zone step over a radiative floor (visc0), eddy
# size from H_p (rsf_lc None), finite horizontal size (rsf_angular 1)
SUN="par.ricb=0; par.model_type='smoothed sun'; par.N=192; par.lmax=20; par.m=1; par.symm=1; par.aux0=0.99; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=2.6; par.Beyonce=1e3; par.ViscosD=1e-3; par.ThermaD=1e-3; par.visc0=0.01; par.therm0=1.0; par.visc_pow=0.0; par.therm_pow=0.0; par.bco=1; par.bco_thermal=0; par.thermal=0; par.diff_rot=0; par.forcing=11; par.rsf_nsamp=2; par.rsf_taper=0.1"
echo "--- building the runs"
python3 "$H/mk_run.py" sh48  "par.N=48; $RSF"
python3 "$H/mk_run.py" sh128 "par.N=128; $RSF"
# full sphere m 1 and m 2, symm -1: the two parities of P
python3 "$H/mk_run.py" fs160 "par.N=160; par.ricb=0; par.m=1; par.symm=-1; par.lmax=20; $RSF"
python3 "$H/mk_run.py" fs192 "par.N=192; par.ricb=0; par.m=2; par.symm=-1; par.lmax=21; $RSF"
# other powers of r and rho0 in the rows (the parity rows of the full sphere depend on them); the pairing weight
# rho0^(1-rhopower) r^(3-rpower) of noise_check amplifies the truncation of the rows, hence N 192 and 320 (B1 5e-6 and 4e-5
# at N 128 and 192, converging)
python3 "$H/mk_run.py" shpow "par.N=192; $RSF; $POW"
python3 "$H/mk_run.py" fspow "par.N=320; par.ricb=0; par.m=1; par.symm=1; par.lmax=16; $RSF; $POW"
python3 "$H/mk_run.py" sun     "$SUN; par.rsf_theta='nu'"
python3 "$H/mk_run.py" sun_mlt "$SUN; par.rsf_theta='mlt'; par.rsf_lum=7e-7"
# a smooth inner cut-off (rsf_rcut): no stress below r = 0.45, full above 0.65, across the tail of the viscosity step
python3 "$H/mk_run.py" sun_cut "$SUN; par.rsf_theta='mlt'; par.rsf_lum=7e-7; par.rsf_rcut=0.45; par.rsf_rcut_width=0.2"
# the spectrum run: small enough for a dense eigensolve (sizmat 1152), lmax - m + 1 = 24 for 2 MPI ranks in assemble.py;
# no-slip, so that the energy pairing of spectrum_check.py has no boundary terms
python3 "$H/mk_run.py" sp    "par.N=48; par.lmax=24; par.bci=1; par.bco=1; $RSF; par.rsf_lc=0.1; par.rsf_taper=0.2; par.rsf_nsamp=40"

echo "--- builder_check: f = u against kore's inertia rows (1e-9 at N 48 is profile truncation)"
for d in sh48 sh128 fs160 fs192 shpow fspow sun sun_cut; do
  chk "(cd $d && python3 '$T/builder_check.py' $([ $d = sh48 ] && echo 1e-8) | tail -1)"
done
echo "--- noise_check: the noise against kore's viscous operator (fluctuation-dissipation), exactly"
for d in sh128 fs160 fs192 shpow fspow sun sun_mlt sun_cut; do
  chk "(cd $d && python3 '$T/noise_check.py' | sed 's/^/$d: /')"
done

echo "--- assemble.py with 2 ranks gives the same A, B, B_forced"
if ! $MPI -n 2 python3 -c 'import sys; from mpi4py import MPI; sys.exit(MPI.COMM_WORLD.Get_size() != 2)' > /dev/null 2>&1; then
  echo "  '$MPI -n 2' does not start two ranks of mpi4py's MPI: set MPIEXEC (e.g. the mpiexec of the MPI mpi4py was built with)"
  exit 1
fi
cd sp
for f in A B B_forced; do cp $f.npz ${f}_1.npz; done
chk "$MPI -n 2 python3 bin/assemble.py > log_asm_2 2>&1"
chk "python3 - <<'EOF'
import sys, numpy as np
bad = 0
for X in ('A', 'B', 'B_forced'):
    a, b = np.load(X + '_1.npz'), np.load(X + '.npz')
    same = all(np.array_equal(a[k], b[k]) for k in ('data', 'indices', 'indptr', 'shape'))
    bad += not same
    print('  %s: %s' % (X, 'identical' if same else 'DIFFERENT'), end='')
print('   ', 'FAIL' if bad else 'ok')
sys.exit(1 if bad else 0)
EOF"

echo "--- spectrum_check: forced solves (one LU, 40 realisations) against the exact expected spectrum"
python3 "$T/spectrum_check.py" pick
$MPI -n 1 python3 bin/solve.py > log_solve 2>&1
grep 'right-hand sides' log_solve || true
chk "python3 '$T/spectrum_check.py' fields | tail -1"
echo "par.rsf_frequencies = np.load('rsf_freqs.npy')" >> bin/parameters.py
rm -f rsf_spectrum.dat rsf_ke.dat
$MPI -n 2 python3 bin/solve.py > log_scan_2 2>&1
chk "python3 '$T/spectrum_check.py' check"
mv rsf_spectrum.dat rsf_spectrum_2.dat; mv rsf_ke.dat rsf_ke_2.dat
$MPI -n 1 python3 bin/solve.py > log_scan_1 2>&1
chk "python3 - <<'EOF'
import sys, numpy as np
a, b = np.loadtxt('rsf_ke_2.dat'), np.loadtxt('rsf_ke.dat')
d = np.abs(a - b).max()/np.abs(b[:, 1:]).max()
print('  scan with 1 and 2 ranks, KE of every realisation: max rel. diff %.1e   %s' % (d, 'ok' if d < 1e-10 else 'FAIL'))
sys.exit(0 if d < 1e-10 else 1)
EOF"
echo "--- $([ $fail = 0 ] && echo 'all checks passed' || echo 'SOME CHECKS FAILED')"
exit $fail

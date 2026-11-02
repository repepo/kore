#!/bin/bash
# Parameter ramp: runs the problem in bin/parameters.py once for each value of one parameter. Run from the top folder of
# kore (the one containing bin/):
#
#     tools/ramp.sh name value1 value2 ...     e.g.   tools/ramp.sh Ek 10**-7 10**-7.25 10**-7.5
#     tools/ramp.sh                            runs the case as it is, once
#
# Each value (any python expression) is written into bin/parameters.py as  par.name = value. spin_doctor.py appends
# each step's results to the .dat files (see docs/output.md). With par.track_target = 1 every step aims at the
# eigenvalue that the previous step wrote to the file track_target. The ramp stops when a step finds no eigenvalue (or
# a forced step returns NaN).
# Every step runs submatrices.py (under a second for the S2 torsional case at N = 992), assemble.py, solve_nopp.py and
# spin_doctor.py.
# The environment variables ncpus (processes of submatrices.py and spin_doctor.py, default 4), ranks (MPI ranks of
# assemble.py and solve_nopp.py, default ncpus) and opts (extra solver options, see docs/solver.md) can be set.
# FIX (2026-11-02): rewritten; it called the removed bin/solve.py and its sed patterns no longer matched parameters.py.

set -e
ncpus=${ncpus:-4}
ranks=${ranks:-$ncpus}

setpar() {  # replaces the line par.$1 = ... in bin/parameters.py, or adds one before par.set_scales()
    python3 - "$1" "$2" <<'EOF'
import re, sys
name, value = sys.argv[1:]
s    = open('bin/parameters.py').read()
line = 'par.%s = %s' % (name, value)
pat  = re.compile(r'^par\.%s\s*=.*$' % name, re.M)
s    = pat.sub(lambda m: line, s, count=1) if pat.search(s) else re.sub(r'^par\.set_scales\(\)', lambda m: line + '\n' + m[0], s, count=1, flags=re.M)
open('bin/parameters.py', 'w').write(s)
EOF
}

name=$1
shift || true
values=("$@")
[ -z "$name" ] && values=("")   # no ramp: one run of the case as it is

for value in "${values[@]}"; do
    if [ -n "$name" ]; then
        setpar "$name" "$value"
        echo "---------------- par.$name = $value"
    fi
    python3 ./bin/submatrices.py $ncpus
    mpiexec -n $ranks ./bin/assemble.py
    rm -f no_conv_solution
    mpiexec -n $ranks ./bin/solve_nopp.py $opts
    if [ -f no_conv_solution ]; then
        echo "No solution (no eigenvalue converged, or NaN) at par.$name = $value: ramp stopped"
        exit 1
    fi
    python3 ./bin/spin_doctor.py $ncpus
done

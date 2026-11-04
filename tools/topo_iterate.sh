#!/bin/bash
# NEW (2026-11-04): frequency iteration for the latitude-dependent topographic drag (topo_drag = 'rekier'). The drag of
# eq. (B1) depends on the flow frequency w0, which for an eigenvalue problem is the mode's own |Im lambda|/OmgTau. With
# par.track_target = 1 and par.topo_omega = None, assemble.py takes w0 from the solver target, which set_scales() reads
# from the file track_target, and spin_doctor.py writes the eigenvalue nearest the target back to it: each pass is one
# fixed-point step. Run from the top folder of kore (the one containing bin/), with a file track_target holding the
# starting guess (rtau itau):
#
#     tools/topo_iterate.sh [tol] [maxit]      default tol 1e-10 (relative change of lambda), maxit 20
#
# submatrices.py runs once (the drag enters only the boundary rows of A). spin_doctor.py appends every pass to the .dat
# files, so their last rows are the converged ones. The environment variables ncpus (default 4), ranks (MPI ranks of
# assemble.py and solve_nopp.py, default ncpus) and opts (extra solver options) can be set.

set -e
tol=${1:-1e-10}
maxit=${2:-20}
ncpus=${ncpus:-4}
ranks=${ranks:-$ncpus}

[ -f track_target ] || { echo "topo_iterate.sh: needs a file track_target with the starting rtau itau"; exit 1; }
python3 -c "
import sys; sys.path.insert(0, 'bin')
from parameters import par
sys.exit(0 if (par.track_target == 1 and par.topo_drag is not None and par.topo_omega is None and par.forcing == 0) else 1)" \
    || { echo "topo_iterate.sh: needs par.track_target = 1, par.topo_drag = 'rekier', par.topo_omega = None and forcing = 0"; exit 1; }

./bin/submatrices.py $ncpus > submatrices.log
for k in $(seq 1 $maxit); do
    cp track_target track_target.prev
    mpiexec -n $ranks ./bin/assemble.py > assemble.log
    mpiexec -n $ranks ./bin/solve_nopp.py $opts > solve.log
    ./bin/spin_doctor.py $ncpus > spin_doctor.log
    d=$(python3 -c "
import numpy as np
a, b = np.loadtxt('track_target.prev'), np.loadtxt('track_target')
za, zb = complex(a[0], a[1]), complex(b[0], b[1])
print('%.3e' % (abs(zb - za)/abs(zb)))")
    echo "step $k: lambda = $(tr -s ' ' < track_target | tr '\n' ' ') relative change $d"
    if python3 -c "import sys; sys.exit(0 if $d < $tol else 1)"; then echo "converged"; exit 0; fi
done
echo "not converged after $maxit steps"; exit 1

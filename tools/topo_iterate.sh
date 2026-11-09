#!/bin/bash
# NEW (2026-11-08, from conductive_IC): the frequency of the drag at the outer wall (topo_drag = 'rekier',
# docs/drag.md) made the mode's own. The drag depends on the frequency of the flow, which in an eigenvalue problem is the
# mode's, |Im lambda|/Gaspard. With par.track_target = 1 and par.topo_omega = None, assemble.py takes it from the solver's
# target, which utils.py reads from the file track_target, and spin_doctor.py writes the eigenvalue nearest the target
# back to it: each pass is one step of a fixed-point iteration. Run from the top folder of kore (the one that holds bin/),
# with a file track_target holding the first guess (rtau itau):
#
#     tools/topo_iterate.sh [tol] [maxit]      default tol 1e-10 (the relative change of lambda), maxit 20
#
# submatrices.py runs once (the drag is in the wall rows that assemble.py writes). spin_doctor.py appends every pass to
# the .dat files, so that their last rows are the converged ones. The environment variables ncpus (the processes of
# submatrices.py and spin_doctor.py, default par.ncpus), ranks (the MPI ranks of assemble.py and solve.py, default ncpus),
# opts (more solver options) and MPIEXEC (default mpiexec) can be set.

set -e
tol=${1:-1e-10}
maxit=${2:-20}
[ -f bin/parameters.py ] || { echo "topo_iterate.sh: run it from the top folder of kore (bin/parameters.py)"; exit 1; }
[ -f track_target ] || { echo "topo_iterate.sh: needs a file track_target with the first guess, rtau itau"; exit 1; }
python3 -c "
import sys; sys.path.insert(0, 'bin')
from parameters import par
sys.exit(0 if (par.track_target == 1 and par.topo_drag == 'rekier' and par.topo_omega is None and par.forcing == 0) else 1)" \
    || { echo "topo_iterate.sh: needs par.track_target = 1, par.topo_drag = 'rekier', par.topo_omega = None and forcing = 0"; exit 1; }
ncpus=${ncpus:-$(python3 -c "import sys; sys.path.insert(0, 'bin'); from parameters import par; print(par.ncpus)" | tail -n 1)}
ranks=${ranks:-$ncpus}
mpi=${MPIEXEC:-mpiexec}

python3 ./bin/submatrices.py $ncpus > submatrices.log
for k in $(seq 1 $maxit); do
    cp track_target track_target.prev
    $mpi -n $ranks python3 ./bin/assemble.py > assemble.log
    rm -f no_conv_solution
    $mpi -n $ranks python3 ./bin/solve.py $opts > solve.log
    if [ -f no_conv_solution ]; then echo "step $k: no eigenvalue converged (solve.log)"; exit 1; fi
    python3 ./bin/spin_doctor.py $ncpus > spin_doctor.log
    d=$(python3 -c "
import numpy as np
a, b = np.loadtxt('track_target.prev').ravel(), np.loadtxt('track_target').ravel()
za, zb = complex(a[0], a[1]), complex(b[0], b[1])
print('%.3e' % (abs(zb - za)/abs(zb)))")
    echo "step $k: lambda = $(tr -s ' ' < track_target | tr '\n' ' ') relative change $d"
    if python3 -c "import sys; sys.exit(0 if $d < $tol else 1)"; then echo "converged"; exit 0; fi
done
echo "not converged after $maxit steps"; exit 1

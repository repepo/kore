#!/bin/bash
#PBS -N somename
#PBS -l select=1:ncpus=24:mem=248gb
#PBS -l walltime=03:00:00
#
# PBS job for one run folder made by tools/dodirs.sh, which sets the job name above and dir below; tools/dosubs.sh
# submits it. It runs the case in the folder's bin/parameters.py, or a parameter ramp in it (see tools/ramp.sh).
# FIX (2026-11-02): rewritten; it called the removed bin/solve.py.

source $HOME/venv00/bin/activate   # the python environment with petsc4py and slepc4py

export ncpus=24                    # also the MPI ranks of assemble.py and solve_nopp.py (set ranks to change them)

dir=somedir

cd $HOME/data/$dir

tools/ramp.sh > ramp.log 2>&1                                   # the case as it is
# tools/ramp.sh Lambda 1 2 5 10 > ramp.log 2>&1                 # or a ramp in one parameter

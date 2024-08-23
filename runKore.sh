#!/bin/bash

ncpus=2

opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'
#opts='-ksp_type preonly -pc_type lu'

if [ "$1" == "purge" ]; then
    echo "Purging old matrices..."
    rm *.mtx *.npz
fi

./bin/submatrices.py $ncpus
mpiexec -n $ncpus ./bin/assemble.py
mpiexec -n $ncpus ./bin/solve.py $opts
#./postprocess.py

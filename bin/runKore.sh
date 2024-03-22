#!/bin/bash

ncpus=4
opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'

./submatrices.py $ncpus
mpiexec -n $ncpus ./assemble.py
mpiexec -n $ncpus ./solve_nopp.py $opts
./spin_doctor.py

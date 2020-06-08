#!/bin/bash
#
# Submits a series of jobs to a PBS-managed cluster
#
# use exactly the same arguments as used when
# using the dodirs.sh script


pref=$1
var=$2
exp=$3

for k in $(seq $4 $5 $6)
do

	folder=$pref$k 
	
	cd ~/data/$folder

	sleep 0.2
	qsub subramp.sh
	
done

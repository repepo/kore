#!/bin/bash
#
# Script to collect results from a series of run directories
#
# Use as:
# ./reap.sh somename
#
# where 'somename' is the prefix used when creating the
# directories with dodirs.sh. It will produce the files:
# 
# somename.par, collecting all parameters, 
# somename.flo, with flow data,
# somename.eig, with the eigenvalues, (if forcing = 0)
# somename.mag, with magnetic field data (if magnetic = 1)
#
# Use the script getresults.py to further
# process the results.


#cd ~/data/
p=".par"
f=".flo"
m=".mag"
e=".eig"


for d in $(ls -1d $1*)
do

	if [ -f $d/params.dat ]
	then
		cat $d/params.dat >> $1$p
	fi
	
	if [ -f $d/flow.dat ]
	then
		cat $d/flow.dat >> $1$f
	fi
	
	if [ -f $d/magnetic.dat ]
	then
		cat $d/magnetic.dat >> $1$m
	fi
	
	if [ -f $d/eigenvalues.dat ]
	then
		cat $d/eigenvalues.dat >> $1$e
	fi

done

if [ -f $1$p ]
then
	echo "Parameters written to $1$p"
fi

if [ -f $1$f ]
then
	echo "Flow velocity data written to $1$f"
fi

if [ -f $1$m ]
then
	echo "Magnetic field data written to $1$m"
fi

if [ -f $1$m ]
then
	echo "Eigenvalues written to $1$e"
fi





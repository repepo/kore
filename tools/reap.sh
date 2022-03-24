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
# somename.flo, with flow data,          (if hydro = 1)
# somename.eig, with the eigenvalues,    (if forcing = 0)
# somename.mag, with magnetic field data (if magnetic = 1)
# somename.thm, with thermal data        (if thermal = 1)
#
# Use the script getresults.py to further
# process the results.


#cd ~/data/
p=".par"
f=".flo"
m=".mag"
e=".eig"
s=".sla"
t=".thm"

for d in $(ls -1d $1*)
do

	if [ -f $d/flow.dat ]
	then
        lines=`grep -c ^ $d/flow.dat`
        head -n $lines $d/flow.dat >> $1$f   
		#cat $d/flow.dat >> $1$f
	fi

	if [ -f $d/eigenvalues.dat ]
	then
        head -n $lines $d/eigenvalues.dat >> $1$e
	fi

	if [ -f $d/params.dat ]
	then
		head -n $lines $d/params.dat >> $1$p
	fi
	
	if [ -f $d/magnetic.dat ]
	then
		head -n $lines $d/magnetic.dat >> $1$m
	fi
    
	if [ -f $d/thermal.dat ]
	then
		head -n $lines $d/thermal.dat >> $1$t
	fi    

	if [ -f $d/slayer.dat ]
	then
		head -n $lines $d/slayer.dat >> $1$s
	fi

	if [ -f $d/lions.out ]
	then
		head -n $lines $d/lions.out >> $1$f
	fi

	if [ -f $d/params.out ]
	then
		head -n $lines $d/params.out >> $1$p
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

if [ -f $1$e ]
then
	echo "Eigenvalues written to $1$e"
fi

if [ -f $1$t ]
then
	echo "Thermal data written to $1$t"
fi

if [ -f $1$s ]
then
	echo "Shear layer data written to $1$s"
fi




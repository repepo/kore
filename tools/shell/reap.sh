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


cd ~/data/
p=".par"
e=".eig"
f=".flo"
m=".mag"
t=".tmp"
c=".cmp"

for d in $(ls -1d $1*)
do

	if [ -f $d/params.dat ]
	then
		cat $d/params.dat >> $1$p
	fi
		
	if [ -f $d/eigenvalues.dat ]
	then
		cat $d/eigenvalues.dat >> $1$e
	fi
	
	if [ -f $d/flow.dat ]
	then
		cat $d/flow.dat >> $1$f
	fi
	
	if [ -f $d/magnetic.dat ]
	then
		cat $d/magnetic.dat >> $1$m
	fi
	
	if [ -f $d/thermal.dat ]
	then
		cat $d/thermal.dat >> $1$t
	fi
	
	if [ -f $d/compositional.dat ]
	then
		cat $d/compositional.dat >> $1$c
	fi

done

python3 ~/kore/tools/get_data.py $1

if [ -f $1$p ]
then
	rm $1$p
fi

if [ -f $1$e ]
then
	rm $1$e
fi

if [ -f $1$f ]
then
	rm $1$f
fi

if [ -f $1$m ]
then
	rm $1$m
fi

if [ -f $1$t ]
then
	rm $1$t
fi

if [ -f $1$c ]
then
	rm $1$c
fi

if [ -f $1.csv ]
then
	echo "Data written to $1.csv"
else
	echo "Error! Data not retrieved"
fi




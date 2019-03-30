#!/bin/bash
#
# Script to generate run directories under ~/data/
# (modify as neccessary below)
# useful to submit jobs to a PBS managed cluster
#
# Use as
#
# ./dodirs.sh somename var d startvalue step endvalue
# 
# It will generate a series of directories with names
# beginning with 'somename' and ending with a numerical string
# corresponding to the value assigned to the variable 'var'
# in the parameters.py file.
#
# Example:
#
# ./dodirs.sh run_Ek_ Ek e -5 -0.1 -6
#
# will generate directories named run_Ek_-5.0, run_Ek_-5.1, ...
# and so on up to run_Ek_-6.0. In each directory the file parameters.py
# will have the appropriate value assigned to the parameter Ek, the Ekman number,
# ranging from Ek =10**-5.0 to Ek =10**-6.0 across all the diectories.
# 
# If the argument 'e' is changed to 'd' then 'var' will have values
# that change linearly instead of as powers of ten. For example:
#
# ./dodirs.sh run_ricb_ ricb d 0.35 0.01 0.5
#
# will generate directories named run_ricb_0.35, run_ricb_0.36, ...
# up to run_ricb_0.50. The variable 'ricb', the inner core radius,
# will have values ranging from 0.35 to 0.50 in each parameters.py
# file across all the directories generated.


pref=$1
var=$2
exp=$3

for k in $(seq $4 $5 $6)
do

	folder=$pref$k 
	if [ $exp = 'e' ]; then
		value='10**'$k # powers of ten
	else
		value=$k # linear
	fi
	echo $folder $var=$value
	
	mkdir ~/data/$folder

	cd ~/data/$folder
	
	cp ~/tintin-0.1/* . # copies the source files

	# modify variables
	sed -i 's,^\('$var'[ ]*=\).*,\1'$value',' parameters.py	

	sed -i 's,^\(#PBS -N \).*,\1'$folder',' subramp.sh
	
	sed -i 's,^\(dir=\).*,\1'$folder',' subramp.sh
	
done

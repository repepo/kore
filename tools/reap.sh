#!/bin/bash
#
# Collects the results of a series of run folders into one file per table. Use as
#
#     tools/reap.sh somename
#
# where somename is the prefix of the folders (the one given to tools/dodirs.sh). It writes, with the rows of every
# folder in the order ls lists them (one row per solution, as spin_doctor.py writes them):
#
# somename.par, the parameters                (params.dat)
# somename.flo, the flow                      (flow.dat)
# somename.thm, the entropy                   (thermal.dat, with thermal = 1)
# somename.cmp, the composition               (composition.dat, with compositional = 1)
# somename.mag, the magnetic field            (magnetic.dat, with magnetic = 1)
# somename.rot, the rotating solids           (rotdyn.dat, with rotdyn = 1)
# somename.eig, the eigenvalues               (eigenvalues.dat, eigenvalue problems)
#
# tools/get_data.py somename reads them with named columns, tools/getresults.py into IPython.
# NEW (2026-11-08, from conductive_IC): the number of rows taken from each folder is that of its params.dat, so that the
# rows of the files stay aligned; a folder without params.dat (no finished run) is left out, and the files are written
# anew on each call. composition.dat and rotdyn.dat are collected too.

[ -n "$1" ] || { echo "usage: tools/reap.sh somename"; exit 1; }
tables=("params.dat .par" "flow.dat .flo" "thermal.dat .thm" "composition.dat .cmp" "magnetic.dat .mag"
        "rotdyn.dat .rot" "eigenvalues.dat .eig")
for t in "${tables[@]}"; do rm -f "$1${t#* }"; done

for d in $(ls -1d "$1"*)
do
	[ -f "$d/params.dat" ] || continue
	lines=$(grep -c ^ "$d/params.dat")
	for t in "${tables[@]}"; do
		dat=${t% *}; suffix=${t#* }
		[ -f "$d/$dat" ] && head -n $lines "$d/$dat" >> "$1$suffix"
	done
done

for t in "${tables[@]}"; do
	dat=${t% *}; suffix=${t#* }
	[ -f "$1$suffix" ] && echo "$dat collected in $1$suffix ($(grep -c ^ "$1$suffix") rows)"
done
exit 0

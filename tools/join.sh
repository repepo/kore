#!/bin/bash
#
# Script to concatenate two datasets
# Use as ./join.sh label1 label2 out
# it will concatenate label1.par and label2.par and write it to out.par
# same for .flo, .mag, .eig.
# If one of the labels is "nolabel" then
# it will read from params.dat, eigenvalues.dat, flow.dat and magnetic.dat.

p=".par"
f=".flo"
m=".mag"
e=".eig"

if [ $1 = "nolabel" ]
then
	p1="params.dat"
	e1="eigenvalues.dat"
	f1="flow.dat"
	m1="magnetic.dat"
else
	p1=$1$p
	e1=$1$e
	f1=$1$f
	m1=$1$m	
fi

if [ $2 = "nolabel" ]
then
	p2="params.dat"
	e2="eigenvalues.dat"
	f2="flow.dat"
	m2="magnetic.dat"
else
	p2=$2$p
	e2=$2$e
	f2=$2$f
	m2=$2$m	
fi

if [ -f $p1 ] && [ -f $p2 ]
then
	cat $p1 $p2 >> $3$p
fi

if [ -f $e1 ] && [ -f $e2 ]
then
	cat $e1 $e2 >> $3$e
fi

if [ -f $f1 ] && [ -f $f2 ]
then
	cat $f1 $f2 >> $3$f
fi

if [ -f $m1 ] && [ -f $m2 ]
then
	cat $m1 $m2 >> $3$m
fi

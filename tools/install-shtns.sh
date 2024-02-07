#!/bin/bash

ver="3.5.2"

wget https://bitbucket.org/nschaeff/shtns/downloads/shtns-$ver.tar.gz
tar -xvf shtns-$ver.tar.gz
rm shtns-$ver.tar.gz

if [ -d "shtns-$ver" ]
then
    mv shtns-$ver shtns
fi

cd shtns

opts="--prefix=$HOME/.local --enable-ishioka --enable-openmp --enable-python"

if [[ -n $MKLROOT ]]
then
   echo "MKL found, installing with MKL"
   opts="$opts --enable-mkl"
else
   echo "MKL not found, will try to install with FFTW"
fi

./configure $opts

if [ `echo $CC` ]
then
    sed -i "s/shtcc=gcc/shtcc=${CC}/" Makefile
fi

make -j
make install -j

# Python installation will fail because of sudo, install as --user

python3 setup.py install --user

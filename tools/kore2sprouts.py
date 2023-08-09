#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import scipy.sparse as ss
import scipy.io as sio

sys.path.insert(1,'bin/')
import utils as ut
import parameters as par




def main(Anpz):
    '''
    reshuffle kore matrices to match sprouts ordering
    '''
    
    A = ut.load_csr(str(Anpz))
    
    N = par.N

    ll = int(A.shape[0]/(4*N))

    gb = np.array([4,2,2,2])  # the basis order for each section
    
    N1 = N-gb  # number of rows without bc's for each section
    
    nbc = 2*np.sum(gb)  # total number of bc rows

    Aout = ss.dok_matrix(A.shape, dtype=complex)
        
        
    for j in range(4):  #loop over the four sections
    
        for k in range(ll):
            
            row0 = gb[j] + k*N + j*ll*N  #initial row for kore
            
            row1 = nbc + k*N1[j] + sum(ll*N1[:j])  #initial row for sprouts
            
            Aout[ row1 : row1 + N1[j], : ] = A[ row0 : row0 + N1[j], : ]        
    
    sio.mmwrite('out.mtx', Aout)


    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1]))

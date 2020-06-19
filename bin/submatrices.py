#!/usr/bin/env python3
'''
tintin generates submatrices

To use:
> ./submatrices.py ncpus
where ncpus is the number of cores

This program generates the operators involved in the differential
equations as submatrices. These submatrices are required for the assembly
of the main matrices A and B.
'''

from timeit import default_timer as timer
import multiprocessing as mp
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import warnings
import sys

import parameters as par
import utils as ut


def main(ncpus):
	
	warnings.simplefilter('ignore', ss.SparseEfficiencyWarning)

	tic = timer()
	print('N =', par.N,', lmax =', par.lmax)
	tol = 1e-9
	
	# Chebyshev coefficients of powers of r
	a1 = ut.chebco(1, par.N, tol, par.ricb, ut.rcmb)
	a2 = ut.chebco(2, par.N, tol, par.ricb, ut.rcmb)
	a3 = ut.chebco(3, par.N, tol, par.ricb, ut.rcmb)
	a4 = ut.chebco(4, par.N, tol, par.ricb, ut.rcmb)
	
	# Basis transformation 
	S0 = ut.Slam(0, par.N)
	S1 = ut.Slam(1, par.N)
	S2 = ut.Slam(2, par.N)
	S3 = ut.Slam(3, par.N)
	
	# Derivatives
	D1 = ut.Dlam(1, par.N)
	D2 = ut.Dlam(2, par.N)
	D3 = ut.Dlam(3, par.N)
	D4 = ut.Dlam(4, par.N)
	
	# auxiliary
	S10 = S1*S0
	S32 = S3*S2
	S321 = S32*S1
	
	# Product matrices (powers of r times something)
	# these are the most time consuming to compute
	'''
	M10 = Mlam(a1,0)
	M11 = Mlam(S0*a1,1)
	M20 = Mlam(a2,0)
	M21 = Mlam(S0*a2,1)
	M22 = Mlam(S10*a2,2)
	M30 = Mlam(a3,0)
	M31 = Mlam(S0*a3,1)
	M32 = Mlam(S10*a3 ,2)
	M33 = Mlam(S2*S10*a3,3)
	M40 = Mlam(a4,0)
	M41 = Mlam(S0*a4,1)
	M42 = Mlam(S10*a4,2)
	M43 = Mlam(S2*S10*a4,3)
	M44 = Mlam(S321*S0*a4,4)
	'''
	
	# let's generate them in parallel using multiprocessing:
	# the arguments are
	arg0 = [a1, S0*a1, a2, S0*a2, S10*a2, a3, S0*a3, S10*a3, S2*S10*a3, a4, S0*a4, S10*a4, S2*S10*a4, S321*S0*a4]
	arg1 = [0 , 1    , 0 , 1    , 2     , 0 , 1    , 2     , 3        , 0 , 1    , 2     , 3        , 4         ]
	
	# execute
	pool = mp.Pool(processes=int(ncpus))
	p = [ pool.apply_async(ut.Mlam,args=(arg0[k],arg1[k])) for k in range(14) ]
	
	# recover results
	matlist = [p1.get() for p1 in p]
		
	pool.close()
	pool.join()
	
	M10 = matlist[0]
	M11 = matlist[1]
	
	M20 = matlist[2]
	M21 = matlist[3]
	M22 = matlist[4]
	
	M30 = matlist[5]
	M31 = matlist[6]
	M32 = matlist[7]
	M33 = matlist[8]
	
	M40 = matlist[9]
	M41 = matlist[10]
	M42 = matlist[11]
	M43 = matlist[12]
	M44 = matlist[13]
	
	# Matrices needed for the bottom half (b) for the single curl equations
	r2D2b = M22*D2
	r2D1b = S1*M21*D1
	r2Ib  = S10*M20
	r1D1b = S1*M11*D1
	r1Ib  = S10*M10
	Ib    = S10
	
	# Matrices needed for the thermal equation
	r4D2b = M42*D2
	r3D2b = M32*D2
	r3D1b = S1*M31*D1
	r4Ib  = S10*M40
	r3Ib  = S10*M30
	
	bot    = [ r2D2b,  r2D1b,  r2Ib,  r1D1b,  r1Ib,  Ib,  r4D2b,  r3D1b,  r4Ib , r3Ib , r3D2b ]
	blabel = ['r2D2b','r2D1b','r2Ib','r1D1b','r1Ib','Ib','r4D2b','r3D1b','r4Ib','r3Ib','r3D2b']

		
	# Matrices needed for the top half (t) for the double curl equations
	r4D4t = M44*D4
	r4D3t = S3*M43*D3
	r4D2t = S32*M42*D2
	r4D1t = S321*M41*D1
	r4It  = S321*S0*M40
	
	r3D3t = S3*M33*D3
	r3D2t = S32*M32*D2
	r3D1t = S321*M31*D1
	r3It  = S321*S0*M30
	
	r2D2t = S32*r2D2b
	r2D1t = S32*r2D1b
	r2It  = S32*r2Ib
	
	r1It  = S32*r1Ib
	It    = S32*Ib
	
	top    = [ r4D4t , r4D3t , r4D2t , r4D1t , r4It , r3D3t,  r3D2t , r3D1t,  r3It,  r2D2t,  r2D1t,  r2It,  r1It,  It ]
	tlabel = ['r4D4t','r4D3t','r4D2t','r4D1t','r4It','r3D3t','r3D2t','r3D1t','r3It','r2D2t','r2D1t','r2It','r1It','It']
		
	# Now chop the last 4(2) rows of top(bottom) matrices and
	# shift rows down to make room for boundary conditions
	
	z4 = ss.csr_matrix((4,par.N))
	for i in range( 0, np.size(top) ):
		tmp = ss.vstack( [ z4 , top[i][:-4,:] ], format='csr' )
		sio.mmwrite( tlabel[i], tmp )
	
	z2 = ss.csr_matrix((2,par.N))
	for i in range( 0, np.size(bot) ):
		tmp = ss.vstack( [ z2 , bot[i][:-2,:] ], format='csr' )	
		sio.mmwrite( blabel[i], tmp )
		
	toc = timer()
	
	print('Blocks generated and written to disk in', toc-tic, 'seconds')

	return 0
	

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))

#!/usr/bin/env python3
'''
tintin solves

To use, first export desired solver options:
> export opts='...'

and then execute:
> mpiexec -n ncpus ./solve.py $opts
'''

import sys
import petsc4py
petsc4py.init(sys.argv)
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import scipy.io as sio
import scipy.sparse as ss
from timeit import default_timer as timer
import numpy as np

import parameters as par
import utils as ut
import utils_pp as upp



def main():
	
	Print = PETSc.Sys.Print
	rank = PETSc.COMM_WORLD.getRank()
	size = PETSc.COMM_WORLD.getSize()
	opts = PETSc.Options()
	
	if rank == 0:
		tic = timer()
		
	# ------------------------------------------------------------------ reads matrix A
	A = ut.load_csr('A.npz')
	nb_l,nb_c = A.shape
	
	MA = PETSc.Mat()
	MA.create(PETSc.COMM_WORLD)
	MA.setSizes([nb_l,nb_l])
	MA.setType('mpiaij')
	MA.setFromOptions()
	MA.setUp()
	
	Istart,Iend = MA.getOwnershipRange()
	indptrA = A[Istart:Iend,:].indptr
	indicesA = A[Istart:Iend,:].indices
	dataA = A[Istart:Iend,:].data
	del A
	
	MA.setPreallocationCSR(csr=(indptrA,indicesA))
	MA.setValuesCSR(indptrA,indicesA,dataA)
	MA.assemblyBegin()
	MA.assemblyEnd()
	del indptrA,indicesA,dataA
	# done reading and preparing A
	
	if par.forcing == 0: # --------------------------------------------- if eigenvalue problem, reads matrix B
		
		B = ut.load_csr('B.npz')
		nb_l,nb_c = B.shape
		nbl = opts.getInt('nbl',nb_l)
		
		MB = PETSc.Mat()
		MB.create(PETSc.COMM_WORLD)
		MB.setSizes([nbl,nbl])
		MB.setType('mpiaij')
		MB.setFromOptions()
		MB.setUp()
		
		Istart,Iend = MB.getOwnershipRange()
		indptrB = B[Istart:Iend,:].indptr
		indicesB = B[Istart:Iend,:].indices
		dataB = B[Istart:Iend,:].data
		del B
		
		MB.setPreallocationCSR(csr=(indptrB,indicesB))
		MB.setValuesCSR(indptrB,indicesB,dataB)
		MB.assemblyBegin()
		MB.assemblyEnd()
		del indptrB,indicesB,dataB
		# done reading and preparing B
		
		
		# -------------------------------------------------------------- setup eigenvalue solver
		E = SLEPc.EPS()
		E.create(SLEPc.COMM_WORLD)
		E.setOperators(MA,MB)
		E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
		#E.setDimensions(nev,ncv)
		E.setDimensions(par.nev)
		E.setTolerances(par.tol,par.maxit)
		
		wep = par.which_eigenpairs
		if wep == 'LM':
			E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
		elif wep == 'SM':
			E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
		elif wep == 'LR':
			E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
		elif wep == 'SR':
			E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
		elif wep == 'LI':
			E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
		elif wep == 'SI':
			E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_IMAGINARY)
		elif wep == 'TM':
			E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
		elif wep == 'TR':
			E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
		elif wep == 'TI':
			E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_IMAGINARY)
		
		E.setTarget(par.tau)
		E.setFromOptions()
		# done setting up solver
	
		E.solve() # ---------------------------------------------------- solve and collect solution
		
		class solution:
			pass  # class intentionally empty, this is just to have sol.*stuff*
			
		sol = solution()
		
		# recover results
		sol.its                   = E.getIterationNumber()
		sol.neps_type             = E.getType()
		sol.tol, sol.maxit        = E.getTolerances()
		sol.nev, sol.ncv, sol.mpd = E.getDimensions()
		sol.nconv                 = E.getConverged()
		sol.k                     = np.zeros((1,sol.nconv),dtype=complex)
		sol.vec                   = np.zeros((nb_l,sol.nconv),dtype=complex)
		sol.tau                   = E.getTarget()
		
		if sol.nconv > 0:
			
			# initialize eigenvector placeholder
			v = MA.createVecLeft()
			
			for i in range(0,sol.nconv):
				# gets eigenvalue and eigenvector for each solution found
				# note that petsc uses complex scalars, so k and v are generally complex
				# no need for separate real and imag (hence the None below)
				k = E.getEigenpair(i, v, None)
				# collects and assembles the vector in the zeroth processor 
				tozero,V = PETSc.Scatter.toZero(v)
				tozero.begin(v,V)
				tozero.end(v,V)
				tozero.destroy()
		
				sol.k[0,i] = k
				if rank == 0:
					sol.vec[0:,i] = V[0:]
		
			if rank == 0:
			
				# Eigenvalues
				eigval = np.hstack([np.real(sol.k).transpose(),np.imag(sol.k).transpose()])
				
				# Eigenvectors
				rEigv = np.copy(np.real(sol.vec))
				iEigv = np.copy(np.imag(sol.vec))
				
				ru = rEigv[:2*ut.n,:]
				iu = iEigv[:2*ut.n,:]
				
				if par.magnetic == 1:
					rb = rEigv[2*ut.n:,:]
					ib = iEigv[2*ut.n:,:]
				
				success = sol.nconv
				
		else:
			
			if rank == 0:
				
				success = 0
				print('No converged solution found')
				
		MA.destroy()
		MB.destroy()
				
				
				
				
	else: # ------------------------------------------------------------ if forced problem, reads forcing vector
		
		b0 = ut.load_csr('B_forced.npz')
		x, bvec = MA.createVecs()
	
		#b.set(0)
		Istart,Iend = bvec.getOwnershipRange()
		bvec.setValues(range(Istart,Iend),b0[Istart:Iend,0].toarray())
		bvec.assemblyBegin()
		bvec.assemblyEnd()
		del b0
		
		# -------------------------------------------------------------- setup, solve & collect
		K = PETSc.KSP()
		K.create(PETSc.COMM_WORLD)
		K.setOperators(MA)
		K.setFromOptions()
		
		# solve
		K.solve(bvec, x)
		
		# collect result
		tozero,VR = PETSc.Scatter.toZero(x)
		tozero.begin(x,VR)
		tozero.end(x,VR)
		tozero.destroy()
	
		# cleanup
		K.destroy()
		MA.destroy()
		bvec.destroy()
		x.destroy()
		
		if rank == 0:
			
			ru = np.reshape(np.real(VR[:2*ut.n]),(-1,1))
			iu = np.reshape(np.imag(VR[:2*ut.n]),(-1,1))
			
			if par.magnetic == 1:
				rb = np.reshape(np.real(VR[2*ut.n:]),(-1,1))
				ib = np.reshape(np.imag(VR[2*ut.n:]),(-1,1))
					
			if np.sum([np.isnan(ru), np.isnan(iu), np.isinf(ru), np.isinf(iu)]) > 0:
				success = 0
				print('Solver crashed, got nan\'s!')
			else:
				success = 1 # got actual numbers (... but it could still be a bad solution)
				print('Solution(s) computed')
		
	PETSc.COMM_WORLD.Barrier()
	
	
	# ------------------------------------------------------------------ Postprocessing: compute energy, dissipation, etc.
	
	if rank == 0:
		
		if success > 0:
			
			toc1 = timer()
			
			kid = np.zeros((success,7))
			if par.magnetic == 1:
				ohm = np.zeros((success,4))
			params = np.zeros((success,17))
			
			for i in range(success):
				
				#print('Solution',i)
				
				a = np.copy(ru[:,i])
				b = np.copy(iu[:,i])
				
				if par.forcing == 0:
					w = eigval[i,1]
					sigma = eigval[i,0]
				else:
					w = ut.wf
					sigma = 0
				
				kid[i,:] = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
				par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing)
		
				KE = kid[i,0]+kid[i,1]
				
				#print('--')
				#print('omega*Energy =', w*KE )
				#print('Imag power   =', kid[i,6] )
				#print('Imag dissip  =', Ek*kid[i,4] )
				print('--')
				print('sigma*Energy   =', sigma*KE    )
				print('kinetic Dissp  =', kid[i,3]*par.Ek )
				print('real Power     =', kid[i,5]    )
				print('internal Dissp =', kid[i,2]*par.Ek )
				print('--')
				
				if par.magnetic == 1:
					
					a = np.copy(rb[:,i])
					b = np.copy(ib[:,i])
					
					ohm[i,:] = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus)
					# use -symm above because magnetic field has the opposite
					# symmetry as the flow field --if applied field is antisymm (vertical uniform).
					
				params[i,:] = np.array([par.Ek, par.m, par.symm, par.ricb, par.bci, par.bco, par.projection, par.forcing, \
				par.forcing_amplitude, par.forcing_frequency, par.magnetic, par.Em, par.Le2, par.N, par.lmax, toc1-tic, par.ncpus])
				
				
			
			# ---------------------------------------------------------- write post-processed data and parameters to disk
			
			#print('Writing results')
			
			with open('params.dat','ab') as dpar:
				np.savetxt(dpar, params, \
				fmt=['%.6e','%d','%d','%.8e','%d','%d','%d','%d','%.6e','%.6e','%d','%.6e','%.6e','%d','%d','%.2f', '%d'])  
			
			with open('flow.dat','ab') as dflo:
				np.savetxt(dflo, kid)
		    
			if par.magnetic == 1:
				with open('magnetic.dat','ab') as dmag:
					np.savetxt(dmag, ohm)
	
			if par.forcing == 0:
				with open('eigenvalues.dat','ab') as deig:
					np.savetxt(deig,eigval)
			
			
			# ---------------------------------------------------------- write solution vector to disk
			
			if par.write_eig == 1:
				
				np.savetxt('real_flow.field',ru)
				np.savetxt('imag_flow.field',iu)
			
				if par.magnetic == 1:
					
					np.savetxt('real_magnetic.field',rb)
					np.savetxt('imag_magnetic.field',ib)
		
		toc2 = timer()
		print('Solve done in',toc2-tic,'seconds')
	
	# ------------------------------------------------------------------ done
	return 0
	
	
	
if __name__ == "__main__": 
	sys.exit(main())

	

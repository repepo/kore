#!/usr/bin/env python3
'''
tintin solves

To use, first export desired solver options:
> export opts='...'

and then execute:
> mpiexec -n ncpus ./solve.py $opts
'''

import sys
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
		K.setTolerances(rtol=par.tol,max_it=par.maxit)
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
				success = 1 # got actual numbers ... but it could still be a bad solution ;)
				print('Solution(s) computed')
		
	PETSc.COMM_WORLD.Barrier()
	
	
	# ------------------------------------------------------------------ Postprocessing: compute energy, dissipation, etc.
	
	if rank == 0:
		
		if success > 0:
			
			toc1 = timer()
			
			kid = np.zeros((success,7))
			Dint_partial = np.zeros((success,3))
			p2t = np.zeros(success)
			y = np.zeros(success)
			
			
			if par.magnetic == 1:
				ohm = np.zeros((success,4))			
				Dohm_partial = np.zeros((success,3))
				o2v = np.zeros(success)
				
				
			params = np.zeros((success,18))
			
			thk = np.sqrt(par.Ek)
			R1 = par.ricb + 15*thk
			R2 = ut.rcmb - 15*thk
			R3 = ut.rcmb - 30*thk
		
			print('Post-processing:')	 
			print('--- -------------- -------------- ---------- ----------')
			print('Sol    Damping        Frequency     Error1     Error2  ')
			print('--- -------------- -------------- ---------- ----------')
			#print('{2d}   {:11.9f}   {:11.9f}   {:10.2g}   {:10.2g}'.format(i, sigma, w, err1, err2))
			
			if par.track_target == 1:  # eigenvalue tracking enabled
				#read target data
				x = np.loadtxt('track_target')
			
			
			
			
			for i in range(success):
				
				#print('Processing solution',i)
				
				a = np.copy(ru[:,i])
				b = np.copy(iu[:,i])
				
				if par.forcing == 0:
					w = eigval[i,1]
					sigma = eigval[i,0]
				else:
					w = ut.wf
					sigma = 0
				
				kid[i,:] = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
				par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, par.ricb, ut.rcmb)
		
				k1 = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
				par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, par.ricb, R1)
				
				k2 = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
				par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, R2, ut.rcmb)
				
				k3 = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
				par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, R3, ut.rcmb)
				
				Dint_partial[i,0] = k1[2]*par.Ek
				Dint_partial[i,1] = k2[2]*par.Ek
				Dint_partial[i,2] = k3[2]*par.Ek
				
				KP = kid[i,0]
				KT = kid[i,1]
				p2t[i] = KP/KT
				KE = KP + KT
				Dint = kid[i,2]*par.Ek
				Dkin = kid[i,3]*par.Ek
				
				repow = kid[i,5]
				
				err1 = abs(-Dint/Dkin -1)
				
				if par.track_target == 1:	
					# compute distance (mismatch) to tracking target
					y0 = abs( (x[0]-sigma)/sigma )		# damping mismatch
					y1 = abs( (x[1]-w)/w )				# frequency mismatch
					y2 = abs( (x[2]-p2t[i])/p2t[i] )	# poloidal to toroidal energy ratio mismatch
					y[i] = y0 + y1 + y2					# overall mismatch
					   
				#print('omega*Energy =', w*KE )
				#print('Imag power   =', kid[i,6] )
				#print('Imag dissip  =', Ek*kid[i,4] )
				#print('--')
				#print('sigma*Energy   =', sigma*KE    )
				#print('kinetic Dissp  =', kid[i,3]*par.Ek )
				#print('real Power     =', kid[i,5]    )
				#print('internal Dissp =', kid[i,2]*par.Ek )

				
				if par.magnetic == 1:
					
					a = np.copy(rb[:,i])
					b = np.copy(ib[:,i])
					
					ohm[i,:] = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb, ut.rcmb)
					# use -symm above because magnetic field has the opposite
					# symmetry as the flow field --if applied field is antisymm (vertical uniform).
					
					o1 = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb, R1)
					o2 = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, R2, ut.rcmb)
					o3 = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, R3, ut.rcmb)
					
					Dohm_partial[i,0] = (o1[2] + o1[3])*par.Le2*par.Em
					Dohm_partial[i,1] = (o2[2] + o2[3])*par.Le2*par.Em
					Dohm_partial[i,2] = (o3[2] + o3[3])*par.Le2*par.Em
					
					Dohm = (ohm[i,2]+ohm[i,3])*par.Le2*par.Em	
					#err2 = abs( 1+(Dohm-Dkin)/(sigma*KE) )
					
					o2v[i] = Dohm/Dint
					
					if par.track_target == 1:
						y3 = abs( (x[3]-o2v[i])/o2v[i] )
						y[i] += y3 	#include ohmic to viscous dissipation ratio in the tracking
				
				else:
					
					Dohm = 0
				
				
				if par.forcing != 0:
					if repow != 0:							# body forcing (input power should match total dissipation)
						err2 = abs( (repow-(Dohm-Dkin))/repow )
					else:									# boundary flow forcing (input power to be implemented)
						err2 = -1.0	
				elif par.forcing == 0:						# eigenvalue problem (damping should match total dissipation)
					err2 = abs( 1+(Dohm-Dkin)/(sigma*KE) )
				
				
				
				print('{:2d}   {: 12.9f}   {: 12.9f}   {:8.2e}   {:8.2e}'.format(i, sigma, w, err1, err2))
				
					
					
				params[i,:] = np.array([par.Ek, par.m, par.symm, par.ricb, par.bci, par.bco, par.projection, par.forcing, \
				par.forcing_amplitude, par.forcing_frequency, par.magnetic, par.Em, par.Le2, par.N, par.lmax, toc1-tic, par.ncpus, par.tol])
				
			print('--- -------------- -------------- ---------- ----------')
			
			#find closest eigenvalue to tracking target and write to target file
			if (par.track_target == 1)&(par.forcing == 0):
				j = y==min(y)
				if par.magnetic == 1:
					with open('track_target','wb') as tg:
						np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j], o2v[j] ] )
				else:
					with open('track_target','wb') as tg:
						np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j] ] )
				print('Closest to target is solution', np.where(j==1)[0][0])
				
				
				
			# ---------------------------------------------------------- write post-processed data and parameters to disk
			
			#print('Writing results')
			
			with open('params.dat','ab') as dpar:
				np.savetxt(dpar, params, \
				fmt=['%.9e','%d','%d','%.9e','%d','%d','%d','%d','%.9e','%.9e','%d','%.9e','%.9e','%d','%d','%.2f', '%d', '%.2e'])  
			
			with open('flow.dat','ab') as dflo:
				np.savetxt(dflo, np.c_[kid, Dint_partial])
		    
			if par.magnetic == 1:
				with open('magnetic.dat','ab') as dmag:
					np.savetxt(dmag, np.c_[ohm, Dohm_partial])
	
			if par.forcing == 0:
				with open('eigenvalues.dat','ab') as deig:
					np.savetxt(deig,eigval)
			
			
			# ---------------------------------------------------------- write solution vector('s) to disk
			
			if par.write_eig == 1: # one solution per columns
				
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

	

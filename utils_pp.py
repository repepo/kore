# utility for postprocessing solutions

import multiprocessing as mp
import scipy.fftpack as sft
import numpy.polynomial.chebyshev as ch
import numpy as np

import parameters as par
import utils as ut


# ----------------------------- global variables, read only for the pool workers

# xk are the colocation points, from -1 to 1
i = np.arange(0,par.N)
xk = np.cos( (i+0.5)*np.pi/par.N )

# Chebyshev polynomials evaluated at xk
chx = ch.chebvander(xk,par.N-1)

# rk are the radial colocation points, from ricb to rcmb
rk = 0.5*(ut.rcmb-par.ricb)*( xk + 1 ) + par.ricb

# the following are needed to compute integrals
sqx = np.sqrt(1-xk**2)
r2 = rk**2
r3 = rk**3
r4 = rk**4

# ------------------------------------------------------------------------------



def pol_worker( l, Pk, N, m, ricb, rcmb, w, projection, forcing): # ------------ 
	'''
	Here we compute various integrals that involve poloidal components, degree l
	We use Chebyshev-Gauss quadratures
	'''
	
	L  = l*(l+1)

	# Pk is the full set of cheb coefficients for a given l
	# we use the full domain [-1,1] if an inner core is present
	
	dPk  = ut.Dcheb(Pk,ricb,rcmb)   # cheb coeffs of the first derivative 
	d2Pk = ut.Dcheb(dPk,ricb,rcmb)  # 2nd derivative
	d3Pk = ut.Dcheb(d2Pk,ricb,rcmb) # 3rd derivative

	# plm's are the poloidal scalars evaluated at the colocation points
	
	plm0 = np.zeros(np.shape(xk),dtype=complex)
	plm1 = np.zeros(np.shape(xk),dtype=complex)
	plm2 = np.zeros(np.shape(xk),dtype=complex)
	plm3 = np.zeros(np.shape(xk),dtype=complex)	
	
	plm0 = np.dot(chx,Pk,plm0)
	plm1 = np.dot(chx,dPk,plm1)
	plm2 = np.dot(chx,d2Pk,plm2)
	plm3 = np.dot(chx,d3Pk,plm3)
	
	# the radial scalars
	qlm0 = L*plm0/rk
	qlm1 = (L*plm1 - qlm0)/rk
	qlm2 = (L*plm2-2*qlm1)/rk
	
	# the consoidal scalars
	slm0 = plm1 + (plm0/rk)
	slm1 = plm2 + (qlm1/L)
	slm2 = plm3 + (qlm2/L)
	
	
	
	# -------------------------------------------------------------------------- kinetic energy, poloidal
	
	f0 = 4*np.pi/(2*l+1)
	f1 = r2*np.absolute( qlm0 )**2
	f2 = r2*L*np.absolute( slm0 )**2
	
	# for the integrals we use Chebyshev-Gauss quadratures
	Ken_pol_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1+f2) )
	
	
	# -------------------------------------------------------------------------- internal energy dissipation, poloidal
	# (symm\nabla u):(symm\nabla u)
	
	f0 = 4*np.pi/(2*l+1)
	f1 = L*np.absolute(qlm0 + rk*slm1 - slm0)**2
	f2 = 3*np.absolute(rk*qlm1)**2
	f3 = L*(l-1)*(l+2)*np.absolute(slm0)**2
	# integral is
	Dint_pol_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1+f2+f3 ) )

	
	# -------------------------------------------------------------------------- kinetic energy dissipation rate, poloidal
	# (1/2)* u.nabla^2 u
	
	f0 = 4*np.pi/(2*l+1)
	f1 = L * r2 * np.conj(slm0) * slm2
	f2 = 2 * rk * L * np.conj(slm0) * slm1
	f3 = -(L**2)*( np.conj(slm0)*slm0 ) - (l**2+l+2) * ( np.conj(qlm0)*qlm0 )
	f4 = 2 * rk * np.conj(qlm0)*qlm1 + r2 * np.conj(qlm0) * qlm2
	f5 = 2 * L *( np.conj(qlm0)*slm0 + qlm0*np.conj(slm0) )
 	# integral is
	Dkin_pol_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1+f2+f3+f4+f5 ) )
	
	
	if (projection == 1 and forcing == 0) or forcing == 1: # ------------------- power from Lin2018 forcing, poloidal
		
		if l==2 and m==2:
			f0 = (8*np.pi/35)/(ricb**5-1)
			#f1 = 7j* np.conj(qlm0) *( r3   +  (ricb**5)/r2 ) 
			#f2 = 7j* np.conj(slm0) *( 3*r3 - 2*(ricb**5)/r2 ) 
			f1 = 7j * r3 * ( np.conj(qlm0) + 3*np.conj(slm0) )
			f2 = 7j *(ricb**5)* ( np.conj(qlm0) - 2*np.conj(slm0) )/r2
			power_pol_l =  (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1+f2 ) )
		else:
			power_pol_l =  0
			
	elif (projection == 3 and forcing == 0) or forcing == 3: # ----------------- power from Marc's forcing, poloidal
		
		C0 = 1j; C1 = 1; C2 = 1j
		
		if l == 2:	
			if m == 0:
				f0 = (4*np.pi/105)*C0
				f1 = -21j*w*r4*np.conj(qlm0)
			elif m == 1:
				f0 = (4*np.pi/105)*C1
				f1 = -21j*w*r4*np.conj(qlm0) + 42j*r4*np.conj(slm0)
			elif m == 2:	
				f0 = (-4j*np.pi/35)*C2
				f1 = 7*w*r4*np.conj(qlm0) - 28*r4*np.conj(slm0)
			else:
				f0 = 0
				f1 = 0
			power_pol_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 ) )
		else:
			power_pol_l = 0
					
	elif (projection == 4 and forcing == 0) or forcing == 4: # ----------------- power from forcing test 1, poloidal
		
		X = ut.ftest1(ricb)
		XA = X[0]
		XB = X[1]
		XC = X[2]
		
		if l==2 and m==2:
			f0 = -8j*np.pi*XC/35
			f1 = 7 * XA * r3 * ( np.conj(qlm0) + 3*np.conj(slm0) )
			f2 = 7 * XB * ( np.conj(qlm0) - 2*np.conj(slm0) )/r2		
			power_pol_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 + f2 ) )
		else:
			power_pol_l =  0
					
	elif (projection == 5 and forcing == 0) or forcing == 5: # ----------------- power from forcing test 2, poloidal
		power_pol_l = 0
		
	else:
		power_pol_l = 0
	
	
	return [Ken_pol_l, Dint_pol_l, np.real(Dkin_pol_l), np.imag(Dkin_pol_l),\
	 np.real(power_pol_l), np.imag(power_pol_l)]


	

def tor_worker( l, Tk, N, m, ricb, rcmb, w, projection, forcing): # ------------
	'''
	Here we compute integrals that involve toroidal components, degree l
	Same way as for the poloidals above
	'''
		
	L  = l*(l+1)
	
	dTk  = ut.Dcheb(Tk,ricb,rcmb)
	d2Tk = ut.Dcheb(dTk,ricb,rcmb)
	
	tlm0 = np.zeros(np.shape(xk),dtype=complex)
	tlm1 = np.zeros(np.shape(xk),dtype=complex)
	tlm2 = np.zeros(np.shape(xk),dtype=complex)
	
	tlm0 = np.dot(chx,Tk,tlm0)
	tlm1 = np.dot(chx,dTk,tlm1)
	tlm2 = np.dot(chx,d2Tk,tlm2)
	
	
	# -------------------------------------------------------------------------- kinetic Energy, toroidal
	
	f0 = 4*np.pi/(2*l+1)
	f1 = (r2)*L*np.absolute(tlm0)**2
	# integral is:
	Ken_tor_l = (np.pi/N)*(rcmb-ricb)*0.5 * np.sum( sqx*f0*( f1 ) )
	
	
	# -------------------------------------------------------------------------- internal energy dissipation, toroidal
	
	f0 = 4*np.pi/(2*l+1)
	f1 = L*np.absolute( rk*tlm1-tlm0 )**2
	f2 = L*(l-1)*(l+2)*np.absolute( tlm0 )**2
	# integral is
	Dint_tor_l =  (np.pi/N)*(rcmb-ricb)*0.5 * np.sum( sqx*f0*( f1+f2 ) )
	
	
	# -------------------------------------------------------------------------- kinetic energy dissipation rate, toroidal
	
	f0 = 4*np.pi/(2*l+1)
	f1 = L * r2 * np.conj(tlm0) * tlm2 
	f2 = 2 * rk * L * np.conj(tlm0) * tlm1
	f3 = -(L**2)*( np.conj(tlm0)*tlm0 )
	# integral is:
	Dkin_tor_l = (np.pi/N) * (rcmb-ricb)*0.5 * np.sum( sqx*f0*( f1+f2+f3 ) )
	
	
	if (projection == 1 and forcing == 0) or forcing == 1: # ------------------- power from Lin2018 forcing, toroidal
		
		if l==3 and m==2:
			f0 = (8*np.pi/35)/(ricb**5-1)
			f1 = 10*np.sqrt(5)*(ricb**5)* np.conj(tlm0) /r2
			power_tor_l =  (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 ) )
		else:
			power_tor_l =  0

		
	elif (projection == 3 and forcing == 0) or forcing == 3: # ----------------- power from Marc's forcing, toroidal
		
		C0 = 1j; C1 = 1; C2 = 1j
		
		if l == 1:
			if m == 0:
				f0 = (4*np.pi/105)*C0
				f1 = 28*r4*np.conj(tlm0)
			elif m == 1:
				f0 = (4*np.pi/105)*C1
				f1 = 14*np.sqrt(3)*r4*np.conj(tlm0)
			else:
				f0 = 0
				f1 = 0
			power_tor_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 ) )

		elif l == 3:
			if m == 0:
				f0 = (4*np.pi/105)*C0
				f1 = -72*r4*np.conj(tlm0)
			elif m == 1:
				f0 = (4*np.pi/105)*C1
				f1 = -48*np.sqrt(2)*r4*np.conj(tlm0)
			elif m == 2:
				f0 = (-4j*np.pi/35)*C2
				f1 = -8j*np.sqrt(5)*r4*np.conj(tlm0)
			else:
				f0 = 0
				f1 = 0
			power_tor_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 ) )
		
		else:
			power_tor_l = 0


	elif (projection == 4 and forcing == 0) or forcing == 4: # ----------------- power from forcing test 1, toroidal
		
		X = ut.ftest1(ricb)
		XA = X[0]
		XB = X[1]
		XC = X[2]
		
		if l==3 and m==2:
			f0 = -8j*np.pi*XC/35
			f1 = -10j*np.sqrt(5)*XB*np.conj(tlm0)/r2
			power_tor_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 ) )
		else:
			power_tor_l =  0
			
			
	elif (projection == 5 and forcing == 0) or forcing == 5: # ----------------- Power from forcing test 2, toroidal
		
		if l==1 and m==0:
			f0 = -16*np.pi/105
			f1 = 7 * np.conj(tlm0)
			power_tor_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 ) )
			print('Tor Power =',power_tor_l)
		elif l==3 and m==0:
			f0 = -16*np.pi/105
			f1 = 27* np.conj(tlm0)
			power_tor_l = (np.pi/N)*(rcmb-ricb)*0.5*np.sum( sqx*f0*( f1 ) )
			print('Tor Power =',power_tor_l)			
		else:
			power_tor_l =  0
				
	else:
		power_tor_l = 0
	
	
	return [Ken_tor_l, Dint_tor_l, np.real(Dkin_tor_l), np.imag(Dkin_tor_l),\
	 np.real(power_tor_l), np.imag(power_tor_l)]
		
		
		

def pol_ohm( l, Pk, N, ricb, rcmb): # ------------------------------------------
	
	L  = l*(l+1)
	
	dPk  = ut.Dcheb(Pk,ricb,rcmb)
	d2Pk = ut.Dcheb(dPk,ricb,rcmb)
	
	plm0 = np.zeros(np.shape(xk),dtype=complex)
	plm1 = np.zeros(np.shape(xk),dtype=complex)
	plm2 = np.zeros(np.shape(xk),dtype=complex)
	
	plm0 = np.dot(chx,Pk,plm0)
	plm1 = np.dot(chx,dPk,plm1)
	plm2 = np.dot(chx,d2Pk,plm2)
	

	qlm0 = L*plm0/rk
	qlm1 = (L*plm1 - qlm0)/rk
	
	slm0 = plm1 + (plm0/rk)
	slm1 = plm2 + (qlm1/L)
	
	# -------------------------------------------------------------------------- magnetic field energy, poloidal
	
	f0 = 4*np.pi/(2*l+1)
	f1 = (rk**2)*np.absolute( qlm0 )**2
	f2 = (rk**2)*L*np.absolute( slm0 )**2
	
	benergy_pol_l = 0.5*(rcmb-ricb)*(np.pi/N)*np.sum( sqx*f0*( f1+f2 ) ) 
	
	
	# -------------------------------------------------------------------------- Ohmic dissipation, poloidal
	
	f0 = 4*np.pi*L/(2*l+1)
	f1 = np.absolute( qlm0 - slm0 - rk*slm1 )**2
	
	odis_pol_l = 0.5*(rcmb-ricb)*(np.pi/N)*np.sum( sqx*f0*f1 ) 
	
	
	return [benergy_pol_l, odis_pol_l]


	
def tor_ohm( l, Tk, N, ricb, rcmb): # ------------------------------------------
	
	L  = l*(l+1)

	# Tk is the full set of cheb coefficients for a given l
	# we use the full domain [-1,1] if an inner core is present
	
	dTk  = ut.Dcheb(Tk,ricb,rcmb)

	tlm0 = np.zeros(np.shape(xk),dtype=complex)
	tlm1 = np.zeros(np.shape(xk),dtype=complex)
	
	tlm0 = np.dot(chx,Tk,tlm0)
	tlm1 = np.dot(chx,dTk,tlm1)
	
	# -------------------------------------------------------------------------- magnetic field energy, toroidal
	
	f0 = 4*np.pi/(2*l+1)
	f1 = (rk**2)*L*np.absolute( tlm0 )**2

	benergy_tor_l = 0.5*(rcmb-ricb)*(np.pi/N)*np.sum( sqx*f0*( f1 ) ) 
		
	# -------------------------------------------------------------------------- Ohmic dissipation, toroidal
	
	f0 = 4*np.pi*L/(2*l+1)
	f1 = np.absolute( rk*tlm1 + tlm0 )**2
	f2 = L*np.absolute( tlm0 )**2 
	
	odis_tor_l = 0.5*(rcmb-ricb)*(np.pi/N)*np.sum( sqx*f0*( f1+f2 ) ) 
	
	
	return [benergy_tor_l, odis_tor_l]
	


def ken_dis( a, b, N, lmax, m, symm, ricb, rcmb, ncpus, w, projection, forcing):
	'''
	Computes total kinetic energy, internal and kinetic energy dissipation,
	and input power from body forces.
	'''
	
	if m > 0 :
		symm1 = symm
		if symm == 1:
			m_top = m
			m_bot = m+1				# equatorially symmetric case (symm=1)
			lmax_top = lmax
			lmax_bot = lmax+1
		elif symm == -1:
			m_top = m+1
			m_bot = m				# equatorially antisymmetric case (symm=-1)
			lmax_top = lmax+1
			lmax_bot = lmax
	elif m == 0 :
		symm1 = -symm 
		if symm == 1:
			m_top = 2
			m_bot = 1				# equatorially symmetric case (symm=1)
			lmax_top = lmax+2
			lmax_bot = lmax+1
		elif symm == -1:
			m_top = 1
			m_bot = 2				# equatorially antisymmetric case (symm=-1)
			lmax_top = lmax+1
			lmax_bot = lmax+2
	
	n = int(N*(lmax-m+1)/2)   # use this if there is an inner core
	
	ev0 = a + 1j * b
	Pk0 = ev0[:n] #  N/2 elements on each l block
	Tk0 = ev0[n:n+n] #  N/2 elemens on each l block
	
	# these are the cheb coefficients, reorganized
	Pk2 = np.reshape(Pk0,(int((lmax-m+1)/2),N))
	Tk2 = np.reshape(Tk0,(int((lmax-m+1)/2),N))
	
	Ncut = min(1200,N) 
	
	# process each l component in parallel
	pool = mp.Pool(processes=ncpus)
	
	p = [ pool.apply_async(pol_worker, args=( l, Pk2[k,:Ncut], Ncut, m, ricb, rcmb, w, projection, forcing))\
	 for k,l in enumerate(np.arange(m_top,lmax_top,2)) ]
	
	t = [ pool.apply_async(tor_worker, args=( l, Tk2[k,:Ncut], Ncut, m, ricb, rcmb, w, projection, forcing))\
	 for k,l in enumerate(np.arange(m_bot,lmax_bot,2)) ]
	
	res_pol = np.sum([p1.get() for p1 in p],0)
	res_tor = np.sum([t1.get() for t1 in t],0)
	
	pool.close()
	pool.join()
	
	KP = res_pol[0]
	KT = res_tor[0]
	
	internal_dis = res_pol[1]+res_tor[1]
	rekin_dis = res_pol[2]+res_tor[2]
	imkin_dis = res_pol[3]+res_tor[3]
	
	repower = res_pol[4]+res_tor[4]
	impower = res_pol[5]+res_tor[5]
		
	return [KP, KT, internal_dis, rekin_dis, imkin_dis, repower, impower]



def ohm_dis( a, b, N, lmax, m, bsymm, ricb, rcmb, ncpus):
	'''
	Computes the total energy in the induced magnetic field and the Ohmic dissipation.
	bsymm is the symmetry of the *induced magnetic field*, which is
	opposed to that of the flow if the applied field is antisymmetric.
	'''
	
	if m > 0 :
		bsymm1 = bsymm
		if bsymm == 1:
			m_top = m
			m_bot = m+1				# equatorially symmetric case (symm=1)
			lmax_top = lmax
			lmax_bot = lmax+1
		elif bsymm == -1:
			m_top = m+1
			m_bot = m				# equatorially antisymmetric case (symm=-1)
			lmax_top = lmax+1
			lmax_bot = lmax
	elif m == 0 :
		bsymm1 = -bsymm 
		if bsymm == 1:
			m_top = 2
			m_bot = 1				# equatorially symmetric case (symm=1)
			lmax_top = lmax+2
			lmax_bot = lmax+1
		elif bsymm == -1:
			m_top = 1
			m_bot = 2				# equatorially antisymmetric case (symm=-1)
			lmax_top = lmax+1
			lmax_bot = lmax+2
	
	n = int(N*(lmax-m+1)/2)
	# Use n=(N/2)*(lmax-m+1)/2 if there is no inner core
	
	ev0 = a + 1j*b
	Pk0 = ev0[0:  n] #  N/2 elements on each l block
	Tk0 = ev0[n:2*n] #  N/2 elemens on each l block
	
	# these are the cheb coefficients, reorganized
	Pk2 = np.reshape(Pk0,(int((lmax-m+1)/2),N))
	Tk2 = np.reshape(Tk0,(int((lmax-m+1)/2),N))
	
	# process each l component in parallel
	pool = mp.Pool(processes=ncpus)
	p = [ pool.apply_async(pol_ohm,args=(l, Pk2[k,:], N, ricb, rcmb)) for k,l in enumerate(np.arange(m_top,lmax_top,2.)) ]
	t = [ pool.apply_async(tor_ohm,args=(l, Tk2[k,:], N, ricb, rcmb)) for k,l in enumerate(np.arange(m_bot,lmax_bot,2.)) ]
	
	res_pol = np.sum([p1.get() for p1 in p],0)
	res_tor = np.sum([t1.get() for t1 in t],0)
	
	pool.close()
	pool.join()
	
	MagEnerPol = res_pol[0]
	MagEnerTor = res_tor[0]
	
	OhmDissPol = res_pol[1]
	OhmDissTor = res_tor[1]
	
	
	return np.real([MagEnerPol, MagEnerTor, OhmDissPol, OhmDissTor])


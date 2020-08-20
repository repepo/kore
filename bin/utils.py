# Various function definitions and utilities

from scipy.linalg import toeplitz
from scipy.linalg import hankel
import scipy.sparse as ss
import scipy.special as scsp
import scipy.fftpack as sft
import scipy.misc as sm
import numpy.polynomial.chebyshev as ch
import numpy as np

import parameters as par
#import bc_variables as bv


# some global variables

if par.forcing == 0:
	wf = 0
else:
	wf = par.forcing_frequency
rcmb = 1
n    = int(par.N*(par.lmax-par.m+1)/2)

if par.m > 0 :
	symm1 = par.symm
	if par.symm == 1:						# m>0, equatorially symmetric case (symm=1)
		m_top = par.m
		m_bot = par.m + 1						
		lmax_top = par.lmax
		lmax_bot = par.lmax + 1
	elif par.symm == -1:					# m>0, equatorially antisymmetric case (symm=-1)
		m_top = par.m + 1
		m_bot = par.m				
		lmax_top = par.lmax + 1
		lmax_bot = par.lmax
elif par.m == 0 :
	symm1 = -par.symm 
	if par.symm == 1:						# m=0, equatorially symmetric case (symm=1)
		m_top = 2
		m_bot = 1						
		lmax_top = par.lmax + 2
		lmax_bot = par.lmax + 1
	elif par.symm == -1:					# m=0, equatorially antisymmetric case (symm=-1)
		m_top = 1
		m_bot = 2				
		lmax_top = par.lmax + 1
		lmax_bot = par.lmax + 2

'''
Special arrangement schematic for m=0:

symm	antisymm
P0		P1					P2		P1
P2		P3					P4		P3
P4		P5			-->		P6		P5
T1		T0					T1		T2
T3		T2					T3		T4
T5		T4					T5		T6

'''

# this gives the size (rows or columns) of the main matrices
sizmat = 2*n + 2*n*par.magnetic + n*par.thermal



def chebco(powr, N, tol, ricb, rcmb):
	'''
	Returns the first N Chebyshev coefficients
	from 0 to N-1, of the function
	( ricb + (rcmb-ricb)*( x + 1 )/2. )**powr
	'''
	i = np.arange(0,N)
	xi = np.cos(np.pi*(i+0.5)/N)
	ai = ( ricb + (rcmb-ricb)*(xi+1)/2. )**powr
	tmp = sft.dct(ai)
	
	out = tmp/N
	out[0]=out[0]/2.
	out[np.absolute(out)<=tol]=0.
	
	return out

	

def Dlam(lamb,N):
	'''
	Order lamb (>=1) derivative matrix, size N*N
	'''
	const1 = (2./(rcmb-par.ricb))**lamb
	const2 = sm.factorial(lamb-1.)*2**(lamb-1.)
	tmp = lamb + np.arange(0,N-lamb)
    
	return const1*const2*ss.diags(tmp,lamb, format='csr')
	
	
	
def Slam(lamb,N):
	'''
	Converts C^(lamb) series coefficients to C^(lamb+1) series
	'''
	
	if lamb == 0:
		diag0 = 0.5*np.ones(N); diag0[0]=1.
		diag1 = -0.5*np.ones(N-2)
	else:
		tmp = np.arange(0.,N)
		diag0 = lamb/(lamb+tmp)
		diag1 = -lamb/(lamb+tmp[2:])
	
	return ss.diags([diag0,diag1],[0,2], format='csr')
	
	
	
def csl0( s, lamb, j, k):
	'''
	Computes the c_s^lambda(j,k) needed for the Mlam (multiplication) matrix
	'''
	
	p1=1; p3=1
	for t in range(0,s):
		p1 = p1*(lamb+t)/float(1+t)
		p3 = p3*(2*lamb+j+k-2*s+t)/float(lamb+j+k-2*s+t)
	
	p2=1; p4=1
	for t in range(0,j-s):
		p2 = p2*(lamb+t)/float(1+t)
		p4 = p4*(k-s+1+t)/float(k-s+lamb+t)		
		
	return p1*p2*p3*p4*(j+k+lamb-2.*s)/float(j+k+lamb-s)
		


def csl(svec,lamb,j,k):
	'''
	recursion for c_s^lambda, starting from c_svec[0]^lambda(j,k)
	svec must be a vector of s values
	**do not confuse with the (j,k) entry of the Mlam matrix**
	'''
	out = np.zeros(np.shape(svec))
	out[0] = csl0(svec[0], lamb, j, k)
	for i,s in enumerate(svec[0:-1],1) :
		tmp1 = (j+k+lamb-s)*(lamb+s)*(j-s)*(2*lamb+j+k-s)*(k-s+lamb)
		tmp2 = (j+k+lamb-s+1)*(s+1)*(lamb+j-s-1)*(lamb+j+k-s)*(k-s+1)
		out[i] = out[i-1]*tmp1/float(tmp2)
		k=k+2

	return out
	
	
	
def Mlam(a0,lamb):
	'''
	Multiplication matrix. a0 are the cofficients in the C^(lamb) basis and lamb
	is the order of the C^(lamb) basis. (This basis should match the
	one from the highest derivative order appearing in the equation)
	'''	
	N = np.size(a0)
	bw = max(np.nonzero(a0)[0])

	a1 = np.zeros(2*N)
	a1[:N] = a0
	
	if lamb > 0:
		
		out = ss.dok_matrix((N,N))
		for j in range(0,N):
			
			k1 = max( 0, j-bw-1 )
			k2 = min( N, j+bw+2 )
			
			for k in range(k1,k2):
				
				s0 = max(0,k-j)
				s = np.arange(s0,k+1)
				idx = 2*s+j-k
				a = a1[idx]
				
				if s0 == 0:
					cvec = csl(s,lamb,k,j-k)
				elif s0 == k-j:
					cvec = csl(s,lamb,k,k-j)
				
				out[j,k] = np.dot(a,cvec)
				
		out = out.tocsr()
				
	else:
		
		a2 = np.copy(a0);
		a2[0] = 2*a2[0]
		tmp1 = toeplitz(a2)
		tmp2 = hankel(a2)
		tmp2[0,:]=np.zeros(N)
		
		tmp = 0.5*(tmp1+tmp2)
		out = ss.csr_matrix(tmp)
	'''
	The case lamb = 1 should be reducible too to a Hankel+Toeplitz
	not done yet
	'''	
	return out
	


def Dcheb(ck, ricb, rcmb):
	'''
	The derivative of a Chebyshev expansion with coefficients ck
	returns the coefficients of the derivative in the Chebyshev basis
	assumes ck computed for r in the domain [ricb,rcmb]
	''' 
	c = np.copy(ck)
	c[0] = 2.*c[0]
	s =  np.size(c)
	out = np.zeros(s,dtype=np.complex128)
	out[-2] = 2.*(s-1.)*c[-1]
	for k in range(s-3,-1,-1):
		out[k] = out[k+2] + 2.*(k+1)*ck[k+1]
	out[0] = out[0]/2.
		
	return 2*out/(rcmb-ricb)
	
	
	
def som(alpha,E):
	'''
	theoretical (asymptotic)
	spin-over mode frequency in an ellipsoid
	viscous, no-slip, from Zhang 2004
	'''	
	eps2 = 2*alpha - alpha**2
	reG = -2.62047 - 0.42634 * eps2
	imG = 0.25846 + 0.76633 * eps2
	sigma = 1./(2.-eps2) #inviscid freq
	G = reG + 1.j*imG

	return  1j*( 2*sigma - 1j*G * E**0.5 )



def cw( q, alpha):
	'''
	theoretical Chandler Wobble frequency
	'''
	q1 = 8.*np.pi*q/5.
	out = (q1 + 1.) * alpha * ( 1. + 3.*alpha/14.)
	
	return -out
	

	
def fcn( q, alpha):
	'''
	theoretical FCN frequency
	'''
	q1 = 8.*np.pi*q/5.
	out = -1. - (q1 + 1.) * alpha * ( 1. + 3.*alpha/14.)
	
	return -out	



def marc_tide(omega, l, m, loc, N, ricb, rcmb):
	'''
	Tidal body force as used by Rovira-Navarro et al, 2018
	'''
	
	C0 = 1j; C1 = 1; C2 = 1j
	
	if l==2 and loc == 'top':
		r5 = chebco(5, N-4, 2e-16, ricb, rcmb)
				
		if m == 0:
			out = -6j*C0*omega*r5
		if m == 1:
			out = -6j*C1*(omega+1)*r5
		if m == 2:
			out = -6j*C2*(omega+2)*r5
	
	elif loc == 'bot':		
		r4 = chebco(4, N-2, 2e-16, ricb, rcmb)
		
		if m == 0:
			if l == 1:
				out = (4/5)*C0*r4
			elif l == 3:
				out = (-24/5)*C0*r4
				
		elif m == 1:
			if l == 1:
				out = (2/5)*np.sqrt(3)*C1*r4
			elif l == 3:
				out = -(16/5)*np.sqrt(2)*C1*r4
		
		elif m == 2:
			if l == 3:
				out = -(8/np.sqrt(5))*C2*r4
				
	return out



def eccen_tide(m, eta, boundary):
	'''
	Eccentricity tide as boundary forcing.
	Computed by Jeremy
	eta = ricb
	'''
	
	if boundary == 'cmb':
	
		if m == -2:
		
			P = (3.00927601455473e-7 + 9.923006738722803e-7*(-0.8809523809523809 + eta) + 2.0022902696912132e-6*(-0.8809523809523809 + eta)**2 \
			+ 2.078150971132207e-6*(-0.8809523809523809 + eta)**3)/(0.9796806966104893 + 10.138063592395676*(-0.8809523809523809 + eta) \
			+ 22.793605369253733*(-0.8809523809523809 + eta)**2 + 49.52238501116611*(-0.8809523809523809 + eta)**3)

			dP = (3.6923409583928235e-8 + 5.525597059314317e-7*(-0.8809523809523809 + eta) + 1.491072486300637e-6*(-0.8809523809523809 + eta)**2 \
			+ 1.7131360325476716e-6*(-0.8809523809523809 + eta)**3)/(0.990016066975477 + 10.903688391808103*(-0.8809523809523809 + eta) \
			+ 28.973467794986213*(-0.8809523809523809 + eta)**2 + 54.35051022571307*(-0.8809523809523809 + eta)**3)

		elif m == 0:
			
			P = (-1.0530272472650882e-7 - 3.4723290320109935e-7*(-0.8809523809523809 + eta) - 7.006556396692487e-7*(-0.8809523809523809 + eta)**2 \
			- 7.272013553918712e-7*(-0.8809523809523809 + eta)**3)/(0.9796806966105374 + 10.138063592396957*(-0.8809523809523809 + eta) \
			+ 22.793605369262107*(-0.8809523809523809 + eta)**2 + 49.52238501117491*(-0.8809523809523809 + eta)**3)

			dP = (-1.2920501863486681e-8 - 1.933556188506115e-7*(-0.8809523809523809 + eta) - 5.217666801343201e-7*(-0.8809523809523809 + eta)**2 \
			- 5.994727342457128e-7*(-0.8809523809523809 + eta)**3)/(0.990016066975396 + 10.903688391807878*(-0.8809523809523809 + eta) \
			+ 28.973467794991336*(-0.8809523809523809 + eta)**2 + 54.35051022572941*(-0.8809523809523809 + eta)**3)

		elif m == 2:
			
			P = (-4.2989657350784934e-8 - 1.4175723912463045e-7*(-0.8809523809523809 + eta) - 2.860414670987557e-7*(-0.8809523809523809 + eta)**2 \
			- 2.9687871016172726e-7*(-0.8809523809523809 + eta)**3)/(0.9796806966105591 + 10.1380635923966*(-0.8809523809523809 + eta) \
			+ 22.793605369256564*(-0.8809523809523809 + eta)**2 + 49.522385011165795*(-0.8809523809523809 + eta)**3)

			dP = (-5.274772797703058e-9 - 7.893710084733917e-8*(-0.8809523809523809 + eta) - 2.1301035518586083e-7*(-0.8809523809523809 + eta)**2 \
			- 2.4473371893559314e-7*(-0.8809523809523809 + eta)**3)/(0.990016066975294 + 10.903688391807277*(-0.8809523809523809 + eta) \
			+ 28.973467794994058*(-0.8809523809523809 + eta)**2 + 54.35051022573984*(-0.8809523809523809 + eta)**3)

	elif boundary == 'icb':

		if m == -2:
			
			P = (1.450645275222846e-9 + 1.6226580438374377e-8*(-0.8809523809523809 + eta) + 3.8162533600883e-8*(-0.8809523809523809 + eta)**2 \
			+ 2.1621482543044536e-8*(-0.8809523809523809 + eta)**3)/(1.0040474189168584 + 9.816014125318514*(-0.8809523809523809 + eta) \
			+ 12.467747302017761*(-0.8809523809523809 + eta)**2 - 0.13326826070343542*(-0.8809523809523809 + eta)**3)

			dP = (5.690951704457902e-11 + 6.365766130462465e-10*(-0.8809523809523809 + eta) + 1.4971346844873185e-9*(-0.8809523809523809 + eta)**2 \
			+ 8.482212366654875e-10*(-0.8809523809523809 + eta)**3)/(1.0040474189168667 + 9.816014125318599*(-0.8809523809523809 + eta) \
			+ 12.467747302017914*(-0.8809523809523809 + eta)**2 - 0.13326826070340814*(-0.8809523809523809 + eta)**3)

		elif m == 0:
			
			P = (-5.076201031536069e-10 - 5.678120334892032e-9*(-0.8809523809523809 + eta) - 1.3354104944854432e-8*(-0.8809523809523809 + eta)**2 \
			- 7.565942816136093e-9*(-0.8809523809523809 + eta)**3)/(1.0040474189168602 + 9.81601412531852*(-0.8809523809523809 + eta) \
			+ 12.467747302017683*(-0.8809523809523809 + eta)**2 - 0.13326826070345546*(-0.8809523809523809 + eta)**3)

			dP = (-1.991418260963427e-11 - 2.22755412021775e-10*(-0.8809523809523809 + eta) - 5.238880076023669e-10*(-0.8809523809523809 + eta)**2 \
			- 2.96815602688989e-10*(-0.8809523809523809 + eta)**3)/(1.0040474189168587 + 9.816014125318501*(-0.8809523809523809 + eta) \
			+ 12.46774730201766*(-0.8809523809523809 + eta)**2 - 0.13326826070345907*(-0.8809523809523809 + eta)**3)

		elif m == 2:
			
			P = (-2.072350393175482e-10 - 2.3180829197677505e-9*(-0.8809523809523809 + eta) - 5.451790514411792e-9*(-0.8809523809523809 + eta)**2 \
			- 3.0887832204348696e-9*(-0.8809523809523809 + eta)**3)/(1.0040474189168525 + 9.816014125318436*(-0.8809523809523809 + eta) \
			+ 12.46774730201756*(-0.8809523809523809 + eta)**2 - 0.13326826070347508*(-0.8809523809523809 + eta)**3)

			dP = (-8.129931006368303e-12 - 9.093951614946213e-11*(-0.8809523809523809 + eta) - 2.138763834981828e-10*(-0.8809523809523809 + eta)**2 \
			- 1.2117446238077878e-10*(-0.8809523809523809 + eta)**3)/(1.0040474189168571 + 9.816014125318477*(-0.8809523809523809 + eta) \
			+ 12.467747302017562*(-0.8809523809523809 + eta)**2 - 0.13326826070347858*(-0.8809523809523809 + eta)**3)

	return np.array([P,dP])
	

	
def ftest1(ricb):
	'''
	Tidal body force constants for Enceladus
	Computed by Jeremy 
	'''
	
	A = (2.776146590187793e-8 - 1.1936189474389837e-7*ricb + 1.7976942241610017e-7*ricb**2 - 1.0431683766634872e-7*ricb**3 \
	+ 1.0203567472726217e-8*ricb**4 + 5.944217795551174e-9*ricb**5)/(2.039370654683117 - 12.426922103588517*ricb \
	+ 29.534185532382605*ricb**2 - 33.58608820683908*ricb**3 + 17.080450537622824*ricb**4 - 1.6409964140402016*ricb**5 - 1.*ricb**6)

	B = (7.525664633338715e-9 - 5.8895735957722854e-8*ricb + 1.8268212761498169e-7*ricb**2 - 2.798206106175817e-7*ricb**3 \
	+ 2.1103777701038198e-7*ricb**4 - 6.252914009737955e-8*ricb**5)/(2.039370654683117 - 12.426922103588517*ricb \
	+ 29.534185532382605*ricb**2 - 33.58608820683908*ricb**3 + 17.080450537622824*ricb**4 - 1.6409964140402016*ricb**5 - 1.*ricb**6)
	
	C = 1e6
	
	#A = 1.0
	#B = ricb**5
	#C = 1/(1-ricb**5)
	
	return np.array([A,B,C])
	
	
	
def Ylm(l, m, theta, phi):
	# The Spherical Harmonics, seminormalized
	out = scsp.sph_harm(m, l, phi, theta)
	return out*np.sqrt(4*np.pi/(2*l+1))
	


def Ylm_full(lmax, m, theta, phi):
	# array of Spherical Harmonics with a range of l's
	m1 = max(m,1)
	if m == 0 :
		lmax1 = lmax+1
	else:
		lmax1 = lmax
	out = np.zeros(lmax-m+1,dtype=np.complex128)
	for l in np.arange(m1,lmax1+1):
		out[l-m1]=Ylm(l,m,theta,phi)
	return out



def Ylm_symm(lmax, m, theta, phi, symm, scalar):
	out = np.zeros((lmax-m+1)/2,dtype=np.complex128)
	if (symm == 1 and scalar == 'pol') or (symm == -1 and scalar == 'tor'):	
		m_sym = m				
		lmax_sym = lmax
	else :
		m_sym = m+1					
		lmax_sym = lmax+1
	for l in np.arange(m_sym,lmax_sym,2.):
		out[(l-m_sym)/2]=Ylm(l,m,theta,phi)
	return out



def load_csr(filename):
	# utility to load sparse matrices efficiently 
	loader = np.load(filename)
	return ss.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])




def Tk(x, N, lamb_max):
	'''
	Chebyshev polynomial from order 0 to N (as rows)
	and its derivatives ** with respect to r **, up to lamb_max (as columns),
	evaluated at x=-1 (r=ricb) or x=1 (r=rcmb).
	'''
	
	out = np.zeros((N+1,lamb_max+1))
	
	for k in range(0,N+1):
		
		out[k,0] = x**k
		
		tmp = 1.
		for i in range(0, lamb_max):
			tmp = tmp * ( k**2 - i**2 )/( 2.*i + 1. )
			out[k,i+1] = x**(k+i+1.) * tmp * (2./(rcmb-par.ricb))**(i+1)
		
	return out




def gamma_visc(a1,a2,a3):

	out = np.zeros((1,n+n),dtype=complex) 
		
	
	#Tb3 = copy(Tb2)
	#Tb3[nc:,:] = 0
	#Tb3 = np.copy(bv.Tb)
	Tb3 = Tk( 1, par.N-1, 5)
	
	
	P0 = Tb3[:,0]
	P1 = Tb3[:,1]
	P2 = Tb3[:,2]
	P3 = Tb3[:,3]
	P4 = Tb3[:,4]
	P5 = Tb3[:,5]

	T0 = Tb3[:,0]
	T1 = Tb3[:,1]
	T2 = Tb3[:,2]
	T3 = Tb3[:,3]
	T4 = Tb3[:,4]
	
		
	# this is for a spheroid with long semiaxis a=1
	
	I = 1j
	
	pol2 = (  a1*((0. + 24.624956739107787*I)*P0 - (0. + 24.624956739107787*I)*P1 + (0. + 12.312478369553894 *I)*P2 
				+ (0. + 4.104159456517965 *I)*P3) 
			+ a2*((0. + 5.2767764440945255*I)*P0 - (0. + 5.2767764440945255*I)*P1 + (0. + 2.6383882220472628 *I)*P2 
				- (0. + 9.67409014750663  *I)*P3 - (0. + 1.758925481364842 *I)*P4) 
			+ a3*((0. - 1.758925481364842 *I)*P0 + (0. + 1.758925481364842 *I)*P1 - (0. + 0.879462740682421  *I)*P2 
				+ (0. + 0.2931542468941404*I)*P3 + (0. + 3.517850962729684 *I)*P4 + (0. + 0.4885904114902339 *I)*P5) )
			
	pol4 = (  a2*((0. - 42.817918360629186*I)*P0 + (0. + 42.817918360629186*I)*P1 + (0. + 3.5681598633857656 *I)*P2 
				- (0. + 6.422687754094378 *I)*P3 - (0. + 0.7136319726771531*I)*P4)  
			+ a3*((0. - 31.140304262275777*I)*P0 + (0. + 31.140304262275777*I)*P1 - (0. + 20.111446502719772 *I)*P2 
				+ (0. + 2.465274087430165 *I)*P3 + (0. + 3.6979111311452484*I)*P4 + (0. + 0.324378169398706  *I)*P5))
			
	pol6 = (  a3*((0. + 45.56049760657571*I)*P0 - (0. + 45.56049760657571  *I)*P1 - (0. + 22.780248803287854 *I)*P2 
				+ (0. + 3.254321257612551*I)*P3 + (0. + 1.30172850304502   *I)*P4 + (0. + 0.07231825016916779*I)*P5))
			
	tol1 = ( -11.847687835088976     *T0 + 11.847687835088976*T1 
			+ a1*(9.47815026807118   *T0 - 9.47815026807118  *T1 - 4.73907513403559  *T2) 
			+ a2*(2.031032200300967  *T0 - 2.031032200300967 *T1 + 5.077580500752418 *T2 + 1.5232741502257257 *T3) 
			+ a3*(0.4513404889557705 *T0 - 0.4513404889557705*T1 - 1.8241678095295728*T3 - 0.37611707412980877*T4))
			
	tol3 = (  a1*(39.799940335196546 *T0 - 6.633323389199425 *T1 - 3.3166616945997127*T2) 
			+ a2*(-19.899970167598273*T0 - 13.26664677839885 *T1 + 8.291654236499282 *T2 + 1.6583308472998564 *T3) 
			+ a3*(-14.472705576435107*T0 + 7.93988708707204  *T1 + 2.010097996727099 *T2 - 3.5679239441906003 *T3 - 0.5025244991817748  *T4))
			
	tol5 = (  a2*(-57.208391908193846*T0 - 9.534731984698974 *T1 + 3.8138927938795897*T2 + 0.4767365992349487 *T3) 
			+ a3*(4.400645531399526  *T0 + 34.960683943896235*T1 + 0.6845448604399262*T2 - 2.725955426394706  *T3 - 0.24448030729997366 *T4))
			
	tol7 = (  a3*(59.85704517989213  *T0 + 24.316924604331177*T1 - 0.8016568550878411*T2 - 0.7571203631385165 *T3 - 0.044536491949324505*T4))
	
	
	# assemble the torque (row vector)
		
	for l in np.arange(m_top,lmax_top,2.):
		
		colP = int( (par.N)*(l-m_top)/2 )
		
		if l==2 and par.m==1:
			out[0,colP:colP+par.N] = pol2
		elif l==4 and par.m==1:
			out[0,colP:colP+par.N] = pol4
		elif l==6 and par.m==1:
			out[0,colP:colP+par.N] = pol6

	for l in np.arange(m_bot,lmax_bot,2.):
		
		colT = n + int( (par.N)*(l-m_bot)/2 )
		
		if l==1 and par.m==1:
			out[0,colT:colT+par.N] = tol1
		elif l==3 and par.m==1:
			out[0,colT:colT+par.N] = tol3
		elif l==5 and par.m==1:
			out[0,colT:colT+par.N] = tol5
		elif l==7 and par.m==1:
			out[0,colT:colT+par.N] = tol7
			
	
	return out



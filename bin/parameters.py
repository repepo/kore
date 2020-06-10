import numpy as np

def som(alpha,E):
	'''
	spin-over freq and damping according to Zhang (2004)
	'''
	eps2 = 2*alpha - alpha**2
	reG = -2.62047 - 0.42634 * eps2
	imG = 0.25846 + 0.76633 * eps2
	sigma = 1./(2.-eps2) #inviscid freq
	G = reG + 1.j*imG
	return  1j*( 2*sigma - 1j*G * E**0.5 )

def Ncheb(Ek):
	'''
	Returns the truncation level N for the Chebyshev expansion according to the Ekman number
	'''
	x=-np.log10(Ek)
	if x>=9 :
		out = 380*x-2700
	else:
		out = 104*x-216
	return int(1.1*out)

def wattr(n,Ek):
	'''
	Useful to set targets when reproducing Figs 3 and 4 of Rieutord & Valdettaro, JFM (2018)
	See also equation (3.2) in that paper
	'''
	w0 = 0.782413
	tau1 = 0.485
	phi1 = -np.pi/3
	tau2 = 1.82
	phi2 = -np.pi/4
	out0 = 1j*w0
	out1 = -2*tau1*( np.cos(phi1)+1j*np.sin(phi1) )*(Ek**(1/3))
	out2 = -(n+0.5)*np.sqrt(2)*tau2*( np.cos(phi2)+1j*np.sin(phi2) )*(Ek**(1/2))
	return out0+out1+out2



# -------------------------------------------------- Physical parameters

# Azimuthal wave number m (>=0)
m = 1

# For equatorially symmetric modes set symm = 1. 
# Set symm = -1 for antisymmetric.
symm = -1

# Inner core radius, CMB radius is one. Use bci = 2 below if ricb = 0.
# Do not set ricb = 0 unless the regularity condition is implemented
ricb = 0.35

# Inner core spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
# Use 2 for no inner core (regularity condition), *not implemented here*
bci = 1

# CMB spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
bco = 1

# Ekman number
Ek = 10**-4

forcing = 0  # For eigenvalue problems
# forcing = 1  # For Lin & Ogilvie 2018 tidal body force, m=2, symm. OK
# forcing = 2  # For boundary flow forcing, use with bci=1 and bco=1.
# forcing = 3  # For Rovira-Navarro 2018 tidal body forcing, m=0,2 must be symm, m=1 antisymm. Leaks power!
# forcing = 4  # first test case, Lin body forcing with X(r)=(A*r^2 + B/r^3)*C, (using Jeremy's calculation), m=2,symm. OK
# forcing = 5  # second test case, Lin body forcing with X(r)=1/r, m=0,symm. OK
# forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
# forcing = 7  # Longitudinal libration boundary forcing, m=0, symm, no-slip

# Use this when solving a forced problem 
freq0 = -0.9975
delta = 0
forcing_frequency = freq0 + delta  # negative is prograde
forcing_amplitude = 1.0 

# if solving an eigenvalue problem, compute projection of eigenmode
# and some hypothetical forcing. Cases as described above (use only 1,3 or 4)
projection = 1

# Whether to include magnetic fields (imposes vertical uniform field)
# magnetic = 0 solves the purely hydrodynamical problem.
magnetic = 0

# Elsasser number
Lambda =10**0.4

# Magnetic Ekman number
Pm = 10**-5.5
Em = Ek/Pm 
#Em = 10*Ek**(2/3)
#Em = 1e-6

# Lehnert number
# Le = 10**-0.5
Le2 = Lambda*Ek/Pm
# Le2 = Le**2
Le = np.sqrt(Le2)

# writes eigenvalue or solution vector to disk if = 1	
write_eig = 0



# ----------------------------------------------------------- Resolution

# Number of cpus
ncpus = 4

# Truncation level
N = Ncheb(Ek) 

# Approx lmax/N ratio
g = 1.0  
lmax = int( 2*ncpus*( np.floor_divide( g*N, 2*ncpus ) ) + m - 1 )

# Max angular degree lmax, must be even if m is odd,
# and lmax-m+1 should be divisible by 2*ncpus 
# lmax = (ncpus*2) * N0 + m - 1



# -------------------------------------------- Eigenvalue solver options

# Set track_target = 1 to track an eigenvalue
# assumes a preexisting 'track_target' file with target data
# Set track_target = 2 to write initial 'track_target' filem, see also solve.py
track_target = 0
# track_target = 1
# track_target = 2


if track_target == 1 :  # read target from file and sets target accordingly
    tt = np.loadtxt('track_target')
    rtau = tt[0]
    itau = tt[1]
else:                   # set target manually
    rtau = 0
    itau = 1

# tau is the actual target for the solver
# real part is damping
# imaginary part is frequency (positive is retrograde)
tau = rtau + itau*1j
#tau = 2*(wattr(n0,Ek/2))

which_eigenpairs = 'TM'
# L/S/T & M/R/I
# L largest, S smallest, T target
# M magnitude, R real, I imaginary

# Number of desired eigenvalues
nev = 4

# Number of vectors in Krylov space for solver
# ncv = 100

# Maximum iterations to converge to an eigenvector
maxit = 30

# Tolerance for solver
tol = 1e-13

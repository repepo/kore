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
	return int(1.0*out)

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



# ------------------------------------------------------------------------------ Physical parameters

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
bci = 0

# CMB spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
bco = 1

# Ekman number (use 2* to match Dintrans 1999)
Ek = 10**-3

forcing = 0  # For eigenvalue problems
# forcing = 1  # For Lin & Ogilvie 2018 tidal body force, m=2, symm. OK
# forcing = 2  # For boundary flow forcing, use with bci=1 and bco=1.
# forcing = 3  # For Rovira-Navarro 2018 tidal body forcing, m=0,2 must be symm, m=1 antisymm. Leaks power!
# forcing = 4  # first test case, Lin body forcing with X(r)=(A*r^2 + B/r^3)*C, (using Jeremy's calculation), m=2,symm. OK
# forcing = 5  # second test case, Lin body forcing with X(r)=1/r, m=0,symm. OK
# forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
# forcing = 7  # Longitudinal libration boundary forcing, m=0, symm, no-slip

# Forcing frequency and amplitude (ignored if forcing = 0)
freq0 = -0.9975
delta = 0
forcing_frequency = freq0 + delta  # negative is prograde
forcing_amplitude = 1.0 

# if solving an eigenvalue problem, compute projection of eigenmode
# and some hypothetical forcing. Cases as described above (use only 1,3 or 4)
projection = 1



# ------------------------------------------------------------------------ Magnetic field parameters

# magnetic = 0   # solves the purely hydrodynamical problem.
magnetic = 1   # soves the MHD problem, with imposed axial field

# magnetic boundary conditions on the ICB:
# innercore = 'insulator'
# innercore = 'perfect conductor, material'  # tangential *material* electric field jump [nxE']=0 across the ICB
innercore = 'perfect conductor, spatial'   # tangential *spatial* electric field jump [nxE]=0 across the ICB
# Note: 'material' or 'spatial' are identical if ICB is no-slip (bci = 1 above)

# Magnetic field strength and magnetic diffusivity:
# Either use the Elsasser number and magnetic Prandtl number (uncomment the following three lines):
# Lambda = 0.01
# Pm = 10**-4
# Em = Ek/Pm; Le2 = Lambda*Em; Le = np.sqrt(Le2)
# Or use the Lehnert number and magnetic Ekman number (uncomment the following three lines):
Le = 0.01
Em = 1e-3
Le2 = Le**2



# ------------------------------------------------------------------------------- Thermal parameters

thermal = 0
# thermal = 1

# Background temperature gradient (following Dormy 2004)
heating = 'internal'       # internal heating,     dT/dr = r
# heating = 'differential'   # differential heating, dT/dr = r**-2

# Dimensionless Brunt-Vaisala frequency (use 2* to match Dintrans 1999)
Brunt = 2*2.5

# Prandtl number
Prandtl = 1.0

# Thermal boundary conditions
# 0 for isothermal, theta=0
# 1 for constant heat flux, (d/dr)theta=0
bci_thermal = 0   # ICB
bco_thermal = 0   # CMB



# ------------------------------------------------------------ writes solution vector to disk if = 1	
write_eig = 0
# write_eig = 1



# --------------------------------------------------------------------------------------- Resolution

# Number of cpus
ncpus = 24

# Chebyshev polynomial truncation level
N = Ncheb(Ek) 
# N = 20

# Spherical harmonic truncation lmax and approx lmax/N ratio:
g = 1.0  
lmax = int( 2*ncpus*( np.floor_divide( g*N, 2*ncpus ) ) + m - 1 )

# If manually setting the max angular degree lmax, then it must be even if m is odd,
# and lmax-m+1 should be divisible by 2*ncpus 
# lmax = 8



# ----------------------------------------------------------------------------------- Solver options

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
# tau = 2*(wattr(n0,Ek/2))

which_eigenpairs = 'TM'
# L/S/T & M/R/I
# L largest, S smallest, T target
# M magnitude, R real, I imaginary

# Number of desired eigenvalues
nev = 3

# Number of vectors in Krylov space for solver
# ncv = 100

# Maximum iterations to converge to an eigenvector
maxit = 50

# Tolerance for solver
tol = 1e-13

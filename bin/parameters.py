import numpy as np


# -------------------------------------------------- Physical parameters

# Azimuthal wave number
m = 0

# For equatorially symmetric modes set symm = 1. 
# Set symm = -1 for antisymmetric.
symm = 1

# Inner core radius, CMB radius is one. Use bci = 2 below if ricb = 0.
# Do not set ricb = 0 unless the regularity condition is implemented,
ricb = 0.834

# Inner core spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
# Use 2 for no inner core (regularity condition), *not implemented here*
bci = 1

# CMB spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
bco = 1

# Ekman number
Ek = 10**-5

forcing = 0  # For eigenvalue problems
# forcing = 1  # For Lin & Ogilvie 2018 tidal body force, m=2, symm. OK
# forcing = 2  # For boundary flow forcing, use with bci=1 and bco=1.
# forcing = 3  # For Rovira-Navarro 2018 tidal body forcing, m=0,2 must be symm, m=1 antisymm. Leaks power!
# forcing = 4  # first test case, Lin body forcing with X(r)=(A*r^2 + B/r^3)*C, (using Jeremy's calculation), m=2,symm. OK
# forcing = 5  # second test case, Lin body forcing with X(r)=1/r, m=0,symm. OK
# forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
# forcing = 7  # Longitudinal libration boundary forcing, m=0, symm, no-slip

freq0 = 1.0
delta = 0
forcing_frequency = freq0 + delta  # negative is prograde
forcing_amplitude = 1.0

# if solving an eigenvalue problem, compute projection of eigenmode
# and some hypothetical forcing. Cases as described above (use only 1,3 or 4)
projection = 1

# Whether to include magnetic fields (imposes vertical uniform field)
# magnetic = 0 solves the purely hydrodynamical problem.
magnetic = 0

# Lehnert number
Le = 1e-3
Le2 = Le**2

#Magnetic Ekman number
Em = 10**-7

# writes eigenvalue or solution vector to disk if = 1	
write_eig = 0	 



# ----------------------------------------------------------- Resolution

N0 = min( 100, max( 10, int(np.log(2e-22*Ek**-7)/2) ) )
# Max angular degree, must be even if m is odd,
# and lmax-m+1 should be divisible by 2*ncpus 
lmax = 24*N0 + m -1
#lmax = 8+m-1
# Max Chebyshev polynomials order
N = 20*N0
#N = 20
# Number of cpus (used only for postprocessing)
ncpus = 4



# -------------------------------------------- Eigenvalue solver options

# target eigenvalue
# real part is damping
# imaginary part is frequency (positive is retrograde)
rtau = -0.05
itau = 0.94
tau = rtau + itau*1j

which_eigenpairs = 'TM'
# L/S/T & M/R/I
# L largest, S smallest, T target
# M magnitude, R real, I imaginary

# Number of desired eigenvalues
nev = 1

# Number of vectors in Krylov space for solver
# ncv = 100

# Maximum iterations to converge to an eigenvector
maxit = 180

# Tolerance for solver
tol = 1e-18

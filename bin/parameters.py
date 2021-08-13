import numpy as np
#import targets as tg


def Ncheb(Ek):
    '''
    Returns the truncation level N for the Chebyshev expansion according to the Ekman number
    Please experiment and adapt to your particular problem.
    '''
    x=-np.log10(Ek)
    if x>=9 :
        out = 380*x-2700
    else:
        out = 104*x-216
    out1 = max(50, int(0.7*out))
    return out1 + out1%2



# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
# ----------------------------------------------------------------------------------------------------------------------

# Azimuthal wave number m (>=0)
m = 1

# Equatorial symmetry. Use 1 for symmetric, -1 for antisymmetric. 
symm = -1

# Inner core radius, CMB radius is unity. 
ricb = 0

# Inner core spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
# Ignored if ricb = 0
bci = 1

# CMB spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
bco = 1

# Ekman number (use 2* to match Dintrans 1999)
Ek = 10**-6

forcing = 0  # For eigenvalue problems
# forcing = 1  # For Lin & Ogilvie 2018 tidal body force, m=2, symm. OK
# forcing = 2  # For boundary flow forcing, use with bci=1 and bco=1.
# forcing = 3  # For Rovira-Navarro 2018 tidal body forcing, m=0,2 must be symm, m=1 antisymm. Leaks power!
# forcing = 4  # first test case, Lin body forcing with X(r)=(A*r^2 + B/r^3)*C, (using Jeremy's calculation), m=2,symm. OK
# forcing = 5  # second test case, Lin body forcing with X(r)=1/r, m=0,symm. OK
# forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
# forcing = 7  # Longitudinal libration boundary forcing, m=0, symm, no-slip

# Forcing frequency and amplitude (ignored if forcing = 0)
freq0 = 1
delta = 0
forcing_frequency = freq0 + delta  # negative is prograde
forcing_amplitude = 1.0 

# if solving an eigenvalue problem, compute projection of eigenmode
# and some hypothetical forcing. Cases as described above (use only 1,3 or 4)
projection = 1



# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------- Magnetic field parameters
# ----------------------------------------------------------------------------------------------------------------------

magnetic = 0  # Use 0 for pure hydro, 1 for MHD. ** Needs ricb > 0 ** 

# Imposed background magnetic field
B0 = 'axial'  # Axial, uniform field along the spin axis
# B0 = 'dipole'  # Singular at origin

# Magnetic boundary conditions on the ICB:
innercore = 'insulator'
# innercore = 'perfect conductor, material'  # tangential *material* electric field jump [nxE']=0 across the ICB
# innercore = 'perfect conductor, spatial'   # tangential *spatial* electric field jump [nxE]=0 across the ICB
# Note: 'material' or 'spatial' are identical if ICB is no-slip (bci = 1 above)

# Magnetic field strength and magnetic diffusivity:
# Either use the Elsasser number and the magnetic Prandtl number (uncomment and set the following three lines):
# Lambda = 0.01
# Pm = 10**-4
# Em = Ek/Pm; Le2 = Lambda*Em; Le = np.sqrt(Le2)
# Or use the Lehnert number and the magnetic Ekman number (uncomment and set the following three lines):
Le = 0.01
Em = 1e-3
Le2 = Le**2



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------- Thermal parameters
# ----------------------------------------------------------------------------------------------------------------------

thermal = 1  # Use 1 or 0 to include or not the temperature equation and the buoyancy force (Boussinesq)

# Prandtl number: ratio of viscous to thermal diffusivity
Prandtl = 1.0

# Background isentropic temperature gradient choices, uncomment the appropriate line below:
# heating = 'internal'      # dT/dr = beta * (r/rcmb),     temp scale = rcmb*beta, Dintrans1999
# heating = 'differential'  # dT/dr = beta * (r/rcmb)**-2, temp scale = Delta T,   Dormy2004, set Ra below
# heating = 'two zone'      # temp scale = Omega^2*rcmb/(alpha*g_0), Vidal2015, use extra args below
heating = 'user defined'  # Uses the function BVprof in utils.py , use extra args below if needed 

# Ratio of Brunt-Vaisala freq. to rotation. If differential heating then set the Rayleigh number, otherwise just Brunt.
# Ra = 10**6  # Rayleigh number
# Brunt = np.sqrt(Ra/Prandtl) * Ek
Brunt = 2.0

# Additional arguments for 'Two zone' or 'User defined' case (modify if needed).
rc  = 100  # transition radius
h   = 0.1  # transition width
sym = -1    # radial symmetry 
args = [rc, h, sym]  

# Thermal boundary conditions
# 0 for isothermal, theta=0
# 1 for constant heat flux, (d/dr)theta=0
bci_thermal = 0   # ICB
bco_thermal = 0   # CMB



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------- Resolution
# ----------------------------------------------------------------------------------------------------------------------

# Number of cpus
ncpus = 24

# Chebyshev polynomial truncation level. Must be even if ricb = 0. See def at top.
# N = Ncheb(Ek)
N = 250

# Spherical harmonic truncation lmax and approx lmax/N ratio:
g = 2.0
lmax = int( 2*ncpus*( np.floor_divide( g*N, 2*ncpus ) ) + m - 1 )
# If manually setting the max angular degree lmax, then it must be even if m is odd,
# and lmax-m+1 should be divisible by 2*ncpus 
# lmax = 8



# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------- SLEPc solver options
# ----------------------------------------------------------------------------------------------------------------------

# rnd1 = 0
# Set track_target = 1 below to track an eigenvalue, 0 otherwise.
# Assumes a preexisting 'track_target' file with target data
# Set track_target = 2 to write initial 'track_target' file, see also solve.py
track_target = 0
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

which_eigenpairs = 'TM'  # Use 'TM' for shift-and-invert
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



# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------- Writes solution vector to disk if = 1
# ----------------------------------------------------------------------------------------------------------------------
write_solution = 1

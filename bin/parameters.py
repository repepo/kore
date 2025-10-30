import numpy as np
import sys
#import targets as tg


def Ncheb(Ek):
    '''
    Returns the truncation level N for the Chebyshev expansion according to the Ekman number
    Please *experiment and adapt* to your particular problem. N must be even.
    '''
    if Ek !=0 :
        out = int(17*Ek**-0.2)
    else:
        out = 48  #

    return max(48, out + out%2)


aux1 = 1.0  # Auxiliary variable, useful e.g. for ramps
aux2 = 0

# ---------------
magnetic      = 0
thermal       = 1
compositional = 0
# ---------------

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------- Reference state parameters
# ----------------------------------------------------------------------------------------------------------------------

background = 0 # For MESA file format profiles (load MESA or GYRE type model below)
#background = 1 # For implicitly defined profiles (adjust ρ and when required g, (T or p), (N^2 or dS), v and κ in radial_profiles.py)
#background = 2 # For explicitly defined profiles (supply r, ρ and when required g, (T or p), (N^2 or dS), v and κ in array format)

def_pressure = 0 # set to 1 if the background pressure is provided, set to 0 if background temperature is provided
def_entropy = 0 # set to 1 if the background entropy gradient is provided, set to 0 if the squared Brunt-Väisälä frequency is provided
def_viscosity = 0 # set to 1 if using a MESA profile and defining viscosity implicitly, set to 2 if using a MESA profile and defining viscosity explicitly
def_thermal_diffusivity = 1 # set to 1 if using a MESA profile and defining thermal diffusivity implicitly, set to 2 if using a MESA profile and defining thermal diffusivity explicitly

model = 'planet_profile.mesa' # MESA, GYRE, or astropy table model file

# model_type = 'poly'
model_type = 'mesa'
# model_type = 'gsm'
# model_type = 'astropy table'

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
# ----------------------------------------------------------------------------------------------------------------------
hydro = 1  # set to 1 to include the Navier-Stokes equation for the flow velocity, set to 0 otherwise

# Azimuthal wave number m (>=0)
m = 0

# Equatorial symmetry of the flow field. Use 1 for symmetric, -1 for antisymmetric.
symm = 1

# Inner core radius, surface/CMB radius is unity.
ricb = 0

# Inner core spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
bci = 0

# CMB spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
bco = 0

#-----------------
bco_thermal = 1
#-----------------

forcing = 0  # Uncomment this line for eigenvalue problems
# forcing = 1  # For Lin & Ogilvie 2018 tidal body force, m=2, symm. OK
# forcing = 2  # For boundary flow forcing, use with bci=1 and bco=1.
# forcing = 3  # For Rovira-Navarro 2018 tidal body forcing, m=0,2 must be symm, m=1 antisymm. Leaks power!
# forcing = 4  # first test case, Lin body forcing with X(r)=(A*r^2 + B/r^3)*C, (using Jeremy's calculation), m=2,symm. OK
# forcing = 5  # second test case, Lin body forcing with X(r)=1/r, m=0,symm. OK
# forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
# forcing = 7  # Longitudinal libration boundary forcing, m={0, 2}, symm, no-slip
# forcing = 8  # Longitudinal libration as a Poincaré force (body force) in the mantle frame, m=0, symm, no-slip
# forcing = 9  # Radial, symmetric, m=2 boundary flow forcing.

# Forcing frequency (ignored if forcing == 0)
forcing_frequency = 1.0  # negative is prograde

# Forcing amplitude. Body forcing amplitude will use the cmb value
forcing_amplitude_cmb = 1.0
forcing_amplitude_icb = 0.0

# if solving an eigenvalue problem, compute projection of eigenmode
# and some hypothetical forcing. Cases as described above (available only for 1,3 or 4)
projection = 1


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------- Unit of time and force switches
# ----------------------------------------------------------------------------------------------------------------------
# Choose the time scale by specifying the dimensionless angular velocity using the desired time scale. Please see
# the non-dimensionalization notes in the documentation. Uncomment your choice:
# OmgTau = 1     # Rotation time scale
# OmgTau = 1/Ek  # Viscous diffusion time scale
# OmgTau = 1/Le  # Alfvén time scale
# OmgTau = 1/Em  # Magnetic diffusion time scale

Gaspard = 1e-6  # Omega*Tau                   Coriolis force factor. Set to 1 for unit time Tau = 1/Omega
Beyonce = 1  # (N0*Tau)**2                 Buoyancy force factor. Set to 1 for unit time Tau = 1/N0 = sqrt(r0/g0)
Hendrik = 0  # (Tau*B0/r0)**2/(rho0*mu0)   Lorentz force factor. Set to 1 for Alfven time scale
ViscosD = 0  # nu0 * Tau / r0**2           Viscous force factor. Set to 1 for viscous diffusion time scale. This is the Ekman number if Tau = 1/Omega
ThermaD = 0  # kappa0 * Tau / r0**2        Thermal diffusion factor. Set to 1 for thermal diffusion time scale
MagnetD = 0  # eta0 * Tau / r0**2          Magnetic diffusion factor. Set to 1 for magnetic diffusion time scale


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------- Resolution
# ----------------------------------------------------------------------------------------------------------------------

# Number of cpus
ncpus = 24

# Chebyshev polynomial truncation level. Use function def at top or set manually. N must be even if ricb = 0.
# N = Ncheb(Ek)
N = 48*14

# Spherical harmonic truncation lmax and approx lmax/N ratio:
g = 1.0
lmax = int( 2*ncpus*( np.floor_divide( g*N, 2*ncpus ) ) + m - 1 )
# If manually setting the max angular degree lmax, then it must be even if m is odd,
# and lmax-m+1 should be divisible by 2*ncpus
# lmax = (2*ncpus*1 + m - 1)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------- SLEPc solver options
# ----------------------------------------------------------------------------------------------------------------------

# rnd1 = 0
# Set track_target = 1 below to track an eigenvalue, 0 otherwise.
# Assumes a preexisting 'track_target' file with target data
# Set track_target = 2 to write initial 'track_target' file, see also postprocess.py
track_target = 0
if track_target == 1 :  # read target from file and sets target accordingly
    tt = np.loadtxt('track_target')
    rtau = tt[0]
    itau = tt[1]
else:                   # set target manually
    rtau = 0
    itau = 0.36505

# tau is the actual target for the solver
# real part is damping
# imaginary part is frequency (positive is retrograde)
tau = rtau + itau*1j

which_eigenpairs = 'TM'  # Use 'TM' for shift-and-invert
# L/S/T & M/R/I
# L largest, S smallest, T target
# M magnitude, R real, I imaginary

# Number of desired eigenvalues
nev = 16

# Number of vectors in Krylov space for solver
# ncv = 100

# Maximum iterations to converge to an eigenvector
maxit = 50

# Tolerance for solver
tol = 1e-15

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

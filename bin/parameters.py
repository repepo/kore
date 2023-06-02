import numpy as np
#import targets as tg


def Ncheb(Ek):
    '''
    Returns the truncation level N for the Chebyshev expansion according to the Ekman number
    Please experiment and adapt to your particular problem. N must be even.
    '''
    if Ek !=0 :
        out = int(17*Ek**-0.2)
    else:
        out = 48  #

    return max(48, out + out%2)


aux = 1.0  # Auxiliary variable, useful e.g. for ramps



# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
# ----------------------------------------------------------------------------------------------------------------------
hydro = 1  # set to 1 to include the Navier-Stokes equation for the flow velocity, set to 0 otherwise

# Azimuthal wave number m (>=0)
m = 1

# Equatorial symmetry of the flow field. Use 1 for symmetric, -1 for antisymmetric.
symm = -1

# Inner core radius, CMB radius is unity.
ricb = 0.35

# Inner core spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
bci = 1

# CMB spherical boundary conditions
# Use 0 for stress-free, 1 for no-slip or forced boundary flow
bco = 1

# Ekman number (use 2* to match Dintrans 1999). Ek can be set to 0 if ricb=0
# CoriolisNumber = 1.2e3
# Ek_gap = 2/CoriolisNumber
# Ek = Ek_gap*(1-ricb)**2
Ek = 10**-4

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
# -------------------------------------------------------------------------------------------- Magnetic field parameters
# ----------------------------------------------------------------------------------------------------------------------
magnetic = 0  # set to 1 if including the induction equation and the Lorentz force

# Imposed background magnetic field
B0 = 'axial'          # Axial, uniform field along the spin axis
# B0 = 'dipole'         # classic dipole, singular at origin, needs ricb>0
# B0 = 'G21 dipole'     # Felix's dipole (Gerick GJI 2021)
# B0 = 'Luo_S1'         # Same as above, actually (Luo & Jackson PRSA 2022)
# B0 = 'Luo_S2'         # Quadrupole
# B0 = 'FDM'            # Free Poloidal Decay Mode (Zhang & Fearn 1994,1995; Schmitt 2012)
beta = 3.0              # guess for FDM's beta
B0_l = 1                # l number for the FDM mode

# Magnetic boundary conditions at the ICB:
innercore = 'insulator'
# innercore = 'TWA'  # Thin conductive wall layer (Roberts, Glatzmaier & Clune, 2010)
c_icb     = 0  # Ratio (h*mu_wall)/(ricb*mu_fluid) (if innercore='TWA')
c1_icb    = 0  # Thin wall to fluid conductance ratio (if innercore='TWA')
# innercore = 'perfect conductor, material'  # tangential *material* electric field jump [nxE']=0 across the ICB
# innercore = 'perfect conductor, spatial'   # tangential *spatial* electric field jump [nxE]=0 across the ICB
# Note: 'perfect conductor, material' or 'perfect conductor, spatial' are identical if ICB is no-slip (bci = 1 above)

# Magnetic boundary conditions at the CMB
mantle   = 'insulator'
# mantle = 'TWA'  # Thin conductive wall layer (Roberts, Glatzmaier & Clune, 2010)
c_cmb  = 0  # Ratio (h*mu_wall)/(rcmb*mu_fluid)  (if mantle='TWA')
c1_cmb = 0  # Thin wall to fluid conductance ratio (if mantle='TWA')

# Relative permeability (fluid/vacuum)
mu = 1.0

# Magnetic field strength and magnetic diffusivity:
# Either use the Elsasser number and the magnetic Prandtl number (i.e. Lambda and Pm: uncomment and set the following three lines):
# Lambda = 0.1
# Pm = 0.001
# Em = Ek/Pm; Le2 = Lambda*Em; Le = np.sqrt(Le2)
# Or use the Lehnert number and the magnetic Ekman number (i.e. Le and Em: uncomment and set the following three lines):
Le = 10**-3; Lu=2e3
Em = Le/Lu
Le2 = Le**2

# Normalization of the background magnetic field
cnorm = 'rms_cmb'                     # Sets the radial rms field at the CMB as unity
# cnorm = 'mag_energy'                  # Unit magnetic energy as in Luo & Jackson 2022 (I. Torsional oscillations)
# cnorm = 'Schmitt2012'                 # as above but times 2
# cnorm = 3.86375                       # G101 of Schmitt 2012, ricb = 0.35
# cnorm = 4.067144                      # Zhang & Fearn 1994,   ricb = 0.35
# cnorm = 15*np.sqrt(21/(46*np.pi))     # G21 dipole,           ricb = 0
# cnorm = 1.09436                       # simplest FDM, l=1,    ricb = 0
# cnorm = 3.43802                       # simplest FDM, l=1,    ricb = 0.001
# cnorm = 0.09530048175738767           # Luo_S1 ricb = 0, unit mag_energy
# cnorm = 0.6972166887783963            # Luo_S1 ricb = 0, rms_Bs=1
# cnorm = 0.005061566801979833          # Luo_S2 ricb = 0, unit mag_energy
# cnorm = 0.0158567582314039            # Luo_S2 ricb = 0, rms_Bs=1



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------- Thermal parameters
# ----------------------------------------------------------------------------------------------------------------------
thermal = 0  # Use 1 or 0 to include or not the temperature equation and the buoyancy force (Boussinesq)

# Prandtl number: ratio of viscous to thermal diffusivity
Prandtl = 1.0
# "Thermal" Ekman number
Etherm = Ek/Prandtl

# Background isentropic temperature gradient dT/dr choices, uncomment the appropriate line below:
heating = 'internal'      # dT/dr = -beta * r         temp_scale = beta * ro**2
# heating = 'differential'  # dT/dr = -beta * r**-2     temp_scale = Ti-To      beta = (Ti-To)*ri*ro/(ro-ri)
# heating = 'two zone'      # dT/dr = K * ut.twozone()  temp_scale = -ro * K
# heating = 'user defined'  # dT/dr = K * ut.BVprof()   temp_scale = -ro * K

# Rayleigh number as Ra = alpha * g0 * ro^3 * temp_scale / (nu*kappa), alpha is the thermal expansion coeff,
# g0 the gravity accel at ro, ro is the cmb radius (the length scale), nu is viscosity, kappa is thermal diffusivity.
Ra = 0.0
# Ra_Silva = 0.0; Ra = Ra_Silva * (1/(1-ricb))**6
# Ra_Monville = 0.0; Ra = 2*Ra_Monville

# Alternatively, you can specify directly the squared ratio of a reference Brunt-Väisälä freq. and the rotation rate.
# The reference Brunt-Väisälä freq. squared is defined as -alpha*g0*temp_scale/ro. See the non-dimensionalization notes
# in the documentation.
# BV2 = -Ra * Ek**2 / Prandtl
BV2 = 0.0

# Additional arguments for 'Two zone' or 'User defined' case (modify if needed).
rc  = 0.7  # transition radius
h   = 0.1  # transition width
sym = -1    # radial symmetry
args = [rc, h, sym]

# Thermal boundary conditions
# 0 for isothermal, theta=0
# 1 for constant heat flux, (d/dr)theta=0
bci_thermal = 1   # ICB
bco_thermal = 1   # CMB



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------- Compositional parameters
# ----------------------------------------------------------------------------------------------------------------------
compositional = 0  # Use 1 or 0 to include compositional transport or not (Boussinesq)

# Schmidt number: ratio of viscous to compositional diffusivity
Schmidt = 1.0
# "Compositional" Ekman number
Ecomp = Ek/Schmidt 

# Background isentropic composition gradient dC/dr choices, uncomment the appropriate line below:
comp_background = 'internal'      # dC/dr = -beta * r         comp_scale = beta * ro**2
# comp_background = 'differential'  # dC/dr = -beta * r**-2     comp_scale = Ci-Co

# Compositional Rayleigh number
Ra_comp = 0.0
# Ra_comp_Silva = 0.0; Ra_comp = Ra_comp_Silva * (1/(1-ricb))**6
# Ra_comp_Monville = 0.0; Ra_comp = 2*Ra_comp_Monville

# Alternatively, specify directly the squared ratio of a reference compositional Brunt-Väisälä frequency
# and the rotation rate.
# BV2_comp = -Ra_comp * Ek**2 / Schmidt
BV2_comp = 0.0

# Additional arguments for 'Two zone' or 'User defined' case (modify if needed).
rc  = 0.7  # transition radius
h   = 0.1  # transition width
sym = -1    # radial symmetry
args_comp = [rc, h, sym]

# Compositional boundary conditions
# 0 for constant composition, xi=0
# 1 for constant flux, (d/dr)xi=0
bci_compositional = 1   # ICB
bco_compositional = 1   # CMB



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------- Time scale
# ----------------------------------------------------------------------------------------------------------------------
# Choose the time scale by specifying the dimensionless angular velocity using the desired time scale. Please see
# the non-dimensionalization notes in the documentation. Uncomment your choice:
OmgTau = 1     # Rotation time scale
# OmgTau = 1/Ek  # Viscous diffusion time scale
# OmgTau = 1/Le  # Alfvén time scale
# OmgTau = 1/Em  # Magnetic diffusion time scale



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------- Resolution
# ----------------------------------------------------------------------------------------------------------------------

# Number of cpus
ncpus = 4

# Chebyshev polynomial truncation level. Use function def at top or set manually. N must be even if ricb = 0.
N = Ncheb(Ek)
# N = 24

# Spherical harmonic truncation lmax and approx lmax/N ratio:
g = 1.0
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
# Set track_target = 2 to write initial 'track_target' file, see also postprocess.py
track_target = 0
if track_target == 1 :  # read target from file and sets target accordingly
    tt = np.loadtxt('track_target')
    rtau = tt[0]
    itau = tt[1]
else:                   # set target manually
    rtau = 0.0
    itau = 1.0

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
tol = 1e-15
# Tolerance for the thermal/compositional matrix
tol_tc = 1e-6

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

#import targets as tg


# def Ncheb(Ek):
#     '''
#     Returns the truncation level N for the Chebyshev expansion according to the Ekman number
#     Please *experiment and adapt* to your particular problem. N must be even.
#     '''
#     if Ek !=0 :
#         out = int(17*Ek**-0.2)
#     else:
#         out = 48  #

#     return max(48, out + out%2)

class default_params():

    def __init__(self):


        self.aux1 = 1.0  # Auxiliary variable, useful e.g. for ramps
        self.aux2 = 0

        # ---------------
        self.hydro         = 1  # set to 1 to include Navier-Stokes equation
        self.magnetic      = 0  # set to 1 to include induction equation
        self.thermal       = 0  # set to 1 to include thermal equation
        self.compositional = 0  # set to 1 to include compositional equation
        # ---------------

        ##### Control parameters #####
        self.Ek = 1e-4

        self.Ra = 0.0
        self.Prandtl = 1.0

        self.Le = 0.0
        self.Pm = 1.0
        self.Em = self.Ek/self.Pm
        self.Els = self.Le**2/self.Em

        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------- Reference state parameters
        # ----------------------------------------------------------------------------------------------------------------------

        self.background = 'bouss' # 'bouss' for Boussinesq, else anelastic reference state
        # self.background = 'model' # For MESA file format profiles (load MESA or GYRE type model below)
        # self.background = 'userdef' # For implicitly defined profiles (adjust ρ and when required g, (T or p), (N^2 or dS), in radial_profiles.py)
        # self.background = 'array' # For explicitly defined profiles (supply r, ρ and when required g, (T or p), (N^2 or dS), in array format)

        self.def_pressure = 0 # set to 1 if the background pressure is provided, set to 0 if background temperature is provided
        self.def_entropy = 0 # set to 1 if the background entropy gradient is provided, set to 0 if the squared Brunt-Väisälä frequency is provided
        self.def_viscosity = 0 # set to 1 if defining viscosity implicitly, set to 2 if defining viscosity explicitly
        self.def_thermal_diffusivity = 1 # set to 1 if using a MESA profile and defining thermal diffusivity implicitly, set to 2 if using a MESA profile and defining thermal diffusivity explicitly

        self.model = 'planet_profile.mesa' # MESA, GYRE, or astropy table model file

        # self.model_type = 'poly'
        self.model_type = 'mesa'
        # self.model_type = 'gsm'
        # self.model_type = 'astropy table'

        # ----------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
        # ----------------------------------------------------------------------------------------------------------------------

        # Azimuthal wave number m (>=0)
        self.m = 1

        # Equatorial symmetry of the flow field. Use 1 for symmetric, -1 for antisymmetric.
        self.symm = -1

        # Inner core radius, surface/CMB radius is unity.
        self.ricb = 0

        # Inner core mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
        self.bci = 1

        # CMB mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow
        self.bco = 1

        if self.background == 'bouss':
            self.heating = 'differential'  # 'internal' or 'differential' heating

        #----------------- Thermal boundary conditions: 0 for Dirichlet, 1 for Neumann
        self.bci_thermal = 0
        self.bco_thermal = 0
        #-----------------

        self.forcing = 0  # Uncomment this line for eigenvalue problems
        # self.forcing = 1  # For Lin & Ogilvie 2018 tidal body force, m=2, symm. OK
        # self.forcing = 2  # For boundary flow forcing, use with bci=1 and bco=1.
        # self.forcing = 3  # For Rovira-Navarro 2018 tidal body forcing, m=0,2 must be symm, m=1 antisymm. Leaks power!
        # self.forcing = 4  # first test case, Lin body forcing with X(r)=(A*r^2 + B/r^3)*C, (using Jeremy's calculation), m=2,symm. OK
        # self.forcing = 5  # second test case, Lin body forcing with X(r)=1/r, m=0,symm. OK
        # self.forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
        # self.forcing = 7  # Longitudinal libration boundary forcing, m={0, 2}, symm, no-slip
        # self.forcing = 8  # Longitudinal libration as a Poincaré force (body force) in the mantle frame, m=0, symm, no-slip
        # self.forcing = 9  # Radial, symmetric, m=2 boundary flow forcing.

        # Forcing frequency (ignored if forcing == 0)
        self.forcing_frequency = 1.0  # negative is prograde

        # Forcing amplitude. Body forcing amplitude will use the cmb value
        self.forcing_amplitude_cmb = 1.0
        self.forcing_amplitude_icb = 0.0

        # if solving an eigenvalue problem, compute projection of eigenmode
        # and some hypothetical forcing. Cases as described above (available only for 1,3 or 4)
        self.projection = 1


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------- Unit of time and force switches
        # ----------------------------------------------------------------------------------------------------------------------
        self.timescale = "rotation" # "rotation" for 1/Omega, "buoyancy" for 1/N0 = sqrt(r0/g0), "alfven" for r0/Va, "viscous" for r0^2/nu0

        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------- Resolution
        # ----------------------------------------------------------------------------------------------------------------------

        # Number of cpus
        self.ncpus = 4

        # Chebyshev polynomial truncation level. Use function def at top or set manually. N must be even if ricb = 0.
        if self.Ek !=0 :
            out = int(17*self.Ek**-0.2)
        else:
            out = 48  #

        self.N = max(48, out + out%2)

        # Spherical harmonic truncation lmax and approx lmax/N ratio:
        self.g = 1.0
        self.lmax = int( 2*self.ncpus*( np.floor_divide( self.g*self.N, 2*self.ncpus ) ) + self.m - 1 )
        # If manually setting the max angular degree lmax, then it must be even if m is odd,
        # and lmax-m+1 should be divisible by 2*ncpus
        # self.lmax = (2*self.ncpus*1 + self.m - 1)


        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------- SLEPc solver options
        # ----------------------------------------------------------------------------------------------------------------------

        # rnd1 = 0
        # Set track_target = 1 below to track an eigenvalue, 0 otherwise.
        # Assumes a preexisting 'track_target' file with target data
        # Set track_target = 2 to write initial 'track_target' file, see also postprocess.py
        self.track_target = 0
        if self.track_target == 1 :  # read target from file and sets target accordingly
            tt = np.loadtxt('track_target')
            self.rtau = tt[0]
            self.itau = tt[1]
        else:                   # set target manually
            self.rtau = 0
            self.itau = 0.36505

        # tau is the actual target for the solver
        # real part is damping
        # imaginary part is frequency (positive is retrograde)
        # self.tau = rtau + itau*1j

        self.which_eigenpairs = 'TM'  # Use 'TM' for shift-and-invert
        # L/S/T & M/R/I
        # L largest, S smallest, T target
        # M magnitude, R real, I imaginary

        # Number of desired eigenvalues
        self.nev = 16

        # Number of vectors in Krylov space for solver
        # self.ncv = 100

        # Maximum iterations to converge to an eigenvector
        self.maxit = 50

        # Tolerance for solver
        self.tol = 1e-15

    def set_scales(self):
        '''
        Set dimensional scales based on chosen time scale
        '''

        if self.timescale == "rotation":
            self.Gaspard = 1  # Omega*Tau                     Coriolis force factor. Set to 1 for unit time Tau = 1/Omega
            self.Beyonce = -self.Ra * self.Ek/self.Prandtl  # (N0*Tau)**2    Buoyancy force factor. Set to 1 for unit time Tau = 1/N0 = sqrt(r0/g0)
            self.Hendrik = self.Le  # (Tau*B0/r0)**2/(rho0*mu0)    Lorentz force factor. Set to 1 for Alfven time scale
            self.ViscosD = self.Ek  # nu0 * Tau / r0**2            Viscous force factor. Set to 1 for viscous diffusion time scale. This is the Ekman number if Tau = 1/Omega
            self.ThermaD = self.Ek/self.Prandtl  # kappa0 * Tau / r0**2 Thermal diffusion factor. Set to 1 for thermal diffusion time scale
            self.MagnetD = self.Em  # eta0 * Tau / r0**2           Magnetic diffusion factor. Set to 1 for magnetic diffusion time scale
        elif self.timescale == "viscous":
            self.Gaspard = 1/self.Ek  # Omega*Tau
            self.Beyonce = -self.Ra/self.Prandtl  # (N0*Tau)**2
            self.Hendrik = 1/(self.Ek*self.Pm)  # (Tau*B0/r0)**2/(rho0*mu0)
            self.ViscosD = 1  # nu0 * Tau / r0**2
            self.ThermaD = 1/self.Prandtl  # kappa0 * Tau / r0**2
            self.MagnetD = 1/self.Pm  # eta0 * Tau / r0**2
        elif self.timescale == "alfven":
            self.Gaspard = 1/self.Le  # Omega*Tau
            self.Beyonce = 0  # (N0*Tau)**2
            self.Hendrik = 1  # (Tau*B0/r0)**2/(rho0*mu0)
            self.ViscosD = self.Ek/self.Le  # nu0 * Tau / r0**2
            self.ThermaD = self.Ek/(self.Le*self.Prandtl)  # kappa0 * Tau / r0**2
            self.MagnetD = self.Em/self.Le  # eta0 * Tau / r0**2
        # elif timescale == "buoyancy": # Not sure anyone uses this?
        #     Gaspard =   # Omega*Tau
        #     Beyonce =   # (N0*Tau)**2
        #     Hendrik = 0  # (Tau*B0/r0)**2/(rho0*mu0)
        #     ViscosD = 0  # nu0 * Tau / r0**2
        #     ThermaD = 0  # kappa0 * Tau / r0**2
        #     MagnetD = 0  # eta0 * Tau / r0**2
        else: #Custom factors
            self.Gaspard = 1
            self.Beyonce = 1
            self.Hendrik = 1
            self.ViscosD = 1
            self.ThermaD = 1
            self.MagnetD = 1
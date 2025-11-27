import numpy as np
#import targets as tg



class default_params():

    def __init__(self):

        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------ Physics modules       
        # ----------------------------------------------------------------------------------------------------------------------
        self.hydro         = 1  # set to 1 to include the Navier-Stokes equation
        self.magnetic      = 0  # set to 1 to include the induction equation
        self.thermal       = 0  # set to 1 to include the thermal equation
        self.compositional = 0  # set to 1 to include the compositional equation



        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------- Solution symmetry and inner core size
        # ----------------------------------------------------------------------------------------------------------------------
        self.m    = 0  # Azimuthal wave number
        self.symm = 1  # Equatorial symmetry, 1 for symmetric, -1 for antisymmetric
        self.ricb = 0  # Solid inner core radius 



        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------- Star/planet model parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.model = 'poly.n1_gamma3.h5' # MESA, GYRE, or astropy table model file

        self.model_type = 'poly'              # For polytropic profiles supplied by Gyre
        # self.model_type = 'mesa'              # For MESA file format profiles
        # self.model_type = 'gsm'               # For Gyre file format profiles
        # self.model_type = 'astropy table'     # For explicitly defined profiles (supply r, ρ and when required g, (T or p), (N^2 or dS), in array format)
        # self.model_type = 'Boussinesq'        #  for Boussinesq reference state, else anelastic reference state
        # self.model_type = 'user def'          # For implicitly defined profiles (adjust ρ and when required g, p, pdS in radial_profiles.py)

        # profile smoothing power
        self.smopo = 0

        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------- Auxiliary variables        
        # ----------------------------------------------------------------------------------------------------------------------
        self.Ek     = 0
        self.Em     = 0
        self.Etherm = 0
        self.Le     = 0 

        self.aux0   = 0
        self.aux1   = 0
        self.aux2   = 0
        self.aux3   = 0
        self.aux4   = 0
        self.aux5   = 0



        # ----------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
        # ----------------------------------------------------------------------------------------------------------------------
        # Inner core mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
        self.bci = 0

        # CMB mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow
        self.bco = 0



        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Thermal parameters
        # ----------------------------------------------------------------------------------------------------------------------
        if self.model_type == 'Boussinesq':
            self.heating = 'differential'  # 'internal' or 'differential' heating

        # Thermal boundary conditions: 0 for Dirichlet, 1 for Neumann
        self.bci_thermal = 0
        self.bco_thermal = 0



        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Forcing parameters
        # ----------------------------------------------------------------------------------------------------------------------
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
        self.timescale = 'free fall'
        self.Gaspard   = 1  # Omega*Tau                 Coriolis force factor. Set to 1 for unit time Tau = 1/Omega
        self.Beyonce   = 0  # (N0*Tau)**2               Buoyancy force factor. Set to 1 for unit time Tau = 1/N0 = sqrt(r0/g0)
        self.Hendrik   = 0  # (Tau*B0/r0)**2/(rho0*mu0) Lorentz force factor. Set to 1 for Alfven time scale
        self.ViscosD   = 0  # nu0 * Tau / r0**2         Viscous force factor. Set to 1 for viscous diffusion time scale. This is the Ekman number if Tau = 1/Omega
        self.ThermaD   = 0  # kappa0 * Tau / r0**2      Thermal diffusion factor. Set to 1 for thermal diffusion time scale
        self.MagnetD   = 0  # eta0 * Tau / r0**2        Magnetic diffusion factor. Set to 1 for magnetic diffusion time scale



        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------- Resolution
        # ----------------------------------------------------------------------------------------------------------------------
        # Number of cpus
        self.ncpus = 2

        # Chebyshev polynomial truncation level. Use function def at top or set manually. N must be even if ricb = 0.
        self.N = 120

        # Spherical harmonic truncation lmax and approx lmax/N ratio:
        self.g = 1.0
        self.lmax = int( 2*self.ncpus*( np.floor_divide( self.g*self.N, 2*self.ncpus ) ) + self.m - 1 )
        # If manually setting the max angular degree lmax, then it must be even if m is odd,
        # and lmax-m+1 should be divisible by 2*ncpus
        # self.lmax = (2*self.ncpus*1 + self.m - 1)



        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------- SLEPc solver options
        # ----------------------------------------------------------------------------------------------------------------------
        # Set track_target = 1 below to track an eigenvalue, 0 otherwise.
        # Assumes a preexisting 'track_target' file with target data
        # Set track_target = 2 to write initial 'track_target' file, see also solve.py
        self.track_target = 0
        if self.track_target == 1 :  # read target from file and sets target accordingly
            tt = np.loadtxt('track_target')
            self.rtau = tt[0]
            self.itau = tt[1]
        else:                        # set target manually
            self.rtau = 0.0
            self.itau = 1.0

        self.which_eigenpairs = 'TM'  # Use 'TM' for shift-and-invert
        # L/S/T & M/R/I
        # L largest, S smallest, T target
        # M magnitude, R real, I imaginary

        # Number of desired eigenvalues
        self.nev = 5

        # Number of vectors in Krylov space for solver
        # self.ncv = 100

        # Maximum iterations to converge to an eigenvector
        self.maxit = 50

        # Tolerance for solver
        self.tol = 1e-15



        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------



    def set_scales(self):
        '''
        Sets some basic dimensionless parameters based on a chosen time scale
        '''

        if self.timescale == "rotation":  # unit time Tau = 1/Omega

            self.Gaspard = 1            # Omega*Tau                 
            self.Beyonce = 0            # (N0*Tau)**2               
            self.Hendrik = self.Le      # (Tau*B0/r0)**2/(rho0*mu0) 
            self.ViscosD = self.Ek      # nu0 * Tau / r0**2         
            self.ThermaD = self.Etherm  # kappa0 * Tau / r0**2      
            self.MagnetD = self.Em      # eta0 * Tau / r0**2        

        elif self.timescale == "viscous":  # viscous diffusion time scale

            self.Gaspard = 1/self.Ek            # Omega*Tau
            self.Beyonce = 0                    # (N0*Tau)**2
            self.Hendrik = 1/self.Em            # (Tau*B0/r0)**2/(rho0*mu0)
            self.ViscosD = 1                    # nu0 * Tau / r0**2
            self.ThermaD = self.Etherm/self.Ek  # kappa0 * Tau / r0**2
            self.MagnetD = self.Em/self.Ek      # eta0 * Tau / r0**2
        
        elif self.timescale == "Alfven":  # Alfven time scale

            self.Gaspard = 1/self.Le            # Omega*Tau
            self.Beyonce = 0                    # (N0*Tau)**2
            self.Hendrik = 1                    # (Tau*B0/r0)**2/(rho0*mu0)
            self.ViscosD = self.Ek/self.Le      # nu0 * Tau / r0**2
            self.ThermaD = self.Etherm/self.Le  # kappa0 * Tau / r0**2
            self.MagnetD = self.Em/self.Le      # eta0 * Tau / r0**2
        
        elif self.timescale == "free fall":  # unit time Tau = 1/N0 = sqrt(r0/g0)

            self.Gaspard = 0  # Omega*Tau
            self.Beyonce = 1  # (N0*Tau)**2
            self.Hendrik = 0  # (Tau*B0/r0)**2/(rho0*mu0)
            self.ViscosD = 0  # nu0 * Tau / r0**2
            self.ThermaD = 0  # kappa0 * Tau / r0**2
            self.MagnetD = 0  # eta0 * Tau / r0**2



    def Ncheb(self, Ek):
        '''
        Returns the truncation level N for the Chebyshev expansion according to the Ekman number
        Please EXPERIMENT AND ADAPT to your particular problem. N must be even.
        '''
        if Ek !=0 :
            out = int(17*Ek**-0.2)
        else:
            out = 48  #

        return max(48, out + out%2)



    def ellmax(self, ncpus, g, m, N):
        '''
        Returns the lmax given a radial truncation level N,
        ncpus, azimuthal wave number m, and an approx. N/lmax ratio g.
        '''
        #out = int(2*ncpus + m - 1)
        out = int( 2*ncpus*( np.floor_divide( g*N, 2*ncpus ) ) + m - 1 )

        return out
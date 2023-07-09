import numpy as np
#import targets as tg


class default_params():

    def __init__(self) -> None:



    # def Ncheb(self):
    #     '''
    #     Returns the truncation level N for the Chebyshev expansion according to the Ekman number
    #     Please experiment and adapt to your particular problem. N must be even.
    #     '''
    #     if self.Ek !=0 :
    #         out = int(17*self.Ek**-0.2)
    #     else:
    #         out = 48  #

    #     self.Ncheb = max(48, out + out%2)


    # self.aux = 1.0  # Auxiliary variable, useful e.g. for ramps



    # ----------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
    # ----------------------------------------------------------------------------------------------------------------------
        self.hydro = 1  # set to 1 to include the Navier-Stokes equation for the flow velocity, set to 0 otherwise

        # Azimuthal wave number m (>=0)
        self.m = 1

        # Equatorial symmetry of the flow field. Use 1 for symmetric, -1 for antisymmetric.
        self.symm = -1

        # Inner core radius, CMB radius is unity.
        self.ricb = 0.35

        # Inner core spherical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
        self.bci = 1

        # CMB spherical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow
        self.bco = 1

        # Ekman number (use 2* to match Dintrans 1999). Ek can be set to 0 if ricb=0
        # CoriolisNumber = 1.2e3
        # Ek_gap = 2/CoriolisNumber
        # Ek = Ek_gap*(1-ricb)**2
        self.Ek = 1e-4

        self.anelastic = 0
        self.variable_viscosity = 0

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
        # -------------------------------------------------------------------------------------------- Magnetic field parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.magnetic = 0  # set to 1 if including the induction equation and the Lorentz force

        # Imposed background magnetic field
        self.B0 = 'axial'          # Axial, uniform field along the spin axis
        # self.B0 = 'dipole'         # classic dipole, singular at origin, needs ricb>0
        # self.B0 = 'G21 dipole'     # Felix's dipole (Gerick GJI 2021)
        # self.B0 = 'Luo_S1'         # Same as above, actually (Luo & Jackson PRSA 2022)
        # self.B0 = 'Luo_S2'         # Quadrupole
        # self.B0 = 'FDM'            # Free Poloidal Decay Mode (Zhang & Fearn 1994,1995; Schmitt 2012)
        self.beta = 3.0              # guess for FDM's beta
        self.B0_l = 1                # l number for the FDM mode

        # Magnetic boundary conditions at the ICB:
        self.innercore = 'insulator'
        # innercore = 'TWA'  # Thin conductive wall layer (Roberts, Glatzmaier & Clune, 2010)
        self.c_icb     = 0  # Ratio (h*mu_wall)/(ricb*mu_fluid) (if innercore='TWA')
        self.c1_icb    = 0  # Thin wall to fluid conductance ratio (if innercore='TWA')
        # innercore = 'perfect conductor, material'  # tangential *material* electric field jump [nxE']=0 across the ICB
        # innercore = 'perfect conductor, spatial'   # tangential *spatial* electric field jump [nxE]=0 across the ICB
        # Note: 'perfect conductor, material' or 'perfect conductor, spatial' are identical if ICB is no-slip (bci = 1 above)

        # Magnetic boundary conditions at the CMB
        self.mantle   = 'insulator'
        # mantle = 'TWA'  # Thin conductive wall layer (Roberts, Glatzmaier & Clune, 2010)
        self.c_cmb  = 0  # Ratio (h*mu_wall)/(rcmb*mu_fluid)  (if mantle='TWA')
        self.c1_cmb = 0  # Thin wall to fluid conductance ratio (if mantle='TWA')

        # Relative permeability (fluid/vacuum)
        self.mu = 1.0

        # Magnetic field strength and magnetic diffusivity:
        # Either use the Elsasser number and the magnetic Prandtl number (i.e. Lambda and Pm: uncomment and set the following three lines):
        # Lambda = 0.1
        # Pm = 0.001
        # Em = Ek/Pm; Le2 = Lambda*Em; Le = np.sqrt(Le2)
        # Or use the Lehnert number and the magnetic Ekman number (i.e. Le and Em: uncomment and set the following three lines):
        self.Le = 10**-3; self.Lu=2e3
        self.Em = self.Le/self.Lu
        self.Le2 = self.Le**2

        # Normalization of the background magnetic field
        self.cnorm = 'rms_cmb'                     # Sets the radial rms field at the CMB as unity
        # self.cnorm = 'mag_energy'                  # Unit magnetic energy as in Luo & Jackson 2022 (I. Torsional oscillations)
        # self.cnorm = 'Schmitt2012'                 # as above but times 2
        # self.cnorm = 3.86375                       # G101 of Schmitt 2012, ricb = 0.35
        # self.cnorm = 4.067144                      # Zhang & Fearn 1994,   ricb = 0.35
        # self.cnorm = 15*np.sqrt(21/(46*np.pi))     # G21 dipole,           ricb = 0
        # self.cnorm = 1.09436                       # simplest FDM, l=1,    ricb = 0
        # self.cnorm = 3.43802                       # simplest FDM, l=1,    ricb = 0.001
        # self.cnorm = 0.09530048175738767           # Luo_S1 ricb = 0, unit mag_energy
        # self.cnorm = 0.6972166887783963            # Luo_S1 ricb = 0, rms_Bs=1
        # self.cnorm = 0.005061566801979833          # Luo_S2 ricb = 0, unit mag_energy
        # self.cnorm = 0.0158567582314039            # Luo_S2 ricb = 0, rms_Bs=1



        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Thermal parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.thermal = 0  # Use 1 or 0 to include or not the temperature equation and the buoyancy force (Boussinesq)

        # Prandtl number: ratio of viscous to thermal diffusivity
        self.Prandtl = 1.0
        # "Thermal" Ekman number
        self.Etherm = self.Ek/self.Prandtl

        # Background isentropic temperature gradient dT/dr choices, uncomment the appropriate line below:
        self.heating = 'internal'      # dT/dr = -beta * r         temp_scale = beta * ro**2
        # self.heating = 'differential'  # dT/dr = -beta * r**-2     temp_scale = Ti-To      beta = (Ti-To)*ri*ro/(ro-ri)
        # self.heating = 'two zone'      # dT/dr = K * ut.twozone()  temp_scale = -ro * K
        # self.heating = 'user defined'  # dT/dr = K * ut.BVprof()   temp_scale = -ro * K

        # Rayleigh number as Ra = alpha * g0 * ro^3 * temp_scale / (nu*kappa), alpha is the thermal expansion coeff,
        # g0 the gravity accel at ro, ro is the cmb radius (the length scale), nu is viscosity, kappa is thermal diffusivity.
        self.Ra = 0.0
        # Ra_Silva = 0.0; Ra = Ra_Silva * (1/(1-ricb))**6
        # Ra_Monville = 0.0; Ra = 2*Ra_Monville

        # Alternatively, you can specify directly the squared ratio of a reference Brunt-Väisälä freq. and the rotation rate.
        # The reference Brunt-Väisälä freq. squared is defined as -alpha*g0*temp_scale/ro. See the non-dimensionalization notes
        # in the documentation.
        # self.BV2 = -self.Ra * self.Ek**2 / self.Prandtl
        self.BV2 = 0.0

        self.ampStrat = 0
        self.rStrat   = 0.6
        self.thickStrat=0.1
        self.slopeStrat=75

        self.dent_args = [self.ampStrat,
                          self.rStrat,
                          self.thickStrat,
                          self.slopeStrat]

        # Additional arguments for 'Two zone' or 'User defined' case (modify if needed).
        self.rc  = 0.7  # transition radius
        self.h   = 0.1  # transition width
        self.sym = -1    # radial symmetry
        self.args = [self.rc,
                     self.h,
                     self.sym]

        # Thermal boundary conditions
        # 0 for isothermal, theta=0
        # 1 for constant heat flux, (d/dr)theta=0
        self.bci_thermal = 1   # ICB
        self.bco_thermal = 1   # CMB



        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------- Compositional parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.compositional = 0  # Use 1 or 0 to include compositional transport or not (Boussinesq)

        # Schmidt number: ratio of viscous to compositional diffusivity
        self.Schmidt = 1.0
        # "Compositional" Ekman number
        self.Ecomp = self.Ek/self.Schmidt

        # Background isentropic composition gradient dC/dr choices, uncomment the appropriate line below:
        self.comp_background = 'internal'      # dC/dr = -beta * r         comp_scale = beta * ro**2
        # comp_background = 'differential'  # dC/dr = -beta * r**-2     comp_scale = Ci-Co

        # Compositional Rayleigh number
        self.Ra_comp = 0.0
        # Ra_comp_Silva = 0.0; Ra_comp = Ra_comp_Silva * (1/(1-ricb))**6
        # Ra_comp_Monville = 0.0; Ra_comp = 2*Ra_comp_Monville

        # Alternatively, specify directly the squared ratio of a reference compositional Brunt-Väisälä frequency
        # and the rotation rate.
        # BV2_comp = -self.Ra_comp * self.Ek**2 / self.Schmidt
        self.BV2_comp = 0.0

        # Additional arguments for 'Two zone' or 'User defined' case (modify if needed).
        self.rc  = 0.7  # transition radius
        self.h   = 0.1  # transition width
        self.sym = -1    # radial symmetry
        self.args_comp = [self.rc,
                          self.h,
                          self.sym]

        # Compositional boundary conditions
        # 0 for constant composition, xi=0
        # 1 for constant flux, (d/dr)xi=0
        self.bci_compositional = 1   # ICB
        self.bco_compositional = 1   # CMB



        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------- Time scale
        # ----------------------------------------------------------------------------------------------------------------------
        # Choose the time scale by specifying the dimensionless angular velocity using the desired time scale. Please see
        # the non-dimensionalization notes in the documentation. Uncomment your choice:
        self.OmgTau = 1     # Rotation time scale
        # OmgTau = 1/Ek  # Viscous diffusion time scale
        # OmgTau = 1/Le  # Alfvén time scale
        # OmgTau = 1/Em  # Magnetic diffusion time scale



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

        # N = 24

        # Spherical harmonic truncation lmax and approx lmax/N ratio:
        g = 1.0
        self.lmax = int( 2*self.ncpus*( np.floor_divide( g*self.N, 2*self.ncpus ) ) + self.m - 1 )
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
        self.track_target = 0
        if self.track_target == 1 :  # read target from file and sets target accordingly
            tt = np.loadtxt('track_target')
            rtau = tt[0]
            itau = tt[1]
        else:                   # set target manually
            rtau = 0.0
            itau = 1.0

        # tau is the actual target for the solver
        # real part is damping
        # imaginary part is frequency (positive is retrograde)
        self.tau = rtau + itau*1j

        self.which_eigenpairs = 'TM'  # Use 'TM' for shift-and-invert
        # L/S/T & M/R/I
        # L largest, S smallest, T target
        # M magnitude, R real, I imaginary

        # Number of desired eigenvalues
        self.nev = 3

        # Number of vectors in Krylov space for solver
        # ncv = 100

        # Maximum iterations to converge to an eigenvector
        self.maxit = 50

        # Tolerance for solver
        self.tol = 1e-15
        # Tolerance for the thermal/compositional matrix
        self.tol_tc = 1e-6

        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
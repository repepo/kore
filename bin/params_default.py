import numpy as np



def pmak(Pm0, E0, E):
   Pm1 = 1e-6
   E1 = 1e-15
   alpha = np.log(Pm0/Pm1) / np.log(E0/E1)
   k = Pm1/(E1**alpha)
   return k*(E**alpha)

def ssak(ss0, E0, E):
   ss1 = 10
   E1 = 1e-15
   alpha = np.log(ss0/ss1) / np.log(E0/E1)
   k = ss1/(E1**alpha)
   return k*(E**alpha)



class default_params():
    '''
    Default values of all the parameters. parameters.py creates an instance, changes only what it needs, and then
    calls set_scales() to compute the derived parameters (those that are None here).
    '''

    def __init__(self):

        self.aux1 = 1.0  # Auxiliary variables, useful e.g. for ramps
        self.aux2 = 0.0



        # ----------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.hydro = 1  # set to 1 to include the Navier-Stokes equation for the flow velocity, set to 0 otherwise

        # Azimuthal wave number m (>=0)
        self.m = 0

        # Equatorial symmetry of the flow field. Use 1 for symmetric, -1 for antisymmetric.
        self.symm = 1

        # Inner core radius, CMB radius is unity.
        self.ricb = 0.35

        # Inner core spherical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
        self.bci = 1

        # CMB spherical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow
        self.bco = 1

        # Rotation: 1 for a rotating frame, 0 for no rotation (the Coriolis force is then not assembled). Without
        # rotation the time scale cannot be 1/Omega: use timescale = 'viscous' (Ek then only labels the units and
        # can be any nonzero value, the results do not depend on it) or set OmgTau directly, and set N explicitly.
        self.rotation = 1

        # Ekman number (use 2* to match Dintrans 1999). Ek can be set to 0 if ricb=0
        # CoriolisNumber = 1.2e3
        # Ek_gap = 2/CoriolisNumber
        # Ek = Ek_gap*(1-ricb)**2
        self.Ek = 10**-6

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
        self.forcing_frequency = 0.0  # negative is prograde

        # Forcing amplitude. Body forcing amplitude will use the cmb value
        self.forcing_amplitude_cmb = 0.0
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

        # B0_ic sets what the background field does inside the inner core.
        # 'G21 dipole', 'Luo_S1' and 'Luo_S2' are defined for a full sphere. With an inner core (ricb > 0) they do not match
        # a potential field at the ICB, so part of the electric current that makes them would have to flow inside the IC
        # (or in a current sheet on the ICB). Nothing keeps such a current steady in a solid IC of finite conductivity: there
        # is no flow there, hence no EMF, and an axisymmetric azimuthal current decays on the IC's magnetic diffusion time.
        #   'potential'   (default): removes that part of the current. utils.B0_beta_ic adds
        #                 beta*r**-(l+1) to the poloidal scalar h, with beta = ricb**(l+1)*(ricb*h' - l*h)/(2l+1) at the ICB.
        #                 The added term carries no current and keeps the CMB matching, so the currents in the fluid stay the
        #                 same, and B0 continues into the IC as the potential field h(ricb)*(r/ricb)**l (no current sheet on
        #                 the ICB). This is the self-consistent steady B0 for an insulating IC and for any finite IC
        #                 conductivity. To keep exactly the fluid currents of the full-sphere field, use its numeric cnorm
        #                 below (e.g. 0.005061567359972097 for Luo_S2) rather than 'mag_energy'.
        #   'full sphere' : h unchanged, its currents extend into the IC. Self-consistent only for a perfectly conducting
        #                 IC, or as a snapshot of currents frozen in the IC.
        # No effect if ricb = 0, nor for 'axial' and 'FDM' (they already match a potential field at the ICB) and 'dipole'
        # (a point source at the centre).
        self.B0_ic = 'potential'
        # self.B0_ic = 'full sphere'

        # Magnetic boundary conditions at the ICB:
        self.innercore = 'insulator'
        # self.innercore = 'conducting, Chebys'  # For eigenvalue problems
        # self.innercore = 'conducting, Bessel'  # For forced problems
        # self.innercore = 'TWA'  # Thin conductive wall layer (Roberts, Glatzmaier & Clune, 2010)
        # self.innercore = 'perfect conductor, material'  # tangential *material* electric field jump [nxE']=0 across the ICB
        # self.innercore = 'perfect conductor, spatial'   # tangential *spatial* electric field jump [nxE]=0 across the ICB
        self.c_icb  = 0  # Ratio (h*mu_wall)/(ricb*mu_fluid) (if innercore='TWA')
        self.c1_icb = 0  # Thin wall to fluid conductance ratio (if innercore='TWA')
        # Note: 'perfect conductor, material' or 'perfect conductor, spatial' are identical if ICB is no-slip (bci = 1 above)

        # Magnetic boundary conditions at the CMB
        self.mantle = 'insulator'
        # self.mantle = 'TWA'  # Thin conductive wall layer (Roberts, Glatzmaier & Clune, 2010)
        self.c_cmb  = 1e-5  # Ratio (h*mu_wall)/(rcmb*mu_fluid)  (if mantle='TWA')
        self.c1_cmb = 1e-5  # Thin wall to fluid conductance ratio (if mantle='TWA')

        # B0_cmb sets how the background field meets a thin conducting wall at the CMB (mantle = 'TWA').
        # A steady B0 drives no current in the wall: its currents are azimuthal, and the azimuthal electric field on the CMB
        # is set by dB_r/dt, which is zero. So c1_cmb (the wall's eddy currents) does not act on B0, but c_cmb does: B0 has
        # to satisfy (1/mu + l*c_cmb)*(r*h)' + l*h = 0 at the CMB, the thin-wall condition on b with c1_cmb = 0.
        #   'wall'  (default): utils.B0_alpha_cmb adds alpha*r**l to the poloidal scalar h,
        #           with alpha set by that condition. The added term carries no current and keeps the ICB matching of
        #           B0_ic, so the currents in the fluid stay the same. No effect if c_cmb = 0 and mu = 1, where every field
        #           already matches, nor for an insulating mantle or 'axial' (imposed from outside the core).
        #   'plain' : h unchanged. B0 then meets the wall condition only for c_cmb = 0 (and mu = 1).
        self.B0_cmb = 'wall'
        # self.B0_cmb = 'plain'

        # Electrical conductivity and permeability
        self.mu        = 1.0  # magnetic permeability ratio fluid outer core / vacuum
        self.mu_i2o    = 1.0  # magnetic permeability ratio solid inner core / fluid outer core
        self.sigma_i2o = 1.0  # electrical conductivity ratio solid inner core / fluid outer core

        # Magnetic field strength and magnetic diffusivity:
        # Either use the Elsasser number and the magnetic Prandtl number (Lambda and Pm, e.g. Lambda = ssak(0.1,1e-3,Ek),
        # Pm = pmak(1,1e-3,Ek)), from which set_scales() computes Em = Ek/Pm, Le2 = Lambda*Em and Le = sqrt(Le2),
        self.Lambda = 0.1
        self.Pm     = 0.01
        # or set the Lehnert number Le and the magnetic Ekman number Em directly (then Lambda and Pm are not used).
        self.Le  = None
        self.Le2 = None
        self.Em  = None

        # Normalization of the background magnetic field
        self.cnorm = 1
        # self.cnorm = 'rms_cmb'                     # Sets the radial rms field at the CMB as unity
        # self.cnorm = 'mag_energy'                  # Unit magnetic energy as in Luo & Jackson 2022 (I. Torsional oscillations)
        # self.cnorm = 'Schmitt2012'                 # as above but times 2 (integral of B0**2 = 4)
        # self.cnorm = 3.863752890                   # G101 of Schmitt 2012, ricb = 0.35 (FDM l=1, same as 'Schmitt2012')
        # self.cnorm = 4.067144                      # Zhang & Fearn 1994,   ricb = 0.35 (not checked against the other normalisations)
        # self.cnorm = 15*np.sqrt(21/(46*np.pi))     # G21 dipole,           ricb = 0 (same as 'Schmitt2012')
        # self.cnorm = 1.094357234                   # simplest FDM, l=1,    ricb = 0 (same as 'Schmitt2012')
        # self.cnorm = 3.438024656                   # simplest FDM, l=1,    ricb = 0.001 (same as 'Schmitt2012')
        # self.cnorm = 0.09530063707257978           # Luo_S1 ricb = 0, unit mag_energy (full sphere; with an inner core keeps the same currents in the fluid)
        # self.cnorm = 0.6972166887783963            # Luo_S1 ricb = 0, rms_Bs=1 (volume rms of B_s)
        # self.cnorm = 0.005061567359972097          # Luo_S2 ricb = 0, unit mag_energy (full sphere; with an inner core keeps the same currents in the fluid)
        # self.cnorm = 0.0158567582314039            # Luo_S2 ricb = 0, rms_Bs=1 (volume rms of B_s)



        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------- Rotational dynamics parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.rotdyn = 0  # Set to 1 to let the mantle and the solid inner core respond to axial torques

        # Mantle and inner core axial moments of inertia, dimensionless (i.e. in units of rho*L**5 = density*unit_of_length**5)
        self.MoIZ_M  = 11.16
        self.MoIZ_IC = 9.2e-3

        # Gravitational torque constant (divided by rho*L**5*Omega**2)
        self.gTorque = 5e-8

        # Solid inner core relaxation time tauIC times the angular rotation rate Omega (Omega*tauIC, a dimensionless number)
        self.OmgtauIC = 2.3e3  # For Earth use 2.3e3 if tauIC = 1 year

        # torque switchboard
        self.vtrq = 1
        self.mtrq = 1



        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Thermal parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.thermal = 0  # Use 1 or 0 to include or not the temperature equation and the buoyancy force (Boussinesq)

        # Prandtl number: ratio of viscous to thermal diffusivity
        self.Prandtl = 1.0
        # "Thermal" Ekman number, Ek/Prandtl if not set
        self.Etherm = None

        # Background isentropic temperature gradient dT/dr choices:
        self.heating = 'internal'        # dT/dr = -beta * r         temp_scale = beta * ro**2
        # self.heating = 'differential'  # dT/dr = -beta * r**-2     temp_scale = Ti-To      beta = (Ti-To)*ri*ro/(ro-ri)
        # self.heating = 'two zone'      # dT/dr = K * ut.twozone()  temp_scale = -ro * K
        # self.heating = 'user defined'  # dT/dr = K * ut.BVprof()   temp_scale = -ro * K

        # Rayleigh number as Ra = alpha * g0 * ro^3 * temp_scale / (nu*kappa), alpha is the thermal expansion coeff,
        # g0 the gravity accel at ro, ro is the cmb radius (the length scale), nu is viscosity, kappa is thermal diffusivity.
        self.Ra = 2*1.2584e8
        # Ra_Silva = 0.0; Ra = Ra_Silva * (1/(1-ricb))**6
        # Ra_Monville = 0.0; Ra = 2*Ra_Monville

        # Alternatively, you can specify directly the squared ratio of a reference Brunt-Väisälä freq. and the rotation rate.
        # The reference Brunt-Väisälä freq. squared is defined as -alpha*g0*temp_scale/ro. See the non-dimensionalization notes
        # in the documentation. -Ra * Ek**2 / Prandtl if not set.
        self.BV2 = None

        # Additional arguments for 'Two zone' or 'User defined' case (modify if needed).
        self.rc  = 0.7  # transition radius
        self.h   = 0.1  # transition width
        self.rsy = -1   # radial symmetry
        self.args = None  # [rc, h, rsy] if not set

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
        # "Compositional" Ekman number, Ek/Schmidt if not set
        self.Ecomp = None

        # Background isentropic composition gradient dC/dr choices:
        self.comp_background = 'internal'        # dC/dr = -beta * r         comp_scale = beta * ro**2
        # self.comp_background = 'differential'  # dC/dr = -beta * r**-2     comp_scale = Ci-Co

        # Compositional Rayleigh number
        self.Ra_comp = 0.0
        # Ra_comp_Silva = 0.0; Ra_comp = Ra_comp_Silva * (1/(1-ricb))**6
        # Ra_comp_Monville = 0.0; Ra_comp = 2*Ra_comp_Monville

        # Alternatively, specify directly the squared ratio of a reference compositional Brunt-Väisälä frequency
        # and the rotation rate. -Ra_comp * Ek**2 / Schmidt if not set.
        self.BV2_comp = None

        # Additional arguments for 'Two zone' or 'User defined' case (modify if needed).
        self.rcc  = 0.7  # transition radius
        self.hc   = 0.1  # transition width
        self.rsyc = -1   # radial symmetry
        self.args_comp = None  # [rcc, hc, rsyc] if not set

        # Compositional boundary conditions
        # 0 for constant composition, xi=0
        # 1 for constant flux, (d/dr)xi=0
        self.bci_compositional = 1   # ICB
        self.bco_compositional = 1   # CMB



        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------- Time scale
        # ----------------------------------------------------------------------------------------------------------------------
        # Choose the time scale, see the non-dimensionalization notes in the documentation. set_scales() sets the
        # dimensionless angular velocity OmgTau accordingly, unless OmgTau is set directly.
        self.timescale = 'rotation'    # OmgTau = 1
        # self.timescale = 'viscous'   # OmgTau = 1/Ek, viscous diffusion time scale
        # self.timescale = 'Alfven'    # OmgTau = 1/Le, Alfvén time scale
        # self.timescale = 'magnetic'  # OmgTau = 1/Em, magnetic diffusion time scale
        self.OmgTau = None



        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------- Resolution
        # ----------------------------------------------------------------------------------------------------------------------
        # Number of cpus
        self.ncpus = 4

        # Chebyshev polynomial truncation level, Ncheb(Ek) if not set. N must be even if ricb = 0.
        self.N     = None  # for the fluid core
        self.N_cic = 64    # for the field inside the ic (innercore = 'conducting, Chebys'); 48 gave broken eigenvectors at sigma_i2o = 1e-2

        # Spherical harmonic truncation lmax and approx lmax/N ratio g; lmax = ellmax(ncpus, g, m, N) if not set.
        # If setting lmax manually, then it must be even if m is odd, and lmax-m+1 should be divisible by 2*ncpus
        self.g        = 1.0
        self.lmax     = None
        self.lmax_cic = None  # lmax if not set



        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------- SLEPc solver options
        # ----------------------------------------------------------------------------------------------------------------------
        # Set track_target = 1 below to track an eigenvalue, 0 otherwise.
        # Assumes a preexisting 'track_target' file with target data (read by set_scales() into rtau and itau)
        # Set track_target = 2 to write initial 'track_target' file, see also postprocess.py
        self.track_target = 0

        # Target for the solver, tau = rtau + itau*1j if not set
        # real part is damping
        # imaginary part is frequency (positive is retrograde)
        self.rtau = 0.0
        self.itau = 1.0
        self.tau  = None

        self.which_eigenpairs = 'TM'  # Use 'TM' for shift-and-invert
        # L/S/T & M/R/I
        # L largest, S smallest, T target
        # M magnitude, R real, I imaginary

        # Number of desired eigenvalues
        self.nev = 3

        # Number of vectors in Krylov space for solver
        # self.ncv = 100

        # Maximum iterations to converge to an eigenvector
        self.maxit = 50

        # Tolerance for solver
        self.tol = 1e-16
        # Tolerance for the thermal/compositional matrix
        self.tol_tc = 1e-6

        # PETSc/SLEPc/MUMPS runtime options, loaded by solve_nopp.py into the PETSc options database
        # (option name without the leading '-'; use '' for flags without a value).
        # solve_nopp.py passes only the group matching the problem type (forcing == 0 or not).
        # Anything given on the command line (e.g. via $opts in runKore.sh) takes precedence.
        self.petsc_opts = {

            # ---------------------------------------- eigenvalue problems (forcing == 0): eps_*, st_*
            'st_type'                      : 'sinvert',             # shift-and-invert around tau (use with 'TM')
            'st_pc_factor_mat_solver_type' : 'mumps',               # direct LU solve with MUMPS
            'st_mat_mumps_cntl_1'          : 1e-8,                  # pivot threshold; avoids INFOG(1)=-9; 1e-6 with BLR broke eigenvectors at N>=840
            'st_mat_mumps_icntl_35'        : 2,                     # block low-rank (BLR) factorization: ~13% less memory, ~2x faster
            'st_mat_mumps_cntl_7'          : 1e-14,                 # BLR tolerance; 1e-12 broke eigenvectors at N>=640-840
            'eps_error_relative'           : '::ascii_info_detail', # print relative errors after the solve
            # 'eps_balance'                : 'twoside',             # cleaner eigenvalues for final runs, ~+50% time
            # 'st_mat_mumps_icntl_14'      : 50,                    # extra MUMPS workspace (%), only if -9 still appears
            # 'st_mat_mumps_icntl_22'      : 1,                     # out-of-core factors, cuts memory ~half
            # 'st_mat_mumps_ooc_tmpdir'    : '/nvm/scratch',        # put OOC files on a real disk, not tmpfs /tmp
            # 'st_mat_mumps_icntl_28'      : 2,                     # parallel ordering ...
            # 'st_mat_mumps_icntl_29'      : 2,                     # ... with ParMETIS

            # ---------------------------------------- forced problems (forcing > 0): ksp_*, pc_*, mat_*
            'ksp_type'                     : 'preonly',             # direct solve, no Krylov iterations
            'pc_type'                      : 'lu',                  # (default GMRES/ILU does not converge)
            'pc_factor_mat_solver_type'    : 'mumps',               # LU with MUMPS
            # 'mat_mumps_cntl_1'           : 1e-6,                  # pivot threshold, if MUMPS reports INFOG(1)=-9
            # 'mat_mumps_icntl_22'         : 1,                     # out-of-core factors, cuts memory ~half
            # 'mat_mumps_ooc_tmpdir'       : '/nvm/scratch',        # put OOC files on a real disk, not tmpfs /tmp
        }

        # Pre-scaling of the eigenvalue problem (forcing == 0 only; ignored for forced problems).
        # With prescale = 1, solve_nopp.py applies Ruiz row+column equilibration to A - tau*B (10 iterations)
        # and solves Dr*A*Dc y = lambda Dr*B*Dc y instead. Eigenvalues are unchanged; eigenvectors are mapped
        # back (x = Dc*y) before they are written, so spin_doctor and postprocessing see unscaled vectors.
        # Lowers the condition number of A - tau*B by 6-8 orders of magnitude and gave cleaner eigenvectors
        # (spin_doctor resid u ~1e-7 instead of ~1e-4 for modes 1-2 at N = 976, with and without BLR).
        # Costs 7-8 s at N = 976 on 8 ranks, before the factorization; no change in peak memory.
        # solve_nopp's ||Ax-kBx||/||kx|| is then printed for the scaled problem; judge accuracy with spin_doctor.
        # Note: does not converge together with -eps_true_residual at tol = 1e-15 or below.
        self.prescale = 1



    def set_scales(self):
        '''
        Sets the parameters derived from the others, i.e. those that are still None: the magnetic and diffusion
        numbers, the buoyancy factors, the time scale factor OmgTau, the resolution and the solver target.
        Call it after setting the parameters (parameters.py does it at the end). A derived parameter that was set
        explicitly is kept.
        '''

        # magnetic field strength and diffusivity
        if self.Em is None:
            self.Em = self.Ek/self.Pm
        if self.Le is None and self.Le2 is None:
            self.Le2 = self.Lambda*self.Em
        if self.Le is None:
            self.Le = np.sqrt(self.Le2)
        if self.Le2 is None:
            self.Le2 = self.Le**2

        # thermal and compositional diffusion and buoyancy
        if self.Etherm is None:
            self.Etherm = self.Ek/self.Prandtl
        if self.BV2 is None:
            self.BV2 = -self.Ra * self.Ek**2 / self.Prandtl
        if self.args is None:
            self.args = [self.rc, self.h, self.rsy]
        if self.Ecomp is None:
            self.Ecomp = self.Ek/self.Schmidt
        if self.BV2_comp is None:
            self.BV2_comp = -self.Ra_comp * self.Ek**2 / self.Schmidt
        if self.args_comp is None:
            self.args_comp = [self.rcc, self.hc, self.rsyc]

        # time scale
        assert self.rotation in (0, 1), 'rotation must be 0 or 1'
        if self.OmgTau is None:
            assert self.rotation or self.timescale != 'rotation', \
                "rotation = 0 needs another timescale (e.g. 'viscous') or OmgTau set directly"
            self.OmgTau = { 'rotation' : 1,
                            'viscous'  : 1/self.Ek if self.Ek != 0 else None,
                            'Alfven'   : 1/self.Le if self.Le != 0 else None,
                            'magnetic' : 1/self.Em if self.Em != 0 else None }[self.timescale]
            assert self.OmgTau is not None, "timescale = '%s' needs a nonzero diffusivity or field" % self.timescale

        # resolution
        if self.N is None:
            self.N = self.Ncheb(self.Ek)
        if self.lmax is None:
            self.lmax = self.ellmax(self.ncpus, self.g, self.m, self.N)
        if self.lmax_cic is None:
            self.lmax_cic = self.lmax

        # solver target
        if self.track_target == 1:  # read target from file
            tt = np.loadtxt('track_target')
            self.rtau = tt[0]
            self.itau = tt[1]
        if self.tau is None:
            self.tau = self.rtau + self.itau*1j



    def Ncheb(self, Ek):
        '''
        Returns the truncation level N for the Chebyshev expansion according to the Ekman number
        Please experiment and adapt to your particular problem. N must be even.
        '''
        # Fitted to resolution tests (Linux server, torsional-mode case: Pm = 0.1, Lambda = 10, FDM field, TWA
        # mantle, insulating IC, lmax ~ N). N_min, the smallest N with every spin_doctor residual < 1e-4, follows
        # N_min = 3.96*Ek**-0.242 within 7% for 1e-9 <= Ek <= 1e-5 (e.g. 112 at 1e-6, 192 at 1e-7, 368 at 1e-8,
        # 592 at 1e-9). The formula below adds a 15% margin and rounds up to a multiple of 8 (so that lmax from
        # ellmax equals N - 1). With the margin, eigenvalues are converged to ~1e-6 - 1e-5 (relative).
        # Gives 80, 136, 232, 400, 688 at Ek = 1e-5 ... 1e-9 (688: 34 GB with BLR on 8 ranks); 1200 at
        # 1e-10 (extrapolated, ~115-120 GB with BLR, above a 128 GB machine's safe limit).
        # Other setups (stronger fields, other boundary conditions, thermal/compositional) may need more;
        # check spin_doctor's residuals. The old formula, 17*Ek**-0.2, gave about twice N_min.
        if Ek !=0 :
            out = 8*int(np.ceil(4.55*Ek**-0.242/8))
            # out = int(17*Ek**-0.2)  # before 2026-10-06
        else:
            out = 48  #

        return max(48, out + out%2)



    def ellmax(self, ncpus, g, m, N):
        '''
        Returns the lmax given a radial truncation level N,
        ncpus, azimuthal wave number m, and an approx. N/lmax ratio g.
        '''
        return int( 2*ncpus*( np.floor_divide( g*N, 2*ncpus ) ) + m - 1 )

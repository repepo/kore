import numpy as np
#import targets as tg


class default_params():

    def __init__(self):

        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------ Physics modules       
        # ----------------------------------------------------------------------------------------------------------------------
        self.thermal = 0  # set to 1 to include the thermal (entropy) equation; the Navier-Stokes equation is always solved


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------- Solution symmetry and inner core size
        # ----------------------------------------------------------------------------------------------------------------------
        self.m    = 0  # Azimuthal wave number
        self.symm = 1  # Equatorial symmetry, 1 for symmetric, -1 for antisymmetric
        self.ricb = 0 # Solid inner core radius 


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
        self.Etherm = 0

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

        # Differential rotation parameters. Set to 0 for solid body rotation. (Implemented for timescale="rotation") -----------
        # ----------------------------------------------------------------------------------------------------------------------
        self.diff_rot = 0 # If 1, the code will solve for a perturbation to a background state with differential rotation. 
        # Differential rotation type 
        self.diff_rot_type = 'Y20' # Ω(θ) = Ω_ref * [1 + ΔΩ Y20(θ)], ΔΩ = par.diff_rot_amplitude
        # self.diff_rot_type = 'Y20-wall-bounded' # Ω(θ) = Ω_ref * [1 + ΔΩ * (1 - r) * (r - ricb) * Y20(θ)], ΔΩ = par.diff_rot_amplitude
        # self.diff_rot_type = 'shellular' # [Baruteau, Rieutord 2012] : Ω(r) = Ω_ref * (r / R)**σ, σ = par.diff_rot_amplitude
        # self.diff_rot_type = 'cylindrical' # [Baruteau, Rieutord 2012] : Ω(r, θ) = Ω_ref * [1 + (ε * (r / R)**2 * sin(θ)**2)], ε = par.diff_rot_amplitude
        # self.diff_rot_type = 'conical' #[Guenel et al., 2016] : Ω(r, θ) = Ω_ref * [1 + (ε * sin(θ)**2)], ε = par.diff_rot_amplitude
        # self.diff_rot_type = "shellular_boussinesq" #[Mirouh et al. 2016] : Ω(r) = Ω_ref * (1 + (1/2) * (N**2) * (1 - (r / R)**2))
        # self.diff_rot_type = 'solar' # Not implemented yet

        self.diff_rot_amplitude = 1.0  # Amplitude of the differential rotation, corresponds to different parameter depending on the DR type

        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Thermal parameters
        # ----------------------------------------------------------------------------------------------------------------------

        if self.model_type == 'Boussinesq':
            self.heating = 'differential'  # 'internal' or 'differential' heating

        # Thermal boundary conditions: 0 for Dirichlet, 1 for Neumann
        # (the unknown is the entropy s = δs/c_p: 0 is fixed entropy s=0, 1 is zero diffusive entropy flux s'=0; used only if ThermaD>0)
        self.bci_thermal = 0
        self.bco_thermal = 0


        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Forcing parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.forcing = 0  # Uncomment this line for eigenvalue problems
        # self.forcing = 3  # For Rovira-Navarro 2018 tidal body forcing, m=0,2 must be symm, m=1 antisymm. Leaks power!
        # self.forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
        # self.forcing = 7  # Longitudinal libration boundary forcing, m={0, 2}, symm, no-slip
        # self.forcing = 9  # Radial, symmetric, m=2 boundary flow forcing.

        # Forcing frequency (ignored if forcing == 0)
        self.forcing_frequency = 1.0  # negative is prograde

        # Forcing amplitude. Body forcing amplitude will use the cmb value
        self.forcing_amplitude_cmb = 1.0
        self.forcing_amplitude_icb = 0.0


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------- Unit of time and force switches
        # ----------------------------------------------------------------------------------------------------------------------
        self.timescale = 'free fall'
        self.Gaspard   = 1  # Omega*Tau                 Coriolis force factor. Set to 1 for unit time Tau = 1/Omega
        self.Beyonce   = 0  # (N0*Tau)**2               Buoyancy force factor. Set to 1 for unit time Tau = 1/N0 = sqrt(r0/g0)
        self.ViscosD   = 0  # nu0 * Tau / r0**2         Viscous force factor. Set to 1 for viscous diffusion time scale. This is the Ekman number if Tau = 1/Omega
        self.ThermaD   = 0  # kappa0 * Tau / r0**2      Thermal diffusion factor. Set to 1 for thermal diffusion time scale


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
        # Target eigenvalue tau = rtau + i*itau (shift for the shift-and-invert solver)
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
        self.tol = 1e-16

        # PETSc/SLEPc/MUMPS runtime options, loaded by solve.py into the PETSc options database
        # (option name without the leading '-'; use '' for flags without a value).
        # solve.py passes only the group matching the problem type (forcing == 0 or not).
        # Anything given on the command line (e.g. mpiexec -n 4 ./bin/solve.py -eps_balance twoside) takes precedence.
        self.petsc_opts = {

            # ---------------------------------------- eigenvalue problems (forcing == 0): eps_*, st_*
            'st_type'                      : 'sinvert',             # shift-and-invert around tau (use with 'TM')
            'st_pc_factor_mat_solver_type' : 'mumps',               # direct LU solve with MUMPS
            'st_mat_mumps_cntl_1'          : 0.01,                  # pivot threshold, MUMPS's default; 1e-8 cost accuracy in thermal runs
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
        # Default on (set prescale = 0 to switch it off). With prescale = 1, solve.py applies Ruiz row+column equilibration to A - tau*B (10 iterations)
        # and solves Dr*A*Dc y = lambda Dr*B*Dc y instead. Eigenvalues are unchanged; eigenvectors are mapped
        # back (x = Dc*y) before they are written, so spin_doctor and postprocessing see unscaled vectors.
        # Lowers the condition number of A - tau*B by 6-8 orders of magnitude and gave cleaner eigenvectors
        # (conductive_IC torsional-mode tests: spin_doctor resid u ~1e-7 instead of ~1e-4 for modes 1-2 at N = 976).
        # Costs 7-8 s at N = 976 on 8 ranks, before the factorization; no change in peak memory.
        # solve.py's ||Ax-kBx||/||kx|| is then printed for the scaled problem; judge accuracy with spin_doctor.
        self.prescale = 1


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
            self.ViscosD = self.Ek      # nu0 * Tau / r0**2         
            self.ThermaD = self.Etherm  # kappa0 * Tau / r0**2      

        elif self.timescale == "viscous":  # viscous diffusion time scale

            self.Gaspard = 1/self.Ek            # Omega*Tau
            self.Beyonce = 0                    # (N0*Tau)**2
            self.ViscosD = 1                    # nu0 * Tau / r0**2
            self.ThermaD = self.Etherm/self.Ek  # kappa0 * Tau / r0**2
        
        
        elif self.timescale == "free fall":  # unit time Tau = 1/N0 = sqrt(r0/g0)

            self.Gaspard = 0  # Omega*Tau
            self.Beyonce = 1  # (N0*Tau)**2
            self.ViscosD = 0  # nu0 * Tau / r0**2
            self.ThermaD = 0  # kappa0 * Tau / r0**2


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
    
    def set_eigv_frame(self, frame):
        if frame=="inertial":
            self.rtau = self.rtau
            self.itau = self.itau + self.m
        elif frame=="rotating":
            self.rtau = self.rtau
            self.itau = self.itau

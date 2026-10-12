import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
from parameters import par
import utils as ut

if par.diff_rot : import diff_rot_coefficients as dr
if par.diff_rot and par.thermal : import heat_dr_coefficients as hdr


# In the following loop we read all the submatrices needed (as per submatrices.py),
# and create corresponding operator names as global variables
fname = [f for f in glob.glob('*.mtx')]
for label in fname :
    varlabel = label[:-4]
    globals()[varlabel] = ss.csr_matrix(sio.mmread(label))



# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Navier-Stokes equation operators
# ----------------------------------------------------------------------------------------------------------------------



def inertia(l, section, component, offdiag):  # ---------------------------------------------------------------- inertia

    out = 0

    L = l*(l+1.)

    if offdiag == 0:

        if section == 'u' and component == 'upol':

            out =  1.0 * ( -u1lho2_D0-u2lho1_D0+L*u3_D0
                           -u1lho1_D1-2*u2_D1
                           -u1_D2 )

        elif section == 'v' and component == 'utor':

            out = 1.0 * v1_D0

    return out/L**2



def coriolis(l, section, component, offdiag):  # ------------------------------------------------- Coriolis force 2z x u

    out  = 0
    offd = 0
    m = par.m
    L = l*(l+1.)

    if section == 'u':  # ------------------------------------------------------- 2curl

        if component == 'upol':

            if offdiag == 0:
                
                out = 1j*m*( - u3_D0 + ((1/L)-1)*u2lho1_D0 + (1/L)*u1lho2_D0 
                             + (1/L)*( 2*u2_D1 + u1lho1_D1 + u1_D2 ) )

        elif component == 'utor':

            if offdiag == -1:

                C = (l-1)*np.sqrt(l**2-m**2) / (l*(2*l-1.))
                out = C*( (l-1)*u2_D0 - u1_D1 )

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 1:

                C = (l+2.)*np.sqrt((l+m+1.)*(l-m+1)) / ((2.*l+3.)*(l+1))
                out = C*( -(l+2)*u2_D0 - u1_D1 )             

                if ut.symm1 == -1:
                    offd = 1

    if section == 'v':  # ------------------------------------------------------- 1curl

        if component == 'upol':

            if offdiag == -1:

                C = (l-1)*np.sqrt(l**2-m**2) / (l*(2*l-1.))
                out = C*( (l-1)*v2_D0 - v1lho1_D0 - v1_D1 )              

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:

                C = (l+2)*np.sqrt((l+m+1.)*(l-m+1)) / ((l+1)*(2*l+3))
                out = C*( -(l+2)*v2_D0 - v1lho1_D0 - v1_D1 )            

                if ut.symm1 == 1:
                    offd = 1
                

        elif component == 'utor':

            if offdiag == 0:

                out = -(1j*m/L) * v1_D0       

    return [ 2*par.Gaspard * out/L**2, offd ]



def differential_rotation(l, section, component, offdiag):  # ------------------------ Differential rotation forcing terms
    """
        Differential rotation background velocity field forcing terms
        On the form Ω(r, θ) = Ω0 * [ 1 + ΔΩ * (f0(r) * Y00(θ) + f2(r) * Y20(θ)) ], with : 
                ΔΩ : Differential rotation amplitude (depend on the differential rotation type [par.diff_rot_type])
                f0(r) : Differential rotation radial profile for the Y00 part, from which we define : 
                    abu = f0
                    spv = r*f0'
                    psl = r^2*f0''
                f2(r) : Differential rotation radial profile for the Y20 part, from which we define : 
                    aub = f2
                    svp = r*f2'
                    pls = r^2*f2''

        The forcing includes, for both (f(r), g(θ)) = (f0(r), 1) and (f2(r), Y20(θ))
            f(r)*{ΔΩ * [g(θ) * (im * u) + g(θ) * (2*ez x u) + g'(θ) * (sin(θ) * uθ * eϕ)]} 
            + r f'(r)*{ΔΩ * g(θ) * (sin(θ) * ur * eϕ)} 
        times Gaspard, as the background flow is U0 = Gaspard ΔΩ r sin(θ) eϕ in units of the time scale
    """


    out  = 0
    offd = 0
    m = par.m

    if section == 'u':  # ------------------------------------------------------- 2curl

        if component == 'upol':
            
            if offdiag == -2 : 

                out = dr.f_2C_D0P(l, m, offdiag)*u3aub0_D0 \
                    + dr.f_2C_D1P(l, m, offdiag)*u2aub0_D1 \
                    + dr.f_2C_lho1_D0P(l, m, offdiag)*u2aub0lho1_D0 \
                    + dr.f_2C_lho2_D0P(l, m, offdiag)*u1aub0lho2_D0 \
                    + dr.f_2C_lho1_D1P(l, m, offdiag)*u1aub0lho1_D1 \
                    + dr.f_2C_D2P(l, m, offdiag)*u1aub0_D2 \
                    + dr.df_2C_D0P(l, m, offdiag)*u3svp0_D0 \
                    + dr.df_2C_D1P(l, m, offdiag)*u2svp0_D1 \
                    + dr.df_2C_lho1_D0P(l, m, offdiag)*u2svp0lho1_D0 \
                    + dr.d2f_2C_D0P(l, m, offdiag)*u3pls0_D0

                offd = -1

            elif offdiag == 0:
                out = dr.f_2C_D0P(l, m, offdiag)*u3aub0_D0 \
                    + dr.f_2C_D1P(l, m, offdiag)*u2aub0_D1 \
                    + dr.f_2C_lho1_D0P(l, m, offdiag)*u2aub0lho1_D0 \
                    + dr.f_2C_lho2_D0P(l, m, offdiag)*u1aub0lho2_D0 \
                    + dr.f_2C_lho1_D1P(l, m, offdiag)*u1aub0lho1_D1 \
                    + dr.f_2C_D2P(l, m, offdiag)*u1aub0_D2 \
                    + dr.df_2C_D0P(l, m, offdiag)*u3svp0_D0 \
                    + dr.df_2C_D1P(l, m, offdiag)*u2svp0_D1 \
                    + dr.df_2C_lho1_D0P(l, m, offdiag)*u2svp0lho1_D0 \
                    + dr.d2f_2C_D0P(l, m, offdiag)*u3pls0_D0 \
                    + dr.f0_2C_D0P(l, m, offdiag)*u3abu0_D0 \
                    + dr.f0_2C_D1P(l, m, offdiag)*u2abu0_D1 \
                    + dr.f0_2C_lho1_D0P(l, m, offdiag)*u2abu0lho1_D0 \
                    + dr.f0_2C_lho2_D0P(l, m, offdiag)*u1abu0lho2_D0 \
                    + dr.f0_2C_lho1_D1P(l, m, offdiag)*u1abu0lho1_D1 \
                    + dr.f0_2C_D2P(l, m, offdiag)*u1abu0_D2 \
                    + dr.df0_2C_D0P(l, m, offdiag)*u3spv0_D0 \
                    + dr.df0_2C_D1P(l, m, offdiag)*u2spv0_D1 \
                    + dr.df0_2C_lho1_D0P(l, m, offdiag)*u2spv0lho1_D0 \
                    + dr.d2f0_2C_D0P(l, m, offdiag)*u3psl0_D0
            elif offdiag == 2 : 

                out = dr.f_2C_D0P(l, m, offdiag)*u3aub0_D0 \
                    + dr.f_2C_D1P(l, m, offdiag)*u2aub0_D1 \
                    + dr.f_2C_lho1_D0P(l, m, offdiag)*u2aub0lho1_D0 \
                    + dr.f_2C_lho2_D0P(l, m, offdiag)*u1aub0lho2_D0 \
                    + dr.f_2C_lho1_D1P(l, m, offdiag)*u1aub0lho1_D1 \
                    + dr.f_2C_D2P(l, m, offdiag)*u1aub0_D2 \
                    + dr.df_2C_D0P(l, m, offdiag)*u3svp0_D0 \
                    + dr.df_2C_D1P(l, m, offdiag)*u2svp0_D1 \
                    + dr.df_2C_lho1_D0P(l, m, offdiag)*u2svp0lho1_D0 \
                    + dr.d2f_2C_D0P(l, m, offdiag)*u3pls0_D0

                offd = 1

        elif component == 'utor':

            if offdiag == -3:

                out = dr.f_2C_D0T(l, m, offdiag)*u2aub0_D0 \
                    + dr.f_2C_D1T(l, m, offdiag)*u1aub0_D1 \
                    + dr.df_2C_D0T(l, m, offdiag)*u2svp0_D0
                
                if ut.symm1 == -1 :
                    offd = -1
                elif ut.symm1 == 1 :
                    offd = -2

            elif offdiag == -1:

                out = dr.f_2C_D0T(l, m, offdiag)*u2aub0_D0 \
                    + dr.f_2C_D1T(l, m, offdiag)*u1aub0_D1 \
                    + dr.df_2C_D0T(l, m, offdiag)*u2svp0_D0 \
                    + dr.f0_2C_D0T(l, m, offdiag)*u2abu0_D0 \
                    + dr.f0_2C_D1T(l, m, offdiag)*u1abu0_D1 \
                    + dr.df0_2C_D0T(l, m, offdiag)*u2spv0_D0

                if ut.symm1 == 1 :
                    offd = -1

            elif offdiag == 1:
                
                out = dr.f_2C_D0T(l, m, offdiag)*u2aub0_D0 \
                    + dr.f_2C_D1T(l, m, offdiag)*u1aub0_D1 \
                    + dr.df_2C_D0T(l, m, offdiag)*u2svp0_D0 \
                    + dr.f0_2C_D0T(l, m, offdiag)*u2abu0_D0 \
                    + dr.f0_2C_D1T(l, m, offdiag)*u1abu0_D1 \
                    + dr.df0_2C_D0T(l, m, offdiag)*u2spv0_D0
                
                
                if ut.symm1 == -1:
                    offd = 1

            elif offdiag == 3:

                out = dr.f_2C_D0T(l, m, offdiag)*u2aub0_D0 \
                    + dr.f_2C_D1T(l, m, offdiag)*u1aub0_D1 \
                    + dr.df_2C_D0T(l, m, offdiag)*u2svp0_D0
                
                if ut.symm1 == -1:
                    offd = 2
                elif ut.symm1 == 1 : 
                    offd = 1

    if section == 'v':  # ------------------------------------------------------- 1curl

        if component == 'upol':

            if offdiag == -3:

                out = dr.f_1C_D0P(l, m, offdiag) * v2aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, offdiag) * v1aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, offdiag) * v1aub0_D1 \
                    + dr.df_1C_D0P(l, m, offdiag) * v2svp0_D0
                
                if ut.symm1 == -1:
                    offd = -2
                elif ut.symm1 == 1: 
                    offd = -1

            elif offdiag == -1:

                out = dr.f_1C_D0P(l, m, offdiag) * v2aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, offdiag) * v1aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, offdiag) * v1aub0_D1 \
                    + dr.df_1C_D0P(l, m, offdiag) * v2svp0_D0 \
                    + dr.f0_1C_D0P(l, m, offdiag) * v2abu0_D0 \
                    + dr.f0_1C_lho1_D0P(l, m, offdiag) * v1abu0lho1_D0 \
                    + dr.f0_1C_D1P(l, m, offdiag) * v1abu0_D1 \
                    + dr.df0_1C_D0P(l, m, offdiag) * v2spv0_D0

                
                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:

                out = dr.f_1C_D0P(l, m, offdiag) * v2aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, offdiag) * v1aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, offdiag) * v1aub0_D1 \
                    + dr.df_1C_D0P(l, m, offdiag) * v2svp0_D0 \
                    + dr.f0_1C_D0P(l, m, offdiag) * v2abu0_D0 \
                    + dr.f0_1C_lho1_D0P(l, m, offdiag) * v1abu0lho1_D0 \
                    + dr.f0_1C_D1P(l, m, offdiag) * v1abu0_D1 \
                    + dr.df0_1C_D0P(l, m, offdiag) * v2spv0_D0

                if ut.symm1 == 1:
                    offd = 1

            elif offdiag == 3:

                out = dr.f_1C_D0P(l, m, offdiag) * v2aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, offdiag) * v1aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, offdiag) * v1aub0_D1 \
                    + dr.df_1C_D0P(l, m, offdiag) * v2svp0_D0

                if ut.symm1 == -1:
                    offd = 1
                elif ut.symm1 == 1:
                    offd = 2

        elif component == 'utor':

            if offdiag == -2 : 

                out = dr.f_1C_D0T(l, m, offdiag) * v1aub0_D0

                offd = -1

            elif offdiag == 0:

                out = dr.f_1C_D0T(l, m, offdiag) * v1aub0_D0 \
                    + dr.f0_1C_D0T(l, m, offdiag) * v1abu0_D0
            
            elif offdiag == 2 : 
                
                out = dr.f_1C_D0T(l, m, offdiag) * v1aub0_D0

                offd = 1

    return [ par.Gaspard * out/(l*(l+1.))**3, offd ]


def viscous_diffusion(l, section, component, offdiag):  # ------------------------------------------------ viscous force

    out = 0
    L= l*(l+1.)

    if (offdiag == 0)&(par.ViscosD>0):

        if section == 'u' and component == 'upol':

            out = 1.0 * ( - u1lh13vsc0_D0 - u1lh22vsc0_D0 - u1lho4vsc0_D0 + u2lh12vsc0_D0 - 3* u2lho3vsc0_D0 + (2/3)*L* u3lh11vsc0_D0 
                          + 2* u3lho2vsc0_D0 - 2* u4lho1vsc0_D0 + 2*L* u5vsc0_D0 - (L**2)* u5vsc0_D0 - u1lh12vsc1_D0 - 2* u1lho3vsc1_D0 
                          + u2lh11vsc1_D0 - 2* u2lho2vsc1_D0 + 2* u3lho1vsc1_D0 + L* u3lho1vsc1_D0 - 2* u4vsc1_D0 - 2*L* u4vsc1_D0 
                          - u1lho2vsc2_D0 + u2lho1vsc2_D0 + 2*u3vsc2_D0 - L* u3vsc2_D0
                          - 3* u1lh12vsc0_D1 - 3* u1lho3vsc0_D1 - 6* u2lho2vsc0_D1 + 2* u3lho1vsc0_D1 + 2*L* u3lho1vsc0_D1 - u1lh11vsc1_D1 
                          - 4* u1lho2vsc1_D1 - 2* u2lho1vsc1_D1 + 2* u3vsc1_D1 + 2*L* u3vsc1_D1 - u1lho1vsc2_D1
                          - u1lh11vsc0_D2 - 4* u1lho2vsc0_D2 - 4* u2lho1vsc0_D2 + 2*L* u3vsc0_D2 - 3* u1lho1vsc1_D2 - 4* u2vsc1_D2 - u1vsc2_D2
                          - 2* u1lho1vsc0_D3 - 4* u2vsc0_D3 - 2* u1vsc1_D3
                          - u1vsc0_D4 )

        elif section == 'v' and component == 'utor':

            out = 1.0 * ( -v2lho1vsc0_D0 - L*v3vsc0_D0 - v2vsc1_D0
                          +v1lho1vsc0_D1 + 2*v2vsc0_D1 + v1vsc1_D1
                          +v1vsc0_D2 )


    return par.ViscosD * out/L**2



def buoyancy(l, section, component, offdiag): 

    out = 0
    L = l*(l+1.)

    if (section == 'u') and (offdiag == 0) :

        out = 1.0 * u2gra0_D0

    return par.Beyonce * out/L**2



def entropy(l, section, component, offdiag):  # rʰ p s  (rʰ s if ThermaD=0: divided by p; h = 1, or 2 with the thermal wind)

    out = 0
    L = l*(l+1.)
    
    if (section == 'h') and (offdiag == 0) :

        out = h0pss0_D0 if par.ThermaD > 0 else h0_D0

    return out/L**3



def thermal_advection(l, section, component, offdiag):  # −rʰ p(v⋅∇) S = −rʰ p vᵣ dS/dr  (−rʰ vᵣ dS/dr if ThermaD=0)

    out = 0
    L = l*(l+1.)

    if (section == 'h') and (component == 'upol') and (offdiag == 0) :

        out = -1.0 * ( h1pdS0_D0 if par.ThermaD > 0 else h1dSd0_D0 )
    
    return out/L**2



def thermal_diffrot(l, section, component, offdiag):  # ------------------- Differential rotation terms of the heat equation
    """
        With a background rotation Ω = Ω0 [1 + f0(r) + f2(r) P2(cos θ)] the heat equation gains (docs/heat-equation.md)
            p [ i m Gaspard ΔΩ s + uᵣ dS_tw/dr + (u_θ/r) dS_tw/dθ ],    ΔΩ = f0 + f2 P2,
        where S_tw = σ2(r) P2 + σ4(r) P4 [+ σ6(r) P6] is the thermal-wind background entropy (radial_profiles.sigma2,
        sigma4, sigma6; the two advection terms only if par.thermal_wind, P6 only if thermal_wind = 2). Rows are the
        equation times r², as the other section-h operators (divided by p if ThermaD = 0, and times r instead of r² if
        ThermaD = 0 without the thermal wind; with ThermaD = 0 the profiles tw, tp are σn, σn' without p, ut.adiab_tw).
        component 'temp': s column, Δl = 0, ±2;  'upol': P column, Δl = 0, ±2, ±4 (±6);  'utor': T column, Δl = ±1, ±3 (±5).
        The angular coefficients come from heat_dr_coefficients. Returns [matrix, offd]; the matrix already has the
        sign of the A-matrix right-hand side (the heat rows of A hold -(left-hand side terms)), so add it with +.
    """

    out  = 0
    offd = 0
    m = par.m
    L = l*(l+1.)
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    ls = l + offdiag
    dlmax = 6 if par.thermal_wind == 2 else 4   # largest |Δl| of the P column

    if section == 'h':

        if component == 'temp' and offdiag in [-2, 0, 2]:   # i m Gaspard (f0 + f2 P2) s

            if par.ThermaD > 0:
                out = 1j*m * hdr.coef('P2', l, m, offdiag) * h0pss0aub0_D0
                if offdiag == 0:
                    out = out + 1j*m * h0pss0abu0_D0
            else:   # rows rʰ (eq)/p
                out = 1j*m * hdr.coef('P2', l, m, offdiag) * h0aub0_D0
                if offdiag == 0:
                    out = out + 1j*m * h0abu0_D0
            out = par.Gaspard * out
            offd = int(np.where(lp == ls)[0][0] - np.where(lp == l)[0][0])

        elif component == 'upol' and par.thermal_wind and abs(offdiag) <= dlmax and offdiag % 2 == 0:   # u.grad S_tw, poloidal part

            Ls = ls*(ls+1.)   # L of the source degree, from uᵣ = L P/r
            out = Ls * ( hdr.coef('P2', l, m, offdiag) * h1tp20_D0 + hdr.coef('P4', l, m, offdiag) * h1tp40_D0 ) \
                + hdr.coef('dP2D', l, m, offdiag) * ( h2tw20_D0 + h1tw20_D1 + h1tw20lho1_D0 ) \
                + hdr.coef('dP4D', l, m, offdiag) * ( h2tw40_D0 + h1tw40_D1 + h1tw40lho1_D0 )
            if par.thermal_wind == 2:
                out = out + Ls * hdr.coef('P6', l, m, offdiag) * h1tp60_D0 \
                    + hdr.coef('dP6D', l, m, offdiag) * ( h2tw60_D0 + h1tw60_D1 + h1tw60lho1_D0 )
            offd = int(np.where(lp == ls)[0][0] - np.where(lp == l)[0][0])

        elif component == 'utor' and par.thermal_wind and abs(offdiag) < dlmax and offdiag % 2 == 1:   # u.grad S_tw, toroidal part

            out = -1j*m * ( hdr.coef('dP2', l, m, offdiag) * h1tw20_D0 + hdr.coef('dP4', l, m, offdiag) * h1tw40_D0 )
            if par.thermal_wind == 2:
                out = out - 1j*m * hdr.coef('dP6', l, m, offdiag) * h1tw60_D0
            offd = int(np.where(lt == ls)[0][0] - np.where(lp == l)[0][0])

    return [ -out/L**3, offd ]



def thermal_diffusion(l, section, component, offdiag):  # rʰ  ∇⋅(κ p ∇s)

    out = 0
    L = l*(l+1.)
    
    if (section == 'h') and (offdiag == 0) :

        out = - L * h2kps0_D0                   \
              + 2 * h1kps0_D1 +     h0kps1_D1   \
              +     h0kps0_D2 

    return par.ThermaD * out/L**3
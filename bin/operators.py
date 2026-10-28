import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
from parameters import par
import utils as ut

import diff_rot as dr


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
        The force of the background flow U0 = Gaspard ΔΩ r sinθ e_φ on the perturbation (diff_rot.momentum_block): rows of
        section 'u' or 'v', column 'upol' or 'utor', source degree l + offdiag. Returns [matrix, offd] (offd in units of N1).
    """
    return dr.momentum_block(l, section, component, offdiag, globals())


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
        p0 [i m Gaspard ΔΩ s + u_r ∂ᵣS_tw + (u_θ/r) ∂_θS_tw] in the heat rows (diff_rot.heat_block): column 'temp' (s),
        'upol' (P) or 'utor' (T), source degree l + offdiag. Returns [matrix, offd]; the matrix already has the sign of the
        A-matrix right-hand side (the heat rows of A hold -(left-hand-side terms)), so add it with +.
    """
    return dr.heat_block(l, component, offdiag, globals())



def thermal_diffusion(l, section, component, offdiag):  # rʰ  ∇⋅(κ p ∇s)

    out = 0
    L = l*(l+1.)
    
    if (section == 'h') and (offdiag == 0) :

        out = - L * h2kps0_D0                   \
              + 2 * h1kps0_D1 +     h0kps1_D1   \
              +     h0kps0_D2 

    return par.ThermaD * out/L**3


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------- NEW (2026-10-28): Lorentz force and induction equation
# ----------------------------------------------------------------------------------------------------------------------
# Ported formula by formula from the conductive_IC branch (bin/operators.py at 39487bbf: lorentz, b, induction,
# magnetic_diffusion), dipolar B0 (degree 1) only. conductive_IC does not divide its rows by L; here every operator is
# divided by L³ of the row, as all the operators above, so each block is the conductive_IC block over L³ with its labels
# renamed (submatrices.py lists the correspondence). b = ∇×∇×(f r) + ∇×(g r) carries no ρ; rows: r⁵ r̂⋅∇×∇× and
# r³ r̂⋅∇× of the momentum equation (u, v), r² r̂⋅ and r² r̂⋅∇× of the induction equation (f, g).
# These are the Boussinesq forms: with a variable ρ0 the Lorentz force in the u rows and the induction by P need
# (ln ρ0)' terms that are not implemented, and background.check() stops unless structure = 'uniform'.
# Column blocks: the l's of f are those of the toroidal flow (symmB0 = -1), of g those of the poloidal flow, so offd
# follows ut.symm1 as for the Coriolis force.



def lorentz(l, section, component, offdiag):  # --------------------------- Lorentz force (∇×b)×B0 + (∇×B0)×b, per unit mass

    out  = 0
    offd = 0
    m = par.m
    L = l*(l+1.)

    if section == 'u':  # ------------------------------------------------------- 2curl

        if component == 'bpol':

            if offdiag == -1:  # (l-1) terms

                C = np.sqrt(l**2-m**2)*(l**2-1)/(2*l-1)
                out1 = -2*(l**2+2)*u4bpo0ohr1_D1 - 2*(l-2)*u3bpo1ohr1_D1 - (l-4)*u3bpo0ohr1_D2 - (l-2)*u2bpo1ohr1_D2
                out2 = L*(l+2)*u5bpo0ohr1_D0 + L*(l-4)*u4bpo1ohr1_D0 + l*u3bpo2ohr1_D0 + l*u2bpo3ohr1_D0 + 2*u2bpo0ohr1_D3
                out = C*(out1+out2)

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 1:  # (l+1) terms

                C = np.sqrt((1+l+m)*(1+l-m))*l*(l+2)/(2*l+3)
                out1 = -2*(l**2+2*l+3)*u4bpo0ohr1_D1 + 2*(l+3)*u3bpo1ohr1_D1 + (l+5)*u3bpo0ohr1_D2 + (l+3)*u2bpo1ohr1_D2
                out2 = -L*(l-1)*u5bpo0ohr1_D0 - L*(l+5)*u4bpo1ohr1_D0 - (l+1)*u3bpo2ohr1_D0 - (l+1)*u2bpo3ohr1_D0 + 2*u2bpo0ohr1_D3
                out = C*(out1+out2)

                if ut.symm1 == -1:
                    offd = 1

        elif component == 'btor':

            if offdiag == 0:  # l terms

                out = 2j*m*( -u4bpo0ohr1_D0 - (l**2+l-1)*u3bpo1ohr1_D0 + u3bpo0ohr1_D1 + u2bpo1ohr1_D1 + u2bpo0ohr1_D2 )

    elif section == 'v':  # ------------------------------------------------------- 1curl

        if component == 'bpol':

            if offdiag == 0:  # l terms

                out = 1j*m*( 4*v3bpo0ohr1_D1 - L*( 2*v3bpo1ohr1_D0 + v2bpo2ohr1_D0 ) + 2*v2bpo0ohr1_D2 )

        elif component == 'btor':

            if offdiag == -1:  # (l-1) terms

                C = np.sqrt((l-m)*(l+m))*(l**2-1)/(2*l-1)
                out = C*( (l-2)*v3bpo0ohr1_D0 + l*v2bpo1ohr1_D0 - 2*v2bpo0ohr1_D1 )

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:  # (l+1) terms

                C = -np.sqrt((l+m+1)*(l+1-m))*l*(l+2)/(2*l+3)
                out = C*( (l+3)*v3bpo0ohr1_D0 + (l+1)*v2bpo1ohr1_D0 + 2*v2bpo0ohr1_D1 )

                if ut.symm1 == 1:
                    offd = 1

    return [ par.Hendrik * out/L**3, offd ]



def b(l, section, component, offdiag):  # ------------------------------------------------- the magnetic field, for ∂b/∂t

    out = 0
    L = l*(l+1.)

    if offdiag == 0:

        if section == 'f' and component == 'bpol':    # r² r̂⋅b

            out = L*f0_D0

        elif section == 'g' and component == 'btor':  # r² r̂⋅∇×b

            out = L*g0_D0

        elif section == 'fic' and component == 'bpol_ic':  # NEW (2026-10-28): conducting inner core, r² r̂⋅b

            out = L*r2_D0f_ic

        elif section == 'gic' and component == 'btor_ic':  # r² r̂⋅∇×b

            out = L*r2_D0g_ic

    return out/L**3



def induction(l, section, component, offdiag):  # ----------------------------------------------------- induction ∇×(u×B0)

    out  = 0
    offd = 0
    m = par.m
    l = np.longdouble(l)  # as conductive_IC (overflow at high l)
    L = l*(l+1)

    if section == 'f':  # ------------------------------------------------------- r² r̂⋅∇×(u×B0)

        if component == 'upol':

            if offdiag == -1:  # l-1 terms

                C = np.sqrt(l**2-m**2)*(l**2-1)/(2*l-1)
                out = C*( (l-2)*f2bpo0_D0 + l*f1bpo1_D0 - 2*f1bpo0_D1 )

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:  # l+1 terms

                C = np.sqrt((l+1)**2-m**2)*l*(l+2)/(2*l+3)
                out = C*( -(l+3)*f2bpo0_D0 - (l+1)*f1bpo1_D0 - 2*f1bpo0_D1 )

                if ut.symm1 == 1:
                    offd = 1

        elif component == 'utor':

            if offdiag == 0:  # l terms

                out = -2j*m*f1bpo0_D0

    elif section == 'g':  # ------------------------------------------------------- r² r̂⋅∇×∇×(u×B0)

        if component == 'upol':

            if offdiag == 0:  # l terms

                out = 2j*m*( g2bpo0_D1 + g1bpo1_D1 - (l**2+l+1)*g3bpo0_D0 + g2bpo1_D0 + (L/2)*g1bpo2_D0 + g1bpo0_D2 )

        elif component == 'utor':

            if offdiag == -1:  # l-1 terms

                C  = (2*l+1)*np.sqrt( l*(l**2-1)*(l**2-m**2)/(4*l**2-1) )
                C1 = np.sqrt( (l**2-1)/(4*l**3-l) )
                C2 = np.sqrt( l*(l**2-1)/(4*l**2-1) )
                out = C*( C2*g2bpo0_D0 - 2*C1*g1bpo0_D1 + (C2-2*C1)*g1bpo1_D0 )

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 1:  # l+1 terms

                C = l*(l+2)*np.sqrt(1+2*l+l**2-m**2)/(3+2*l)
                out = C*( -2*g1bpo0_D1 - (l+1)*g2bpo0_D0 - (l+3)*g1bpo1_D0 )

                if ut.symm1 == -1:
                    offd = 1

    return [ out/L**3, offd ]



def magnetic_diffusion(l, section, component, offdiag):  # -------------------------------------------- −MagnetD ∇×(η∇×b)

    out = 0
    L = l*(l+1.)

    if offdiag == 0:

        if section == 'f' and component == 'bpol':    # -r² r̂⋅∇×(η∇×b)

            out = L*( -L*f2eta0_D0 + 2*f1eta0_D1 + f0eta0_D2 )

        elif section == 'g' and component == 'btor':  # -r² r̂⋅∇×∇×(η∇×b)

            out = L*( -L*g2eta0_D0 + 2*g1eta0_D1 + g0eta0_D2 + g1eta1_D0 + g0eta1_D1 )

        elif section == 'fic' and component == 'bpol_ic':  # NEW (2026-10-28): conducting inner core, uniform diffusivity

            out = L*( -L*r0_D0f_ic + 2*r1_D1f_ic + r2_D2f_ic ) / ( par.mu_i2o * par.sigma_i2o )

        elif section == 'gic' and component == 'btor_ic':

            out = L*( -L*r0_D0g_ic + 2*r1_D1g_ic + r2_D2g_ic ) / ( par.mu_i2o * par.sigma_i2o )

    return par.MagnetD * out/L**3

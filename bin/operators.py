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
# With a variable ρ0 (NEW 2026-10-28, step 3; derived independently in tests/magnetic/derive.py), λ1 = (ln ρ0)', λ2 = (ln ρ0)'':
#   rows f: u's consoidal part is P' + P/r + λ1 P, so every d/dr on P in the induction rows f becomes d/dr + λ1;
#   rows g: the extra 2im[ r h λ1 P' + r h λ2 P + (1 - L/2) r h' λ1 P - (L/2) h λ1 P ] (diagonal in l for B0 of degree 1);
#   rows u: ρ0 r̂⋅∇×∇×(ρ0⁻¹𝐆) = r̂⋅∇×∇×𝐆 - (λ1/r) ∇₁⋅𝐆_H, the extra term λ1 r⁴ (∇₁⋅𝐆_H)_l (conductive_IC scale);
#   rows v, the T columns, the diffusion, the walls and the inner core have no ρ0 terms.
# The labels with lho vanish exactly for structure 'Boussinesq', so its matrices are those of the Boussinesq forms.
# Column blocks: the l's of f are those of the toroidal flow (symmB0 = -1), of g those of the poloidal flow, so offd
# follows ut.symm1 as for the Coriolis force.



def lorentz_pol(l, section, component, offdiag):  # ----------------------- Lorentz force (∇×b)×B0 + (∇×B0)×b, per unit mass, poloidal B0

    if ut.bf.B0_l == 2:   # NEW (2026-10-28)
        return lorentz_pol2(l, section, component, offdiag)
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
                out = C*(out1+out2) + lorentz_rho(l, l-1, np.sqrt(l**2-m**2)/(2*l-1))  # NEW (2026-10-28)

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 1:  # (l+1) terms

                C = np.sqrt((1+l+m)*(1+l-m))*l*(l+2)/(2*l+3)
                out1 = -2*(l**2+2*l+3)*u4bpo0ohr1_D1 + 2*(l+3)*u3bpo1ohr1_D1 + (l+5)*u3bpo0ohr1_D2 + (l+3)*u2bpo1ohr1_D2
                out2 = -L*(l-1)*u5bpo0ohr1_D0 - L*(l+5)*u4bpo1ohr1_D0 - (l+1)*u3bpo2ohr1_D0 - (l+1)*u2bpo3ohr1_D0 + 2*u2bpo0ohr1_D3
                out = C*(out1+out2) + lorentz_rho(l, l+1, np.sqrt((l+1)**2-m**2)/(2*l+3))  # NEW (2026-10-28)

                if ut.symm1 == -1:
                    offd = 1

        elif component == 'btor':

            if offdiag == 0:  # l terms

                out = 2j*m*( -u4bpo0ohr1_D0 - (l**2+l-1)*u3bpo1ohr1_D0 + u3bpo0ohr1_D1 + u2bpo1ohr1_D1 + u2bpo0ohr1_D2 )
                # NEW (2026-10-28): variable ρ0, λ1 r⁴ (∇₁⋅𝐆_H)_l of the toroidal field: -i m λ1 [(L+2) r² h g + L r³ h' g + 2 r³ h g']
                out += -1j*m*( (L+2)*u3bpo0lho1ohr1_D0 + L*u2bpo1lho1ohr1_D0 + 2*u2bpo0lho1ohr1_D1 )

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



def lorentz_rho(l, lc, y):
    '''
    NEW (2026-10-28): the variable-ρ0 part of the Lorentz force of the poloidal field f of degree lc in the rows u of degree l
    (B0 of degree 1), λ1 r⁴ (∇₁⋅𝐆_H)_l in conductive_IC's scale (rows r⁵ r̂⋅∇×∇×, A sign), y = <Y_l|cos θ Y_lc>:
    (y/2) λ1 r³ [ 2a h (f'' + 2f'/r - Lc f/r²) + Lc b (h'' + 2h'/r - 2h/r²) f ],  a = 2 - Lc - L,  b = Lc - 2 - L
    '''
    L, Lc = l*(l+1.), lc*(lc+1.)
    a, b = 2 - Lc - L, Lc - 2 - L
    return (y/2)*( 2*a*u2bpo0lho1ohr1_D2 + 4*a*u3bpo0lho1ohr1_D1 - 2*a*Lc*u4bpo0lho1ohr1_D0
                   + Lc*b*u2bpo2lho1ohr1_D0 + 2*Lc*b*u3bpo1lho1ohr1_D0 - 2*Lc*b*u4bpo0lho1ohr1_D0 )



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



def induction_pol(l, section, component, offdiag):  # --------------------------------------- induction ∇×(u×B0), poloidal B0

    if ut.bf.B0_l == 2:   # NEW (2026-10-28)
        return induction_pol2(l, section, component, offdiag)
    out  = 0
    offd = 0
    m = par.m
    l = np.longdouble(l)  # as conductive_IC (overflow at high l)
    L = l*(l+1)

    if section == 'f':  # ------------------------------------------------------- r² r̂⋅∇×(u×B0)

        if component == 'upol':

            if offdiag == -1:  # l-1 terms

                C = np.sqrt(l**2-m**2)*(l**2-1)/(2*l-1)
                out = C*( (l-2)*f2bpo0_D0 + l*f1bpo1_D0 - 2*f1bpo0_D1 - 2*f1bpo0lho1_D0 )  # NEW (2026-10-28): + λ1 P

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:  # l+1 terms

                C = np.sqrt((l+1)**2-m**2)*l*(l+2)/(2*l+3)
                out = C*( -(l+3)*f2bpo0_D0 - (l+1)*f1bpo1_D0 - 2*f1bpo0_D1 - 2*f1bpo0lho1_D0 )  # NEW (2026-10-28): + λ1 P

                if ut.symm1 == 1:
                    offd = 1

        elif component == 'utor':

            if offdiag == 0:  # l terms

                out = -2j*m*f1bpo0_D0

    elif section == 'g':  # ------------------------------------------------------- r² r̂⋅∇×∇×(u×B0)

        if component == 'upol':

            if offdiag == 0:  # l terms

                out = 2j*m*( g2bpo0_D1 + g1bpo1_D1 - (l**2+l+1)*g3bpo0_D0 + g2bpo1_D0 + (L/2)*g1bpo2_D0 + g1bpo0_D2 )
                # NEW (2026-10-28): variable ρ0, from ∇⋅u = -λ1 u_r and the λ1 P in u's consoidal part
                out += 2j*m*( g1bpo0lho1_D1 + g1bpo0lho2_D0 + (1 - L/2)*g1bpo1lho1_D0 - (L/2)*g2bpo0lho1_D0 )

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




# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------- NEW (2026-10-28): toroidal B0, and the totals
# ----------------------------------------------------------------------------------------------------------------------
# B0_tor = ∇×(k Y_1⁰ 𝐫), B_φ = k sinθ (bto = cnorm_tor k): the rows derived with tests/magnetic/derive.py (any ρ0) and fitted in l,
# each a closed form A(l, m) R(l) with A = i m (same degree) or the Coriolis-type factors C₋ = √(l² - m²)/(2l - 1) (column l - 1),
# C₊ = √((l+1)² - m²)/(2l + 3) (column l + 1); tests/magnetic/derive_check.py checks them against the derivation. Its
# couplings are disjoint from those of the poloidal dipole: f, g rows from P at l ± 1 and from T at l only through g;
# u rows from F at l and G at l ± 1; v rows from F at l ± 1 and G at l. The field is equatorially symmetric (symmB0 = 1).


def moff(row, col, dl):
    '''block offset (units of N1) of the column block of degree l + dl for the rows of section row ('u', 'v', 'f', 'g') and the
    unknown col ('P', 'T', 'F', 'G'): as for the Coriolis force, from whether the two use the l's of P ('top') or of T'''
    top = {'u': True, 'v': False, 'P': True, 'T': False, 'f': ut.symmB0 == 1, 'g': ut.symmB0 != 1, 'F': ut.symmB0 == 1, 'G': ut.symmB0 != 1}
    if top[row] == top[col]:
        return int(np.trunc(dl/2))   # Δl = ±2 is one block (Δl = 0: none)
    if top[row]:
        return -1 if (dl == -1 and ut.symm1 == 1) else (1 if (dl == 1 and ut.symm1 == -1) else 0)
    return -1 if (dl == -1 and ut.symm1 == -1) else (1 if (dl == 1 and ut.symm1 == 1) else 0)


def quad_factors(l, m):
    '''NEW (2026-10-28): the angular factors of a degree-2 background field, by Δl = -2..2 (as in conductive_IC's quadrupole terms)'''
    return { 0: 1/((2*l - 1)*(2*l + 3)),
            -1: 1j*m*np.sqrt(l**2 - m**2)/(2*l - 1),
             1: 1j*m*np.sqrt((l+1)**2 - m**2)/(2*l + 3),
            -2: np.sqrt((l**2 - m**2)*((l-1)**2 - m**2))/((2*l - 1)*(2*l - 3)),
             2: np.sqrt(((l+1)**2 - m**2)*((l+2)**2 - m**2))/((2*l + 3)*(2*l + 5)) }


def induction_pol2(l, section, component, offdiag):  # ------------------------- induction ∇×(u×B0), poloidal B0 of degree 2
    '''NEW (2026-10-28): B0 = ∇×∇×(h Y_2⁰ 𝐫) (bpo = cnorm h; 'Luo_S2'), the rows of derive.py fitted in l and m (quad_factors times
    polynomials in l; the same-degree terms carry l(l+1) - 3m²; tests/magnetic/derive_check.py checks them). The f, g rows couple
    to P at l, l ± 2 and to T at l ± 1 (f) or l, l ± 2 (g).'''
    m = par.m
    L = l*(l+1.)
    col = 'P' if component == 'upol' else 'T'
    rows = {
        ('f', 'P', -2): lambda: (-3*(l - 4)*(l - 2)*(l + 1))*f2bpo0_D0 + (9*(l - 2)*(l + 1))*f1bpo0lho1_D0
                                + (-3*(l - 2)*(l - 1)*(l + 1))*f1bpo1_D0 + (9*(l - 2)*(l + 1))*f1bpo0_D1,
        ('f', 'P', 0): lambda: (3*(l - 2)*(l + 3)*(L - 3*m**2))*f2bpo0_D0 + (6*(L - 3)*(L - 3*m**2))*f1bpo0lho1_D0
                                + (-3*l*(l + 1)*(L - 3*m**2))*f1bpo1_D0 + (6*(L - 3)*(L - 3*m**2))*f1bpo0_D1,
        ('f', 'P', 2): lambda: (3*l*(l + 3)*(l + 5))*f2bpo0_D0 + (9*l*(l + 3))*f1bpo0lho1_D0 + (3*l*(l + 2)*(l + 3))*f1bpo1_D0
                                + (9*l*(l + 3))*f1bpo0_D1,
        ('f', 'T', -1): lambda: (18)*f1bpo0_D0,
        ('f', 'T', 1): lambda: (18)*f1bpo0_D0,
        ('g', 'P', -1): lambda: (6*(l**2 + 3))*g3bpo0_D0 + (3*l*(l + 1))*g2bpo0lho1_D0 + (-18)*g1bpo0lho2_D0 + (6*(l - 3))*g2bpo1_D0
                                + (3*(l - 2)*(l + 3))*g1bpo1lho1_D0 + (-3*l*(l - 1))*g1bpo2_D0 + (6*(l - 3))*g2bpo0_D1
                                + (-18)*g1bpo0lho1_D1 + (6*(l - 3))*g1bpo1_D1 + (-18)*g1bpo0_D2,
        ('g', 'P', 1): lambda: (6*(l**2 + 2*l + 4))*g3bpo0_D0 + (3*l*(l + 1))*g2bpo0lho1_D0 + (-18)*g1bpo0lho2_D0
                                + (-6*(l + 4))*g2bpo1_D0 + (3*(l - 2)*(l + 3))*g1bpo1lho1_D0 + (-3*(l + 1)*(l + 2))*g1bpo2_D0
                                + (-6*(l + 4))*g2bpo0_D1 + (-18)*g1bpo0lho1_D1 + (-6*(l + 4))*g1bpo1_D1 + (-18)*g1bpo0_D2,
        ('g', 'T', -2): lambda: (-3*l*(l - 2)*(l + 1))*g2bpo0_D0 + (-3*(l - 3)*(l - 2)*(l + 1))*g1bpo1_D0
                                + (9*(l - 2)*(l + 1))*g1bpo0_D1,
        ('g', 'T', 0): lambda: (3*l*(l + 1)*(L - 3*m**2))*g2bpo0_D0 + (9*(l - 1)*(l + 2)*(L - 3*m**2))*g1bpo1_D0
                                + (6*(L - 3)*(L - 3*m**2))*g1bpo0_D1,
        ('g', 'T', 2): lambda: (3*l*(l + 1)*(l + 3))*g2bpo0_D0 + (3*l*(l + 3)*(l + 4))*g1bpo1_D0 + (9*l*(l + 3))*g1bpo0_D1,
    }
    f = rows.get((section, col, offdiag))
    if f is None:
        return [ 0, 0 ]
    return [ -quad_factors(l, m)[offdiag]*f()/L**3, moff(section, col, offdiag) ]


def lorentz_pol2(l, section, component, offdiag):  # ------------ Lorentz force (∇×b)×B0 + (∇×B0)×b per unit mass, poloidal B0 of degree 2
    '''NEW (2026-10-28): as induction_pol2. The u rows couple to F at l, l ± 2 and to G at l ± 1; the v rows to F at l ± 1 and to G
    at l, l ± 2.'''
    m = par.m
    L = l*(l+1.)
    col = 'F' if component == 'bpol' else 'G'
    rows = {
        ('u', 'F', -2): lambda: (-3*l*(l - 2)*(l + 1)**2*(l + 2))*u5bpo0ohr1_D0 + (-9*l*(l - 2)*(l - 1)*(l + 1))*u4bpo0lho1ohr1_D0
                                + (-3*(l - 2)*(l + 1)*(l**3 - 6*l**2 + l + 2))*u4bpo1ohr1_D0
                                + (6*(l - 2)*(l - 1)*(l + 1))*u3bpo1lho1ohr1_D0 + (-3*(l - 2)**2*(l + 1))*u3bpo2ohr1_D0
                                + (3*(l - 2)*(l - 1)*(l + 1))*u2bpo2lho1ohr1_D0 + (-3*(l - 2)*(l - 1)*(l + 1))*u2bpo3ohr1_D0
                                + (3*(l - 2)*(l + 1)*(3*l**2 - 7*l + 6))*u4bpo0ohr1_D1 + (18*(l - 2)*(l + 1))*u3bpo0lho1ohr1_D1
                                + (6*(l - 2)**2*(l + 1))*u3bpo1ohr1_D1 + (3*(l - 2)*(l + 1))*u2bpo2ohr1_D1
                                + (3*(l - 6)*(l - 2)*(l + 1))*u3bpo0ohr1_D2 + (9*(l - 2)*(l + 1))*u2bpo0lho1ohr1_D2
                                + (3*(l - 3)*(l - 2)*(l + 1))*u2bpo1ohr1_D2 + (-9*(l - 2)*(l + 1))*u2bpo0ohr1_D3,
        ('u', 'F', 0): lambda: (-9*l*(l - 1)*(l + 1)*(l + 2)*(L - 3*m**2))*u5bpo0ohr1_D0
                                + (-6*l**2*(l + 1)**2*(L - 3*m**2))*u4bpo0lho1ohr1_D0 + (9*l**2*(l + 1)**2*(L - 3*m**2))*u4bpo1ohr1_D0
                                + (6*l*(l + 1)*(L - 3*m**2))*u3bpo1lho1ohr1_D0 + (-9*l*(l + 1)*(L - 3*m**2))*u3bpo2ohr1_D0
                                + (3*l*(l + 1)*(L - 3*m**2))*u2bpo2lho1ohr1_D0 + (-3*l*(l + 1)*(L - 3*m**2))*u2bpo3ohr1_D0
                                + (6*(L - 3*m**2)*(l**4 + 2*l**3 + 5*l**2 + 4*l - 6))*u4bpo0ohr1_D1
                                + (12*(L - 3)*(L - 3*m**2))*u3bpo0lho1ohr1_D1 + (-6*(L - 3*m**2)*(5*l**2 + 5*l - 6))*u3bpo1ohr1_D1
                                + (-6*l*(l + 1)*(L - 3*m**2))*u2bpo2ohr1_D1 + (-3*(L - 3*m**2)*(5*l**2 + 5*l - 12))*u3bpo0ohr1_D2
                                + (6*(L - 3)*(L - 3*m**2))*u2bpo0lho1ohr1_D2 + (-9*(l - 1)*(l + 2)*(L - 3*m**2))*u2bpo1ohr1_D2
                                + (-6*(L - 3)*(L - 3*m**2))*u2bpo0ohr1_D3,
        ('u', 'F', 2): lambda: (3*l**2*(l - 1)*(l + 1)*(l + 3))*u5bpo0ohr1_D0 + (-9*l*(l + 1)*(l + 2)*(l + 3))*u4bpo0lho1ohr1_D0
                                + (3*l*(l + 3)*(l**3 + 9*l**2 + 16*l + 6))*u4bpo1ohr1_D0 + (-6*l*(l + 2)*(l + 3))*u3bpo1lho1ohr1_D0
                                + (3*l*(l + 3)**2)*u3bpo2ohr1_D0 + (-3*l*(l + 2)*(l + 3))*u2bpo2lho1ohr1_D0
                                + (3*l*(l + 2)*(l + 3))*u2bpo3ohr1_D0 + (3*l*(l + 3)*(3*l**2 + 13*l + 16))*u4bpo0ohr1_D1
                                + (18*l*(l + 3))*u3bpo0lho1ohr1_D1 + (-6*l*(l + 3)**2)*u3bpo1ohr1_D1 + (3*l*(l + 3))*u2bpo2ohr1_D1
                                + (-3*l*(l + 3)*(l + 7))*u3bpo0ohr1_D2 + (9*l*(l + 3))*u2bpo0lho1ohr1_D2
                                + (-3*l*(l + 3)*(l + 4))*u2bpo1ohr1_D2 + (-9*l*(l + 3))*u2bpo0ohr1_D3,
        ('u', 'G', -1): lambda: (-6*(2*l**2 + 3*l - 3))*u4bpo0ohr1_D0 + (3*(l**2 - l + 6))*u3bpo0lho1ohr1_D0
                                + (6*(l - 1)*(l + 3))*u3bpo1ohr1_D0 + (3*l*(l - 1))*u2bpo1lho1ohr1_D0 + (6*l)*u2bpo2ohr1_D0
                                + (6*(l - 3))*u3bpo0ohr1_D1 + (18)*u2bpo0lho1ohr1_D1 + (6*(l - 3))*u2bpo1ohr1_D1 + (-18)*u2bpo0ohr1_D2,
        ('u', 'G', 1): lambda: (-6*(2*l**2 + l - 4))*u4bpo0ohr1_D0 + (3*(l**2 + 3*l + 8))*u3bpo0lho1ohr1_D0
                                + (6*(l - 2)*(l + 2))*u3bpo1ohr1_D0 + (3*(l + 1)*(l + 2))*u2bpo1lho1ohr1_D0 + (-6*(l + 1))*u2bpo2ohr1_D0
                                + (-6*(l + 4))*u3bpo0ohr1_D1 + (18)*u2bpo0lho1ohr1_D1 + (-6*(l + 4))*u2bpo1ohr1_D1 + (-18)*u2bpo0ohr1_D2,
        ('v', 'F', -1): lambda: (6*l*(l - 1))*v3bpo1ohr1_D0 + (3*l*(l - 1))*v2bpo2ohr1_D0 + (-36)*v3bpo0ohr1_D1 + (-18)*v2bpo0ohr1_D2,
        ('v', 'F', 1): lambda: (6*(l + 1)*(l + 2))*v3bpo1ohr1_D0 + (3*(l + 1)*(l + 2))*v2bpo2ohr1_D0 + (-36)*v3bpo0ohr1_D1
                                + (-18)*v2bpo0ohr1_D2,
        ('v', 'G', -2): lambda: (-3*(l - 4)*(l - 2)*(l + 1))*v3bpo0ohr1_D0 + (-3*(l - 2)*(l - 1)*(l + 1))*v2bpo1ohr1_D0
                                + (9*(l - 2)*(l + 1))*v2bpo0ohr1_D1,
        ('v', 'G', 0): lambda: (3*(l - 2)*(l + 3)*(L - 3*m**2))*v3bpo0ohr1_D0 + (-3*l*(l + 1)*(L - 3*m**2))*v2bpo1ohr1_D0
                                + (6*(L - 3)*(L - 3*m**2))*v2bpo0ohr1_D1,
        ('v', 'G', 2): lambda: (3*l*(l + 3)*(l + 5))*v3bpo0ohr1_D0 + (3*l*(l + 2)*(l + 3))*v2bpo1ohr1_D0 + (9*l*(l + 3))*v2bpo0ohr1_D1,
    }
    f = rows.get((section, col, offdiag))
    if f is None:
        return [ 0, 0 ]
    return [ -par.Hendrik*quad_factors(l, m)[offdiag]*f()/L**3, moff(section, col, offdiag) ]


def induction_tor2(l, section, component, offdiag):  # ------------------------- induction ∇×(u×B0), toroidal B0 of degree 2
    '''NEW (2026-10-28): B_φ = -k ∂θY_2⁰ (bto = cnorm_tor k), the rows of derive.py fitted in l and m (quad_factors times
    polynomials; the same-degree terms carry l(l+1) - 3m²)'''
    out = 0
    m = par.m
    L = l*(l+1.)
    A = quad_factors(l, m).get(offdiag, 0)
    col = 'P' if component == 'upol' else 'T'
    g0 = g2bto0_D0 + g1bto0lho1_D0
    if section == 'f' and component == 'upol' and offdiag in (-1, 1):
        out = A*3*(l*(l-1) if offdiag == -1 else (l+1)*(l+2))*f1bto0_D0
    elif section == 'g' and component == 'upol':
        if offdiag == -2:
            out = A*( 3*l*(l-2)*(l+1)*g0 - 3*(l-2)*(l-1)*(l+1)*g1bto1_D0 + 3*(l-2)*(l+1)*g1bto0_D1 )
        elif offdiag == 0:
            out = A*(L - 3*m**2)*( -3*L*g0 - 3*L*g1bto1_D0 - 6*L*g1bto0_D1 )
        elif offdiag == 2:
            out = A*( -3*l*(l+1)*(l+3)*g0 + 3*l*(l+2)*(l+3)*g1bto1_D0 + 3*l*(l+3)*g1bto0_D1 )
    elif section == 'g' and component == 'utor' and offdiag in (-1, 1):
        out = A*3*L*g1bto0_D0
    return [ -out/L**3, moff(section, col, offdiag) ]


def lorentz_tor2(l, section, component, offdiag):  # ------------ Lorentz force (∇×b)×B0 + (∇×B0)×b per unit mass, toroidal B0 of degree 2
    '''NEW (2026-10-28): as induction_tor2'''
    out = 0
    m = par.m
    L = l*(l+1.)
    A = quad_factors(l, m).get(offdiag, 0)
    col = 'F' if component == 'bpol' else 'G'
    if section == 'u' and component == 'bpol' and offdiag in (-1, 1):
        if offdiag == -1:
            c = [ 3*(l**4 - 3*l**2 - 6), -3*(l**2 - l + 6), -6*(l-3), -3*l*(l-1), 3*l*(l-1), -6*(l-1)*(l+3), -18, -6*(l-3), -3*(l-2)*(l+3) ]
        else:
            c = [ 3*(l**4 + 4*l**3 + 3*l**2 - 2*l - 8), -3*(l**2 + 3*l + 8), 6*(l+4), -3*(l+1)*(l+2), 3*(l+1)*(l+2), -6*(l-2)*(l+2), -18,
                  6*(l+4), -3*(l-2)*(l+3) ]
        out = A*( c[0]*u4bto0ohr1_D0 + c[1]*u3bto0lho1ohr1_D0 + c[2]*u3bto1ohr1_D0 + c[3]*u2bto1lho1ohr1_D0 + c[4]*u2bto2ohr1_D0
                  + c[5]*u3bto0ohr1_D1 + c[6]*u2bto0lho1ohr1_D1 + c[7]*u2bto1ohr1_D1 + c[8]*u2bto0ohr1_D2 )
    elif section == 'u' and component == 'btor' and offdiag in (-2, 0, 2):
        c = { -2: [ 6*l*(l-2)*(l+1), 3*(l-2)*(l+1)*(l+2), -6*(l-2)*(l+1), -6*(l-2)*(l+1) ],
               0: [ -6*L*(L - 3*m**2), 9*(L-2)*(L - 3*m**2), -6*(2*L-3)*(L - 3*m**2), -6*(2*L-3)*(L - 3*m**2) ],
               2: [ -6*l*(l+1)*(l+3), -3*l*(l-1)*(l+3), -6*l*(l+3), -6*l*(l+3) ] }[offdiag]
        out = A*( c[0]*u3bto0ohr1_D0 + c[1]*u2bto0lho1ohr1_D0 + c[2]*u2bto1ohr1_D0 + c[3]*u2bto0ohr1_D1 )
    elif section == 'v' and component == 'bpol' and offdiag in (-2, 0, 2):
        c = { -2: [ 3*(l-4)*(l-2)*(l+1), 3*(l-2)*(l-1)*(l+1), -9*(l-2)*(l+1) ],
               0: [ -3*(L-6)*(L - 3*m**2), 3*L*(L - 3*m**2), -6*(L-3)*(L - 3*m**2) ],
               2: [ -3*l*(l+3)*(l+5), -3*l*(l+2)*(l+3), -9*l*(l+3) ] }[offdiag]
        out = A*( c[0]*v3bto0ohr1_D0 + c[1]*v2bto1ohr1_D0 + c[2]*v2bto0ohr1_D1 )
    elif section == 'v' and component == 'btor' and offdiag in (-1, 1):
        out = A*3*((l-3)*(l+2) if offdiag == -1 else (l-1)*(l+4))*v2bto0ohr1_D0
    return [ -par.Hendrik * out/L**3, moff(section, col, offdiag) ]


def induction_tor(l, section, component, offdiag):  # --------------------------------------- induction ∇×(u×B0), toroidal B0

    if par.B0_tor_l == 2:   # NEW (2026-10-28)
        return induction_tor2(l, section, component, offdiag)
    out = 0
    m = par.m
    L = l*(l+1.)
    Cm = np.sqrt(l**2 - m**2)/(2*l - 1)
    Cp = np.sqrt((l+1)**2 - m**2)/(2*l + 3)
    col = 'P' if component == 'upol' else 'T'

    if section == 'f' and component == 'upol' and offdiag == 0:
        out = -1j*m*L*f1bto0_D0
    elif section == 'g' and component == 'upol' and offdiag == -1:
        out = -Cm*l*(l-1)*(l+1)*( g2bto0_D0 + g1bto0lho1_D0 - g1bto1_D0 )
    elif section == 'g' and component == 'upol' and offdiag == 1:
        out = Cp*l*(l+1)*(l+2)*( g2bto0_D0 + g1bto0lho1_D0 - g1bto1_D0 )
    elif section == 'g' and component == 'utor' and offdiag == 0:
        out = -1j*m*L*g1bto0_D0

    return [ out/L**3, moff(section, col, offdiag) ]


def lorentz_tor(l, section, component, offdiag):  # ------------------- Lorentz force (∇×b)×B0 + (∇×B0)×b per unit mass, toroidal B0

    if par.B0_tor_l == 2:   # NEW (2026-10-28)
        return lorentz_tor2(l, section, component, offdiag)
    out = 0
    m = par.m
    L = l*(l+1.)
    Cm = np.sqrt(l**2 - m**2)/(2*l - 1)
    Cp = np.sqrt((l+1)**2 - m**2)/(2*l + 3)
    col = 'F' if component == 'bpol' else 'G'

    if section == 'u' and component == 'bpol' and offdiag == 0:
        out = -1j*m*( (L**2 - 2*L - 2)*u4bto0ohr1_D0 - (L+2)*u3bto0lho1ohr1_D0 + 2*u3bto1ohr1_D0 - 2*(L-1)*u3bto0ohr1_D1
                      - L*u2bto1lho1ohr1_D0 + L*u2bto2ohr1_D0 - 2*u2bto0lho1ohr1_D1 + 2*u2bto1ohr1_D1 - (L-2)*u2bto0ohr1_D2 )
    elif section == 'u' and component == 'btor' and offdiag == -1:
        out = -Cm*( 2*l*(l**2-1)*u3bto0ohr1_D0 + (l**2-1)*(l+2)*u2bto0lho1ohr1_D0 - 2*(l**2-1)*u2bto1ohr1_D0 - 2*(l**2-1)*u2bto0ohr1_D1 )
    elif section == 'u' and component == 'btor' and offdiag == 1:
        out = Cp*( 2*l*(l+1)*(l+2)*u3bto0ohr1_D0 + l*(l-1)*(l+2)*u2bto0lho1ohr1_D0 + 2*l*(l+2)*u2bto1ohr1_D0 + 2*l*(l+2)*u2bto0ohr1_D1 )
    elif section == 'v' and component == 'bpol' and offdiag == -1:
        out = -Cm*( (l-2)*(l**2-1)*v3bto0ohr1_D0 + l*(l**2-1)*v2bto1ohr1_D0 - 2*(l**2-1)*v2bto0ohr1_D1 )
    elif section == 'v' and component == 'bpol' and offdiag == 1:
        out = Cp*( l*(l+2)*(l+3)*v3bto0ohr1_D0 + l*(l+1)*(l+2)*v2bto1ohr1_D0 + 2*l*(l+2)*v2bto0ohr1_D1 )
    elif section == 'v' and component == 'btor' and offdiag == 0:
        out = -1j*m*(L-2)*v2bto0ohr1_D0

    return [ par.Hendrik * out/L**3, moff(section, col, offdiag) ]


def _total(pol, tor, l, section, component, offdiag):
    '''the poloidal and the toroidal B0 parts of a coupling (they never share one): [matrix, offd], matrix 0 if absent'''
    out = pol(l, section, component, offdiag) if par.B0 is not None else [0, 0]
    if par.B0_tor is not None:
        t = tor(l, section, component, offdiag)
        if not np.isscalar(t[0]):
            out = t if np.isscalar(out[0]) else [out[0] + t[0], out[1]]
    return out


def lorentz(l, section, component, offdiag):  # NEW (2026-10-28): poloidal + toroidal B0
    return _total(lorentz_pol, lorentz_tor, l, section, component, offdiag)


def induction(l, section, component, offdiag):  # NEW (2026-10-28): poloidal + toroidal B0
    return _total(induction_pol, induction_tor, l, section, component, offdiag)

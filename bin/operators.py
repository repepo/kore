import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
from parameters import par
import utils as ut

if par.diff_rot : import diff_rot_coefficients as dr


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

    L = l*(l+1)

    if offdiag == 0:

        if section == 'u' and component == 'upol':

            out =  L * (   L*u3_D0 - u2lho1_D0 - u1lho2_D0
                         - 2*u2_D1 - u1lho1_D1
                         - u1_D2 )

        elif section == 'v' and component == 'utor':

            out = L * v1_D0

    return out



def coriolis(l, section, component, offdiag):  # ------------------------------------------------- Coriolis force 2z x u

    out  = 0
    offd = 0
    m = par.m
    L = l*(l+1)

    if section == 'u':  # ------------------------------------------------------- 2curl

        if component == 'upol':

            if offdiag == 0:
                
                out = 1j*m*( - L*u3_D0 + (1-L)*u2lho1_D0 + u1lho2_D0 
                             + 2*u2_D1 + u1lho1_D1
                             + u1_D2 )
                
        elif component == 'utor':

            if offdiag == -1:

                C = (l**2-1)*np.sqrt(l**2-m**2) / (2*l-1.)
                out = C*( (l-1)*u2_D0 - u1_D1 )

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 1:

                C = l*(l+2.)*np.sqrt((l+m+1.)*(l-m+1)) / (2.*l+3.)
                out = C*( -(l+2)*u2_D0 - u1_D1 )             

                if ut.symm1 == -1:
                    offd = 1

    if section == 'v':  # ------------------------------------------------------- 1curl

        if component == 'upol':

            if offdiag == -1:

                C = (l**2-1)*np.sqrt(l**2-m**2) / (2*l-1.)
                out = C*( (l-1)*v2_D0 - v1lho1_D0 - v1_D1 )              

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:

                C = l*(l+2)*np.sqrt((l+m+1.)*(l-m+1)) / (2*l+3)
                out = C*( -(l+2)*v2_D0 - v1lho1_D0 - v1_D1 )            

                if ut.symm1 == 1:
                    offd = 1
                

        elif component == 'utor':

            if offdiag == 0:

                out = -1j * m * v1_D0       


    return [ 2*par.Gaspard * out, offd ]



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

    return [ out, offd ]


def viscous_diffusion(l, section, component, offdiag):  # ------------------------------------------------ viscous force

    out = 0
    L= l*(l+1)

    if (offdiag == 0)&(par.ViscosD>0):

        if section == 'u' and component == 'upol':

            out = L * ( -   u1moe0lho4_D0 - 2*      u1moe1lho3_D0 -         u1moe2lho2_D0 
                        - 3*u2moe0lho3_D0 - 2*      u2moe1lho2_D0 +         u2moe2lho1_D0 
                        + L*u3moe0lho2_D0 + 2*L*    u3moe1lho1_D0 + (2-L)*  u3moe2_D0 
                        + L*u4moe0lho1_D0 - 2*(1+L)*u4moe1_D0     + L*(2-L)*u5moe0_D0

                        - 3*u1moe0lho3_D1 - 4*u1moe1lho2_D1 - u1moe2lho1_D1
                        - 6*u2moe0lho2_D1 - 2*u2moe1lho1_D1 + L*u3moe0lho1_D1
                        + 2*(L+1)*u3moe1_D1

                        - 3*u1moe0lho2_D2 - 2*u1moe1lho1_D2 - u1moe2_D2
                        - 3*u2moe0lho1_D2 - 4*u2moe1_D2 + 2*L*u3moe0_D2
                        
                        - u1moe0lho1_D3 - 2*u1moe1_D3 - 4*u2moe0_D3
                    
                        - u1moe0_D4 )
					
        elif section == 'v' and component == 'utor':

            out = L * ( - v2moe1_D0 - L*v3moe0_D0
            
                        + v1moe1_D1 + 2*v2moe0_D1
            
                        + v1moe0_D2 )

    return par.ViscosD * out



def buoyancy(l, section, component, offdiag): 

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        out = L * u2gra0_D0

    return par.Beyonce * out



def entropy(l, section, component, offdiag):  # rʰ p s

    out = 0
    
    if (section == 'h') and (offdiag == 0) :

        out = h0pss0_D0

    return out



def thermal_advection(l, section, component, offdiag):  # −rʰ p(v⋅∇) S = −rʰ p vᵣ dS/dr

    out = 0
    L = l*(l+1)

    if (section == 'h') and (component == 'upol') and (offdiag == 0) :

        out = -L * h1pdS0_D0
    
    return out



def thermal_diffusion(l, section, component, offdiag):  # rʰ  ∇⋅(κ p ∇s)

    out = 0
    L = l*(l+1)
    
    if (section == 'h') and (offdiag == 0) :

        out = - L * h2kps0_D0                   \
              + 2 * h1kps0_D1 + 2 * h0kps1_D1   \
              +     h0kps0_D2 

    return par.ThermaD * out  
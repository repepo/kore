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
        On the form ΔΩ(r, θ) = ΔΩ*f(r)*g(θ), with : 
                ΔΩ : Differential rotation amplitude (par.diff_rot_amplitude)
                g(θ) : Differential rotation latitudinal profile. For the moment g(θ)=Y20(θ).
                f(r) : Differential rotation radial profile, from which we define : 
                    aub = f
                    svp = r*f'
                    pls = r^2*f''

        The forcing includes : 
            f(r)*{ΔΩ * [g(θ) * (im * u) + g(θ) * (2*ez x u) + g'(θ) * (sin(θ) * uθ * eϕ)]} 
            + r f'(r)*{ΔΩ * g(θ) * (sin(θ) * ur * eϕ)} 
    """


    out  = 0
    offd = 0
    m = par.m
    w = par.diff_rot_amplitude

    if section == 'u':  # ------------------------------------------------------- 2curl

        if component == 'upol':
            
            if offdiag == -2 : 

                out = dr.f_2C_D0P(l, m, w, offdiag)*u2aub0_D0 \
                    + dr.f_2C_D1P(l, m, w, offdiag)*u1aub0_D1 \
                    + dr.f_2C_lho1_D0P(l, m, w, offdiag)*u1aub0lho1_D0 \
                    + dr.f_2C_lho2_D0P(l, m, w, offdiag)*u0aub0lho2_D0 \
                    + dr.f_2C_lho1_D1P(l, m, w, offdiag)*u0aub0lho1_D1 \
                    + dr.f_2C_D2P(l, m, w, offdiag)*u0aub0_D2 \
                    + dr.df_2C_D0P(l, m, w, offdiag)*u2svp0_D0 \
                    + dr.df_2C_D1P(l, m, w, offdiag)*u1svp0_D1 \
                    + dr.df_2C_lho1_D0P(l, m, w, offdiag)*u1svp0lho1_D0 \
                    + dr.d2f_2C_D0P(l, m, w, offdiag)*u2pls0_D0
                
                offd = -1

            elif offdiag == 0:

                out = dr.f_2C_D0P(l, m, w, offdiag)*u2aub0_D0 \
                    + dr.f_2C_D1P(l, m, w, offdiag)*u1aub0_D1 \
                    + dr.f_2C_lho1_D0P(l, m, w, offdiag)*u1aub0lho1_D0 \
                    + dr.f_2C_lho2_D0P(l, m, w, offdiag)*u0aub0lho2_D0 \
                    + dr.f_2C_lho1_D1P(l, m, w, offdiag)*u0aub0lho1_D1 \
                    + dr.f_2C_D2P(l, m, w, offdiag)*u0aub0_D2 \
                    + dr.df_2C_D0P(l, m, w, offdiag)*u2svp0_D0 \
                    + dr.df_2C_D1P(l, m, w, offdiag)*u1svp0_D1 \
                    + dr.df_2C_lho1_D0P(l, m, w, offdiag)*u1svp0lho1_D0 \
                    + dr.d2f_2C_D0P(l, m, w, offdiag)*u2pls0_D0
            
            elif offdiag == 2 : 
                
                out = dr.f_2C_D0P(l, m, w, offdiag)*u2aub0_D0 \
                    + dr.f_2C_D1P(l, m, w, offdiag)*u1aub0_D1 \
                    + dr.f_2C_lho1_D0P(l, m, w, offdiag)*u1aub0lho1_D0 \
                    + dr.f_2C_lho2_D0P(l, m, w, offdiag)*u0aub0lho2_D0 \
                    + dr.f_2C_lho1_D1P(l, m, w, offdiag)*u0aub0lho1_D1 \
                    + dr.f_2C_D2P(l, m, w, offdiag)*u0aub0_D2 \
                    + dr.df_2C_D0P(l, m, w, offdiag)*u2svp0_D0 \
                    + dr.df_2C_D1P(l, m, w, offdiag)*u1svp0_D1 \
                    + dr.df_2C_lho1_D0P(l, m, w, offdiag)*u1svp0lho1_D0 \
                    + dr.d2f_2C_D0P(l, m, w, offdiag)*u2pls0_D0
                
                offd = -1

        elif component == 'utor':

            if offdiag == -3:

                out = dr.f_2C_D0T(l, m, w, offdiag)*u1aub0_D0 \
                    + dr.f_2C_D1T(l, m, w, offdiag)*u0aub0_D1 \
                    + dr.df_2C_D0T(l, m, w, offdiag)*u1svp0_D0
                
                if ut.symm1 == -1 :
                    offd = -1
                elif ut.symm1 == 1 :
                    offd = -2

            elif offdiag == -1:

                out = dr.f_2C_D0T(l, m, w, offdiag)*u1aub0_D0 \
                    + dr.f_2C_D1T(l, m, w, offdiag)*u0aub0_D1 \
                    + dr.df_2C_D0T(l, m, w, offdiag)*u1svp0_D0
                
                if ut.symm1 == 1 :
                    offd = -1
            
            elif offdiag == 1:

                out = dr.f_2C_D0T(l, m, w, offdiag)*u1aub0_D0 \
                    + dr.f_2C_D1T(l, m, w, offdiag)*u0aub0_D1 \
                    + dr.df_2C_D0T(l, m, w, offdiag)*u1svp0_D0
                if ut.symm1 == -1:
                    offd = 1
            
            elif offdiag == 3:

                out = dr.f_2C_D0T(l, m, w, offdiag)*u1aub0_D0 \
                    + dr.f_2C_D1T(l, m, w, offdiag)*u0aub0_D1 \
                    + dr.df_2C_D0T(l, m, w, offdiag)*u1svp0_D0
                if ut.symm1 == -1:
                    offd = 2
                elif ut.symm1 == 1 : 
                    offd = 1


    if section == 'v':  # ------------------------------------------------------- 1curl

        if component == 'upol':

            if offdiag == -3:

                out = dr.f_1C_D0P(l, m, w, offdiag) * v1_aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, w, offdiag) * v0aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, w, offdiag) * v0aub0_D1 \
                    + dr.df_1C_D0P(l, m, w, offdiag) * v1svp0_D0
                
                if ut.symm1 == -1:
                    offd = -2
                elif ut.symm1 == 1: 
                    offd = -1
            

            elif offdiag == -1:

                out = dr.f_1C_D0P(l, m, w, offdiag) * v1_aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, w, offdiag) * v0aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, w, offdiag) * v0aub0_D1 \
                    + dr.df_1C_D0P(l, m, w, offdiag) * v1svp0_D0
                
                if ut.symm1 == -1:
                    offd = -1
            
            elif offdiag == 1:

                out = dr.f_1C_D0P(l, m, w, offdiag) * v1_aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, w, offdiag) * v0aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, w, offdiag) * v0aub0_D1 \
                    + dr.df_1C_D0P(l, m, w, offdiag) * v1svp0_D0
                
                if ut.symm1 == 1:
                    offd = 1
            
            elif offdiag == 3:

                out = dr.f_1C_D0P(l, m, w, offdiag) * v1_aub0_D0 \
                    + dr.f_1C_lho1_D0P(l, m, w, offdiag) * v0aub0lho1_D0 \
                    + dr.f_1C_D1P(l, m, w, offdiag) * v0aub0_D1 \
                    + dr.df_1C_D0P(l, m, w, offdiag) * v1svp0_D0
                
                if ut.symm1 == -1:
                    offd = 1
                elif ut.symm1 == 1:
                    offd = 2

        elif component == 'utor':

            if offdiag == -2 : 

                out = dr.f_1C_D0T(l, m, w, offdiag) * v0aub0_D0
                offd = -1

            elif offdiag == 0:

                out = dr.f_1C_D0T(l, m, w, offdiag) * v0aub0_D0

            
            elif offdiag == 2 : 
                
                out = dr.f_1C_D0T(l, m, w, offdiag) * v0aub0_D0
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
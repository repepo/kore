import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
from parameters import par
import utils as ut



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



def entropy(l, section, component, offdiag):  # rʰ p s

    out = 0
    L = l*(l+1.)
    
    if (section == 'h') and (offdiag == 0) :

        out = h0pss0_D0

    return out/L**3



def thermal_advection(l, section, component, offdiag):  # −rʰ p(v⋅∇) S = −rʰ p vᵣ dS/dr

    out = 0
    L = l*(l+1.)

    if (section == 'h') and (component == 'upol') and (offdiag == 0) :

        out = -1.0 * h1pdS0_D0
    
    return out/L**2



def thermal_diffusion(l, section, component, offdiag):  # rʰ  ∇⋅(κ p ∇s)

    out = 0
    L = l*(l+1.)
    
    if (section == 'h') and (offdiag == 0) :

        out = - L * h2kps0_D0                   \
              + 2 * h1kps0_D1 + 2 * h0kps1_D1   \
              +     h0kps0_D2 

    return par.ThermaD * out/L**3
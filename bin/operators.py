import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import parameters as par
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

    L = l*(l+1)

    if offdiag == 0:

        if section == 'u' and component == 'upol':

            out =  L * ( L*u3_D0 - u2lho1_D0 - u1lho2_D0
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
                
                out = 1j*m*( -L*u3_D0 + (1-L)*u2lho1_D0 + u1lho2_D0 
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



def viscous_diffusion(l, section, component, offdiag):  # ------------------------------------------------ viscous force

    out = 0
    L= l*(l+1)

    if (offdiag == 0)&(par.ViscosD>0):

        if section == 'u' and component == 'upol':

            out = L * ( -   u1mu0lho4_D0 - 2*      u1mu1lho3_D0 -         u1mu2lho2_D0 
                        - 3*u2mu0lho3_D0 - 2*      u2mu1lho2_D0 +         u2mu2lho1_D0 
                        + L*u3mu0lho2_D0 + 2*L*    u3mu1lho1_D0 + (2-L)*  u3mu2_D0 
                        + L*u4mu0lho1_D0 - 2*(1+L)*u4mu1_D0     + L*(2-L)*u5mu0_D0

                        - 3*u1mu0lho3_D1 - 4*u1mu1lho2_D1 - u1mu2lho1_D1
                        - 6*u2mu0lho2_D1 - 2*u2mu1lho1_D1 + L*u3mu0lho1_D1
                        + 2*(L+1)*u3mu1_D1

                        - 3*u1mu0lho2_D2 - 2*u1mu1lho1_D2 - u1mu2_D2
                        - 3*u2mu0lho1_D2 - 4*u2mu1_D2 + 2*L*u3mu0_D2
                        
                        - u1mu0lho1_D3 - 2*u1mu1_D3 - 4*u2mu0_D3
                    
                        - u1mu0_D4 )
					
        elif section == 'v' and component == 'utor':

            out = L * ( - v2mu1_D0 - L*v3mu0_D0
            
                        + v1mu1_D1 + 2*v2mu0_D1
            
                        + v1mu0_D2 )

    return par.ViscosD * out



def buoyancy(l, section, component, offdiag): 

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        out = L * u2rog0_D0

    return par.Beyonce * out



def entropy(l, section, component, offdiag):  # rʰ ρT s

    out = 0
    
    if (section == 'h') and (offdiag == 0) :

        out = h0roT0_D0

    return out



def thermal_advection(l, section, component, offdiag):  # −rʰ ρT(v⋅∇) S = −rʰ ρT vᵣ dS/dr

    out = 0

    if (section == 'h') and (component == 'upol') and (offdiag == 0) :

        out = -L * h1roT0dSr0_D0
    
    return out



def thermal_diffusion(l, section, component, offdiag):  # rʰ  ∇⋅(κρT ∇s)

    out = 0

    if (section == 'h') and (offdiag == 0) :

        out = - L * h2krT0_D0                   \
              + 2 * h1krT0_D1 + 2 * h0krT1_D1   \
              +     h0krT0_D2 

    return par.ThermaD * out  
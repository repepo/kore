import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import parameters as par




# The following loop read all submatrices needed,
# and creates corresponding operator names as global variables

fname = [f for f in glob.glob('*.mtx')]

for label in fname :

    section = label[0]
    rx      = label[1]
    dx      = label[2]

    if rx == '0' :
        rlabel = ''
    elif rx == '7' :
        rlabel = 'Nr'
    else :
        rlabel = 'r' + rx

    if dx == '0' :
        dlabel = 'I'
    else :
        dlabel = 'D' + dx

    varlabel = rlabel + dlabel + section  # e.g. r3D2v
    
    # add to globals
    globals()[varlabel] = ss.csr_matrix(sio.mmread(label))


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------- Navier-Stokes equation operators
# ----------------------------------------------------------------------------------------------------------------------



def u(l, section, component, offdiag):  # ------------------------------------------------------------------- velocity u

    out = 0
    
    if offdiag == 0:
    
        L = l*(l+1)
        
        if section == 'u' and component == 'upol':
            
            if (par.magnetic == 1 and par.B0 == 'dipole'):
                out = L*( L*r4Iu - 2*r5D1u - r6D2u )  # r6* r.2curl(u)
            else:
                out = L*( L*r2Iu - 2*r3D1u - r4D2u )  # r4* r.2curl(u)
        
        elif section == 'v' and component == 'utor':
            
            if (par.magnetic == 1 and par.B0 == 'dipole'):
                out = L*r5Iv                          # r5* r.1curl(u)
            else:
                out = L*r2Iv                          # r2* r.1curl(u)

    return out




def coriolis(l, section, component, offdiag):  # ------------------------------------------------- Coriolis force 2z x u
    
    out = 0
    L = l*(l+1)
    
    if section == 'u':  # ------------------------------------------------------- 2curl
        
        if component == 'upol':
        
            if offdiag == 0:
        
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2j*par.m*( -L*r4Iu + 2*r5D1u + r6D2u )  # r6* r.2curl(2z x u)
                else:
                    out = 2j*par.m*( -L*r2Iu + 2*r3D1u + r4D2u )  # r4* r.2curl(2z x u)
        
        elif component == 'utor':
            
            if offdiag == -1:
                
                C = (l**2-1)*np.sqrt(l**2-par.m**2) / (2*l-1.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( (l-1)*r5Iu - r6D1u )              # r6* r.2curl(2z x u)
                else:
                    out = 2*C*( (l-1)*r3Iu - r4D1u )              # r4* r.2curl(2z x u)
            
            elif offdiag == 1:
                
                C = l*(l+2.)*np.sqrt((l+par.m+1.)*(l-par.m+1)) / (2.*l+3.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( -(l+2)*r5Iu - r6D1u )             # r6* r.2curl(2z x u)
                else:
                    out = 2*C*( -(l+2)*r3Iu - r4D1u )             # r4* r.2curl(2z x u)
    
    if section == 'v':  # ------------------------------------------------------- 1curl
        
        if component == 'upol':
            
            if offdiag == -1:
                
                C = (l**2-1)*np.sqrt(l**2-par.m**2) / (2*l-1.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( (l-1)*r4Iv - r5D1v )              # r5* r.1curl(2z x u)
                else:
                    out = 2*C*( (l-1)*r1Iv - r2D1v )              # r2* r.1curl(2z x u)
                
            elif offdiag == 1:
                
                C = l*(l+2)*np.sqrt((l+par.m+1.)*(l-par.m+1)) / (2*l+3)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( -(l+2)*r4Iv - r5D1v )             # r5* r.1curl(2z x u)
                else:
                    out = 2*C*( -(l+2)*r1Iv - r2D1v )             # r2* r.1curl(2z x u)               
                
        elif component == 'utor':
            
            if offdiag == 0:
                
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = -2j*par.m*r5Iv                          # r5* r.1curl(2z x u)
                else:
                    out = -2j*par.m*r2Iv                          # r2* r.1curl(2z x u)     
            
    return out
    
    
    
    
def poincare(l, section, component, offdiag):
    
    out = 0
    
    return out
    
    
    
    
def viscous_diffusion(l, section, component, offdiag):  # ----------------------------------- viscous force Ek*nabla^2 u

    out = 0
    if offdiag == 0:
        L = l*(l+1)
        
        if section == 'u' and component == 'upol':
            
            if (par.magnetic == 1 and par.B0 == 'dipole'):
                out = L*( -L*(l+2)*(l-1)*r2Iu + 2*L*r4D2u - 4*r5D3u - r6D4u )  # r6* r.2curl( nabla^2 u )
            else:
                out = L*( -L*(l+2)*(l-1)*Iu + 2*L*r2D2u - 4*r3D3u - r4D4u )    # r4* r.2curl( nabla^2 u )

        elif section == 'v' and component == 'utor':
            
            if (par.magnetic == 1 and par.B0 == 'dipole'):
                out = L*( -L*r3Iv + 2*r4D1v + r5D2v )                          # r5* r.1curl( nabla^2 u )
            else:
                out = L*( -L*Iv + 2*r1D1v + r2D2v )                            # r2* r.1curl( nabla^2 u )
        
    return par.Ek*out




def lorentz(l, section, component, offdiag):  # ---------------------------------------------------------- Lorentz force

    out = 0
        
    return out
    
    
    
    
def buoyancy(l, section, component, offdiag):  # -------------------------------------------------------- buoyancy force

    out = 0
       
    return out



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------- Induction equation operators
# ----------------------------------------------------------------------------------------------------------------------


    
def b(l, section, component, offdiag):

    out = 0
    
    return out




def induction(l, section, component, offdiag):

    out = 0
    
    return out




def magnetic_diffusion(l, section, component, offdiag):

    out = 0
    
    return out



# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------- Heat equation operators
# ----------------------------------------------------------------------------------------------------------------------



def theta(l, section, component, offdiag):

    out = 0
    
    return out




def advection(l, section, component, offdiag):

    out = 0
    
    return out




def thermal_diffusion(l, section, component, offdiag):

    out = 0
    
    return out



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

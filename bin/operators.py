import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import parameters as par




# The following loop read all submatrices needed,
# and creates corresponding operator names as global variables

fname = [f for f in glob.glob('*.mtx')]

for label in fname :

    label = label[:-4]

    section = label[0]
    rx      = label[1]
    dx      = label[-1]

    if len(label) == 3 :
        if rx == '0' :
            rlabel = ''
        elif rx == '7' :
            rlabel = 'Nr'
        else :
            rlabel = 'r' + rx
        hlabel = ''

    elif len(label) == 4 :

        hx = label[2]

        if rx == '0' :
            rlabel = ''
        elif rx == '1' :
            rlabel = 'r'
        elif rx == '6' :
            rlabel = 'q'
        else :
            rlabel = 'r' + rx

        if hx == '0' :
            hlabel = 'h'
        else :
            hlabel = 'h' + hx

    if dx == '0' :
        dlabel = 'I'
    else :
        dlabel = 'D' + dx

    varlabel = rlabel + hlabel + dlabel + section

    globals()[varlabel] = ss.csr_matrix(sio.mmread(label))

# Sign of Ra is going to be important for heat + compositional convection

try:
    signOfRa = np.sign(par.Ra)
except:
    signOfRa = -1.0 # If Ra is not provided, instead only Brunt is present in parameters

try:
    signOfRa_comp = np.sign(par.Ra_comp)
except:
    signOfRa_comp = -1.0

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

    if (par.magnetic==1 and par.tA==1):
        scale_factor = 1.0/par.Le
    else:
        scale_factor = 1.0

    return scale_factor * out




def poincare(l, section, component, offdiag):

    out = 0

    return out




def viscous_diffusion(l, section, component, offdiag):  # ----------------------------------- viscous force Ek*nabla^2 u

    out = 0
    L= l*(l+1)

    inviscid = (par.Ek == 0)

    if (offdiag == 0)&(not inviscid):

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

    if (par.magnetic==1 and par.tA==1):
        scale_factor = par.Ek/(par.Le)
    else:
        scale_factor = par.Ek

    return scale_factor * out




def lorentz(l, section, component, offdiag):  # ---------------------------------------------------------- Lorentz force

    out = 0
    m = par.m
    L = l*(l+1)
    cdipole = ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0))  # boolean

    if section == 'u':    # ------------------------------------------------------- 2curl, section u

        if component == 'bpol':


            if offdiag == -2:  # (l-2) terms  (quadrupole)

                C = (3*(-2 - l + l**2)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)
                out1 = hIu*(2*l + 3*l**2 + l**3) - rhD1u*(6 - 7*l + 3*l**2) + rh1Iu*(2 + l - 6*l**2 + l**3)
                out2 = - r2hD2u*(-6 + l) - 2*r2h1D1u*(-2 + l) + r2h2Iu*(-2 + l) - r3h2D1u + 3*r3hD3u
                out3 = - r3h1D2u*(-3 + l) + r3h3Iu*(-1 + l)
                out = C*(out1+out2+out3)


            elif offdiag == -1:  # (l-1) terms

                C = np.sqrt(l**2-par.m**2)*(l**2-1)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :  # r4*r.2curl
                    out1 = -2*(l**2+2)*rhD1u -2*(l-2)*r2h1D1u - (l-4)*r2hD2u - (l-2)*r3h1D2u
                    out2 = L*(l+2)*hIu + L*(l-4)*rh1Iu + l*r2h2Iu + l*r3h3Iu + 2*r3hD3u
                elif par.B0 == 'dipole':
                    # same but +r2
                    out1 = -2*(l**2+2)*r3hD1u -2*(l-2)*r4h1D1u - (l-4)*r4hD2u - (l-2)*r5h1D2u
                    out2 = L*(l+2)*r2hIu + L*(l-4)*r3h1Iu + l*r4h2Iu + l*r5h3Iu + 2*r5hD3u
                out = C*(out1+out2)


            elif offdiag == 0:  # l terms  (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l + 4*l**2)
                out = C*( 3*hIu*l*(1 + l)*(-2 + l + l**2) - 3*rh1Iu*L**2 + 2*rhD1u*(6 - 4*l - 5*l**2 - 2*l**3 - l**4) \
                          + 3*r2h2Iu*L + r2hD2u*(-12 + 5*l + 5*l**2) + 2*r2h1D1u*(-6 + 5*l + 5*l**2) + 2*r3h2D1u*L \
                          + r3h3Iu*L + 2*r3hD3u*(-3 + l + l**2) + 3*r3h1D2u*(-2 + l + l**2) )


            elif offdiag == 1:  # (l+1) terms

                C = np.sqrt((1+l+par.m)*(1+l-par.m))*l*(l+2)/(2*l+3)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :  # r4*r.2curl
                    out1 = -2*(l**2+2*l+3)*rhD1u + 2*(l+3)*r2h1D1u + (l+5)*r2hD2u + (l+3)*r3h1D2u
                    out2 = -L*(l-1)*hIu - L*(l+5)*rh1Iu - (l+1)*r2h2Iu - (l+1)*r3h3Iu + 2*r3hD3u
                elif par.B0 == 'dipole':
                    # same but +r2
                    out1 = -2*(l**2+2*l+3)*r3hD1u + 2*(l+3)*r4h1D1u + (l+5)*r4hD2u + (l+3)*r5h1D2u
                    out2 = -L*(l-1)*r2hIu - L*(l+5)*r3h1Iu - (l+1)*r4h2Iu - (l+1)*r5h3Iu + 2*r5hD3u
                out = C*(out1+out2)


            elif offdiag == 2:  # (l+2) terms  (quadrupole)

                C = (3*l*(3 + l)*np.sqrt((1 + l - m)*(2 + l + m))*np.sqrt(2 + 3*l + l**2 + m - m**2))/(15 + 16*l + 4*l**2)
                out = C*( hIu*(l - l**3) - rhD1u*(16 + 13*l + 3*l**2) - rh1Iu*(6 + 16*l + 9*l**2 + l**3) + 2*r2h1D1u*(3 + l) \
                          - r2h2Iu*(3 + l) + r2hD2u*(7 + l) - r3h2D1u + 3*r3hD3u + r3h1D2u*(4 + l) - r3h3Iu*(2 + l) )

        elif component == 'btor':

            if offdiag == -1:  # (l-1) terms  (quadrupole)

                C = (6j*m*np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( -rhIu*(3-3*l-2*l**2) - r2hD1u*(l-3) - r2h1Iu*(-3+2*l+l**2) + 3*r3hD2u - r3h1D1u*(l-3) - r3h2Iu*l )

            elif offdiag == 0:  # l terms

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :  # r4*r.2curl
                    out = 2j*par.m*( -rhIu - (l**2+l-1)*r2h1Iu + r2hD1u + r3h1D1u + r3hD2u )
                elif par.B0 == 'dipole':
                    # same but +r2
                    out = 2j*par.m*( -r3hIu - (l**2+l-1)*r4h1Iu + r4hD1u + r5h1D1u + r5hD2u )

            elif offdiag == 1:  # (l+1) terms  (quadrupole)

                C = (6j*m*np.sqrt(1 + 2*l + l**2 - m**2))/(3 + 2*l)
                out = C*( rhIu*(-4+l+2*l**2) + r2hD1u*(4+l) - r2h1Iu*(-4+l**2) + 3*r3hD2u + r3h2Iu*(1+l) + r3h1D1u*(4+l) )


    elif section == 'v':  # ------------------------------------------------------- 1curl, section v

        if component == 'bpol':

            if offdiag == -1:  # (l-1) terms  (quadrupole)

                C = (3j*m*np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( 12*hD1v - 2*h1Iv*(-1 + l)*l + 6*rhD2v - rh2Iv*(-1 + l)*l )

            elif offdiag == 0:  # l terms (dipole)

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = 1j*par.m*( 4*hD1v - L*( 2*h1Iv + rh2Iv ) + 2*rhD2v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = 1j*par.m*( 4*r3hD1v - L*( 2*r3h1Iv + r4h2Iv ) + 2*r4hD2v )  # r5*r.1curl

            elif offdiag == 1:  # (l+1) terms  (quadrupole)

                C = (3j*m*np.sqrt((1 + l - m)*(1 + l + m)))/(3 + 2*l)
                out = C*( 12*hD1v - 2*h1Iv*(1 + l)*(2 + l) + 6*rhD2v - rh2Iv*(1 + l)*(2 + l) )

        elif component == 'btor':

            if offdiag == -2:  # (l-2) terms  (quadrupole)

                C = (3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)
                out = C*( hIv*(-4 + l) - 3*rhD1v + rh1Iv*(-1 + l) )

            elif offdiag == -1:

                C = np.sqrt((l-par.m)*(l+par.m))*(l**2-1)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( (l-2)*hIv + l*rh1Iv -2*rhD1v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l-2)*r3hIv + l*r4h1Iv -2*r4hD1v )  # r5*r.1curl

            elif offdiag == 0:  # l terms  (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))
                out = C*( hIv*(6 - l - l**2) + rh1Iv*L - 2*rhD1v*(-3 + l + l**2) )

            elif offdiag == 1:

                C = -np.sqrt((l+par.m+1)*(l+1-par.m))*l*(l+2)/(2*l+3)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( (l+3)*hIv + (l+1)*rh1Iv + 2*rhD1v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l+3)*r3hIv + (l+1)*r4h1Iv + 2*r4hD1v )  # r5*r.1curl

            elif offdiag == 2:  # (l+2) terms  (quadrupole)

                C = (3*l*(3 + l)*np.sqrt((2 + l - m)*(1 + l + m))*np.sqrt((1 + l - m)*(2 + l + m)))/((3 + 2*l)*(5 + 2*l))
                out = C*( -hIv*(5 + l) - 3*rhD1v - rh1Iv*(2 + l) )


    if (par.magnetic==1 and par.tA==1):
        scale_factor = 1.0
    else:
        scale_factor = par.Le2

    return scale_factor * out




def buoyancy(l, section, component, offdiag):  # -------------------------------------------------------- buoyancy force

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        if (par.magnetic == 1) and (par.B0 == 'dipole') :
            buoy = L * r6Iu
        else:
            buoy = L * r4Iu

        if par.heating == 'two zone' or par.heating == 'user defined' :
            BVsq = 1
        else :
            BVsq = signOfRa * par.Brunt**2

    out = BVsq * buoy

    return out

def comp_buoyancy(l, section, component, offdiag):  # -------------------------------------------------------- buoyancy force

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        if (par.magnetic == 1) and (par.B0 == 'dipole') :
            buoy = L * r6Iu
        else:
            buoy = L * r4Iu

        if par.comp_background == 'two zone' or par.comp_background == 'user defined' :
            BVsq = 1
        else :
            BVsq = signOfRa_comp * par.Brunt_comp**2

    out = BVsq * buoy

    return out



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------- Induction equation operators
# ----------------------------------------------------------------------------------------------------------------------



def b(l, section, component, offdiag):
    '''
    The magnetic field 𝐛
    '''

    out = 0
    L = l*(l+1)

    if offdiag == 0:

        if section == 'f' and component == 'bpol':  #  r² 𝐫⋅𝐛   (×r² if dipole)
            if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                out = L* r2If
            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                out = L* r4If

        elif section == 'g' and component == 'btor':  # r² 𝐫⋅∇×𝐛   (×r³ if dipole)
            if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                out = L* r2Ig
            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                out = L* r5Ig

    return out




def induction(l, section, component, offdiag):
    '''
    The induction term ∇×(𝐁₀×𝐮)
    '''

    l = np.float128(l)  # to avoid overflow errors at high N
    m = par.m

    out = 0
    L = l*(l+1)


    if section == 'f':  # ---------------------------------------------------- nocurl  r² 𝐫⋅∇×(𝐁₀×𝐮)  (×r² if classic dipole)

        if component == 'upol':

            if offdiag == -2:  # l-2 terms (quadrupole)

                C1 = 3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m))
                C2 = 3 - 8*l + 4*l**2
                out = ( hIf*(-4 + l) - 3*rhD1f + rh1If*(-1 + l) )*C1/C2

            elif offdiag == -1:  # l-1 terms (dipole)

                C = np.sqrt(l**2-par.m**2)*(l**2-1)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1'] :
                    out = C*( (l-2)*hIf + l*rh1If - 2*rhD1f )
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l-2)*r2hIf + l*r3h1If - 2*r3hD1f )

            elif offdiag == 0:  # l terms (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))
                out = C*( hIf*(6 - l - l**2) + rh1If*l*(1 + l) - 2*rhD1f*(-3 + l + l**2) )

            elif offdiag == 1:  # l+1 terms (dipole)

                C = np.sqrt((l+1)**2-par.m**2)*l*(l+2)/(2*l+3)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( -(l+3)*hIf -(l+1)*rh1If -2*rhD1f )
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( -(l+3)*r2hIf -(l+1)*r3h1If -2*r3hD1f )

            elif offdiag == 2:  # l+2 terms (quadrupole)

                C1 = 3*l*(3 + l)*np.sqrt((2 + l - m)*(1 + l + m))*np.sqrt((1 + l - m)*(2 + l + m))
                C2 = (3 + 2*l)*(5 + 2*l)
                out = ( hIf*(-5 - l) - 3*rhD1f - rh1If*(2 + l) )*C1/C2

        elif component == 'utor':

            if offdiag == -1:  # l-1 terms (quadrupole)

                out = 18j * rhIf * m * np.sqrt(l**2-m**2)/(1-2*l)

            elif offdiag == 0:  # l terms (dipole)

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = -2j*par.m* rhIf
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = -2j*par.m* r3hIf

            elif offdiag == 1:  # l+1 terms (quadrupole)

                out = -18j * rhIf * m * np.sqrt((1+l-m)*(1+l+m))/(3+2*l)


    elif section == 'g':  # --------------------------------------------- 1curl  r² 𝐫⋅∇×( ∇×(𝐁₀×𝐮) )  (×r³ if dipole)

        if component == 'upol':

            if offdiag == -1:  # l-1 terms (quadrupole)

                C = (3j * m * np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( -2*h1Ig*(-3 + l) - 2*hD1g*(-3 + l) - 2*qhIg*(3 + l**2) + 6*rhD2g - 2*rh1D1g*(-3 + l) + rh2Ig*(-1 + l)*l )

            elif offdiag == 0:  # l terms (dipole)

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = 2j*par.m*( hD1g + rh1D1g -(l**2+l+1)*qhIg + h1Ig + (L/2)*rh2Ig + rhD2g )  # qh=h/r
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = 2j*par.m*( r3hD1g + r4h1D1g -(l**2+l+1)*r2hIg + r3h1Ig + (L/2)*r4h2Ig + r4hD2g )

            elif offdiag == 1:  # l+1 terms (quadrupole)

                C = (3j*m*np.sqrt((1 + l - m)*(1 + l + m)))/(3 + 2*l)
                out = C*( 2*(4+l)*( h1Ig + hD1g + rh1D1g ) - 2*qhIg*(4 + 2*l + l**2) + 6*rhD2g + rh2Ig*(2 + 3*l + l**2) )

        elif component == 'utor':

            if offdiag == -2:  # l-2 terms (quadrupole)

                C = (3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)
                out = C*( hIg*l - 3*rhD1g + rh1Ig*(-3 + l) )

            elif offdiag == -1:  # l-1 terms (dipole)

                C  = (2*l+1)*np.sqrt( l*(l**2-1)*(l**2-par.m**2)/(4*l**2-1) )
                C1 = np.sqrt( (l**2-1)/(4*l**3-l) )
                C2 = np.sqrt( l*(l**2-1)/(4*l**2-1) )

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( C2* hIg -2*C1* rhD1g + (C2-2*C1)* rh1Ig )
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( C2* r3hIg -2*C1* r4hD1g + (C2-2*C1)* r4h1Ig )

            elif offdiag == 0:  # l terms (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))
                out = C*( -hIg*L - 2*rhD1g*(-3 + l + l**2) - 3*rh1Ig*(-2 + l + l**2) )

            elif offdiag == 1:  # l+1 terms

                C  = np.sqrt( (l+2)*(2*l+1)*L*((l+1)**2-par.m**2)/(2*l+3) )
                C1 = np.sqrt( l*(l+2)/(3+11*l+12*l**2+4*l**3) )
                C2 = np.sqrt( L*(l+2)/(3+4*l*(l+2)) )

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( -C2* hIg -2*C1* rhD1g -(2*C1+C2)* rh1Ig )
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( -C2* r3hIg -2*C1* r4hD1g -(2*C1+C2)* r4h1Ig )

            elif offdiag == 2:  # l+2 terms (quadrupole)

                C = (3*l*(3 + l)*np.sqrt(2 + 3*l + l**2 - m - m**2)*np.sqrt(2 + 3*l + l**2 + m - m**2))/(15 + 16*l + 4*l**2)
                out = C*( -hIg*(1 + l) - 3*rhD1g - rh1Ig*(4 + l) )

    return out




def magnetic_diffusion(l, section, component, offdiag):
    '''
    The magnetic difussion term Eₘ∇²𝐛
    '''

    out = 0
    L= l*(l+1)

    if offdiag == 0:

        if section == 'f' and component == 'bpol':  #  r² 𝐫⋅∇²𝐛   (×r² if dipole)
            if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                out = L*( -L*If + 2*r1D1f + r2D2f )
            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                out = L*( -L*r2If + 2*r3D1f + r4D2f )

        elif section == 'g' and component == 'btor':  # r² 𝐫⋅∇×(∇²𝐛)  (×r³ if dipole)
            if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                out = L*( -L*Ig + 2*r1D1g + r2D2g )
            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                out = L*( -L*r3Ig + 2*r4D1g + r5D2g )

    if (par.magnetic==1 and par.tA==1):
        scale_factor = par.Em/par.Le
    else:
        scale_factor = par.Em

    return scale_factor * out







# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------- Heat equation operators
# ----------------------------------------------------------------------------------------------------------------------



def theta(l, section, component, offdiag):

    out = 0
    if (section == 'h') and (offdiag == 0) :

        if par.heating == 'differential' :
            out = r3Ih
        else:
            out = r2Ih

    return out

def composition(l, section, component, offdiag):

    out = 0
    if (section == 'h') and (offdiag == 0) :

        if par.comp_background == 'differential' :
            out = r3Ih
        else:
            out = r2Ih

    return out

def thermal_advection(l, section, component, offdiag):

    out = 0
    L = l*(l+1)

    rcmb = 1

    if ((section == 'h') and (component == 'upol')) and (offdiag == 0) :

        if par.heating == 'internal' :
            conv = signOfRa * L*r2Ih
            #conv = L*r4Ih
        elif par.heating == 'differential' :
            conv = signOfRa * L*Ih * par.ricb/(rcmb-par.ricb)
        elif par.heating == 'two zone' or par.heating == 'user defined' :
            conv = - L * (par.Brunt**2) * NrIh
        out = conv

    return out

def compositional_advection(l, section, component, offdiag):

    out = 0
    L = l*(l+1)

    rcmb = 1

    if ((section == 'h') and (component == 'upol')) and (offdiag == 0) :

        if par.comp_background == 'internal' :
            conv = signOfRa_comp * L*r2Ih
            #conv = L*r4Ih
        elif par.comp_background == 'differential' :
            conv = signOfRa_comp * L*Ih * par.ricb/(rcmb-par.ricb)
        elif par.comp_background == 'two zone' or par.comp_background == 'user defined' :
            conv = - L * (par.Brunt_comp**2) * NrIh
        out = conv

    return out


def thermal_diffusion(l, section, component, offdiag):

    out = 0
    L = l*(l+1)

    if section == 'h' and offdiag == 0 :

        if (par.heating == 'internal') or (par.heating == 'two zone' or par.heating == 'user defined') :
            difus = - L*Ih + 2*r1D1h + r2D2h
        elif par.heating == 'differential' :
            difus = - L*r1Ih + 2*r2D1h + r3D2h
        out = (par.Ek/par.Prandtl) * difus


    return out

def compositional_diffusion(l, section, component, offdiag):

    out = 0
    L = l*(l+1)

    if section == 'h' and offdiag == 0 :

        if (par.comp_background == 'internal') or (par.comp_background == 'two zone' or par.comp_background == 'user defined') :
            difus = - L*Ih + 2*r1D1h + r2D2h
        elif par.comp_background == 'differential' :
            difus = - L*r1Ih + 2*r2D1h + r3D2h
        out = (par.Ek/par.Schmidt) * difus


    return out


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import glob
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
from parameters import par
import utils as ut

# In the following loop we read all submatrices needed,
# and create corresponding operator names as global variables

fname = [f for f in glob.glob('*.mtx')]

for label in fname :

    label = label[:-4]  # to get rid of the ".mtx"

    section = label[0]
    if section in ['u','v','f','g','h','i']:

        prof_id = ''
        proflabel = ''
        hlabel = ''
        if len(label) == 7 :
            prof_id = label[1:4]
            rx      = label[4]
        else:
            rx = label[1]
        dx = label[-1]


        if len(label) == 3 :
            if rx == '0' :
                rlabel = ''
            elif rx == '7' :
                rlabel = 'Nr'
            else :
                rlabel = 'r' + rx

        if len(label) == 4 :

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

        if len(label) == 7 :

            if rx == '0' :
                rlabel = ''
            elif rx == '1' :
                rlabel = 'r'
            else :
                rlabel = 'r' + rx

            if label[5] == '0' :
                proflabel = prof_id
            else:
                proflabel = prof_id + label[5]


        if dx == '0' :
            dlabel = 'I'
        else :
            dlabel = 'D' + dx

        varlabel = rlabel + hlabel + proflabel + dlabel + section

        globals()[varlabel] = ss.csr_matrix(sio.mmread(label+'.mtx'))

if ut.cic:  # for a conductive inner core

    r2_D0f_ic = ss.csr_matrix(sio.mmread('r2_D0f_ic.mtx'))
    r2_D0g_ic = ss.csr_matrix(sio.mmread('r2_D0g_ic.mtx'))
    r0_D0f_ic = ss.csr_matrix(sio.mmread('r0_D0f_ic.mtx'))
    r0_D0g_ic = ss.csr_matrix(sio.mmread('r0_D0g_ic.mtx'))
    r1_D1f_ic = ss.csr_matrix(sio.mmread('r1_D1f_ic.mtx'))
    r1_D1g_ic = ss.csr_matrix(sio.mmread('r1_D1g_ic.mtx'))
    r2_D2f_ic = ss.csr_matrix(sio.mmread('r2_D2f_ic.mtx'))
    r2_D2g_ic = ss.csr_matrix(sio.mmread('r2_D2g_ic.mtx'))



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

    out  = 0
    offd = 0
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

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 1:

                C = l*(l+2.)*np.sqrt((l+par.m+1.)*(l-par.m+1)) / (2.*l+3.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( -(l+2)*r5Iu - r6D1u )             # r6* r.2curl(2z x u)
                else:
                    out = 2*C*( -(l+2)*r3Iu - r4D1u )             # r4* r.2curl(2z x u)

                if ut.symm1 == -1:
                    offd = 1

    if section == 'v':  # ------------------------------------------------------- 1curl

        if component == 'upol':

            if offdiag == -1:

                C = (l**2-1)*np.sqrt(l**2-par.m**2) / (2*l-1.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( (l-1)*r4Iv - r5D1v )              # r5* r.1curl(2z x u)
                else:
                    out = 2*C*( (l-1)*r1Iv - r2D1v )              # r2* r.1curl(2z x u)

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:

                C = l*(l+2)*np.sqrt((l+par.m+1.)*(l-par.m+1)) / (2*l+3)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( -(l+2)*r4Iv - r5D1v )             # r5* r.1curl(2z x u)
                else:
                    out = 2*C*( -(l+2)*r1Iv - r2D1v )             # r2* r.1curl(2z x u)

                if ut.symm1 == 1:
                    offd = 1

        elif component == 'utor':

            if offdiag == 0:

                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = -2j*par.m*r5Iv                          # r5* r.1curl(2z x u)
                else:
                    out = -2j*par.m*r2Iv                          # r2* r.1curl(2z x u)

    return [ par.OmgTau * out, offd ]



def poincare(l, section, component, offdiag):  # Coming soon

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

    return par.OmgTau * par.Ek * out



def lorentz_pol(l, section, component, offdiag):  # ------------------------------------------- Lorentz force, poloidal B0

    out = 0
    offd = 0
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

                offd = -1


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

                if ut.symm1 == 1:
                    offd = -1


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

                if ut.symm1 == -1:
                    offd = 1


            elif offdiag == 2:  # (l+2) terms  (quadrupole)

                C = (3*l*(3 + l)*np.sqrt((1 + l - m)*(2 + l + m))*np.sqrt(2 + 3*l + l**2 + m - m**2))/(15 + 16*l + 4*l**2)
                out = C*( hIu*(l - l**3) - rhD1u*(16 + 13*l + 3*l**2) - rh1Iu*(6 + 16*l + 9*l**2 + l**3) + 2*r2h1D1u*(3 + l) \
                          - r2h2Iu*(3 + l) + r2hD2u*(7 + l) - r3h2D1u + 3*r3hD3u + r3h1D2u*(4 + l) - r3h3Iu*(2 + l) )

                offd = 1


        elif component == 'btor':


            if offdiag == -1:  # (l-1) terms  (quadrupole)

                C = (6j*m*np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( -rhIu*(3-3*l-2*l**2) - r2hD1u*(l-3) - r2h1Iu*(-3+2*l+l**2) + 3*r3hD2u - r3h1D1u*(l-3) - r3h2Iu*l )

                if ut.symm1 == 1:
                    offd = -1


            elif offdiag == 0:  # l terms

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :  # r4*r.2curl
                    out = 2j*par.m*( -rhIu - (l**2+l-1)*r2h1Iu + r2hD1u + r3h1D1u + r3hD2u )
                elif par.B0 == 'dipole':
                    # same but +r2
                    out = 2j*par.m*( -r3hIu - (l**2+l-1)*r4h1Iu + r4hD1u + r5h1D1u + r5hD2u )


            elif offdiag == 1:  # (l+1) terms  (quadrupole)

                C = (6j*m*np.sqrt(1 + 2*l + l**2 - m**2))/(3 + 2*l)
                out = C*( rhIu*(-4+l+2*l**2) + r2hD1u*(4+l) - r2h1Iu*(-4+l**2) + 3*r3hD2u + r3h2Iu*(1+l) + r3h1D1u*(4+l) )

                if ut.symm1 == -1:
                    offd = 1


    elif section == 'v':  # ------------------------------------------------------- 1curl, section v


        if component == 'bpol':


            if offdiag == -1:  # (l-1) terms  (quadrupole)

                C = (3j*m*np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( 12*hD1v - 2*h1Iv*(-1 + l)*l + 6*rhD2v - rh2Iv*(-1 + l)*l )

                if ut.symm1 == -1:
                    offd = -1


            elif offdiag == 0:  # l terms (dipole)

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = 1j*par.m*( 4*hD1v - L*( 2*h1Iv + rh2Iv ) + 2*rhD2v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = 1j*par.m*( 4*r3hD1v - L*( 2*r3h1Iv + r4h2Iv ) + 2*r4hD2v )  # r5*r.1curl


            elif offdiag == 1:  # (l+1) terms  (quadrupole)

                C = (3j*m*np.sqrt((1 + l - m)*(1 + l + m)))/(3 + 2*l)
                out = C*( 12*hD1v - 2*h1Iv*(1 + l)*(2 + l) + 6*rhD2v - rh2Iv*(1 + l)*(2 + l) )

                if ut.symm1 == 1:
                    offd = 1


        elif component == 'btor':


            if offdiag == -2:  # (l-2) terms  (quadrupole)

                C = (3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)
                out = C*( hIv*(-4 + l) - 3*rhD1v + rh1Iv*(-1 + l) )

                offd = -1


            elif offdiag == -1:

                C = np.sqrt((l-par.m)*(l+par.m))*(l**2-1)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( (l-2)*hIv + l*rh1Iv -2*rhD1v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l-2)*r3hIv + l*r4h1Iv -2*r4hD1v )  # r5*r.1curl

                if ut.symm1 == -1:
                    offd = -1


            elif offdiag == 0:  # l terms  (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))
                out = C*( hIv*(6 - l - l**2) + rh1Iv*L - 2*rhD1v*(-3 + l + l**2) )


            elif offdiag == 1:

                C = -np.sqrt((l+par.m+1)*(l+1-par.m))*l*(l+2)/(2*l+3)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( (l+3)*hIv + (l+1)*rh1Iv + 2*rhD1v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l+3)*r3hIv + (l+1)*r4h1Iv + 2*r4hD1v )  # r5*r.1curl

                if ut.symm1 == 1:
                    offd = 1


            elif offdiag == 2:  # (l+2) terms  (quadrupole)

                C = (3*l*(3 + l)*np.sqrt((2 + l - m)*(1 + l + m))*np.sqrt((1 + l - m)*(2 + l + m)))/((3 + 2*l)*(5 + 2*l))
                out = C*( -hIv*(5 + l) - 3*rhD1v - rh1Iv*(2 + l) )

                offd = 1


    return [ (par.OmgTau*par.Le)**2 * out, offd ]



def buoyancy(l, section, component, offdiag):  # -------------------------------------------------------- buoyancy force

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        if (par.magnetic == 1) and (par.B0 == 'dipole') :
            buoy = r6Iu
        else:
            buoy = r4Iu

    out = L * buoy

    return par.OmgTau**2 * par.BV2 * out



def comp_buoyancy(l, section, component, offdiag):  # ------------------------------------- compositional buoyancy force

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        if (par.magnetic == 1) and (par.B0 == 'dipole') :
            buoy = r6Iu
        else:
            buoy = r4Iu

    out = L * buoy

    return par.OmgTau**2 * par.BV2_comp * out



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------- Induction equation operators
# ----------------------------------------------------------------------------------------------------------------------



def b(l, section, component, offdiag):
    '''
    The magnetic field 𝐛
    '''
    cdipole = ((par.B0 == 'dipole') and (par.ricb > 0))  # boolean
    out = 0
    L = l*(l+1)

    if offdiag == 0:

        if section == 'f' and component == 'bpol':  #  r² 𝐫⋅𝐛   (×r² if dipole)
            if not cdipole:
                out = L* r2If
            else:
                out = L* r4If

        elif section == 'g' and component == 'btor':  # r² 𝐫⋅∇×𝐛   (×r³ if dipole)
            if not cdipole:
                out = L* r2Ig
            else:
                out = L* r5Ig

        elif section == 'fic' and component == 'bpol_ic':
            out = L * r2_D0f_ic

        elif section == 'gic' and component == 'btor_ic':
            out = L * r2_D0g_ic

    return out



def induction_pol(l, section, component, offdiag):
    '''
    The induction term ∇×(𝐁₀×𝐮) of a poloidal B0, radial projection (section f) or radial projection of the curl (section g)
    '''

    out = 0
    offd = 0
    m = par.m
    l = np.longdouble(l)  # to avoid overflow errors at high N (float128 on x86, float64 on arm64)
    L = l*(l+1)
    cdipole = ((par.B0 == 'dipole') and (par.ricb > 0))  # boolean


    if section == 'f':  # ---------------------------------------------------- nocurl  r² 𝐫⋅∇×(𝐁₀×𝐮)  (×r² if classic dipole)

        if component == 'upol':

            if offdiag == -2:  # l-2 terms (quadrupole)

                C1 = 3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m))
                C2 = 3 - 8*l + 4*l**2

                out = ( hIf*(-4 + l) - 3*rhD1f + rh1If*(-1 + l) )*C1/C2

                offd = -1

            elif offdiag == -1:  # l-1 terms (dipole)

                C = np.sqrt(l**2-par.m**2)*(l**2-1)/(2*l-1)

                if not cdipole:
                    out = C*( (l-2)*hIf + l*rh1If - 2*rhD1f )
                else:
                    out = C*( (l-2)*r2hIf + l*r3h1If - 2*r3hD1f )

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 0:  # l terms (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))

                out = C*( hIf*(6 - l - l**2) + rh1If*l*(1 + l) - 2*rhD1f*(-3 + l + l**2) )

            elif offdiag == 1:  # l+1 terms (dipole)

                C = np.sqrt((l+1)**2-par.m**2)*l*(l+2)/(2*l+3)

                if not cdipole:
                    out = C*( -(l+3)*hIf -(l+1)*rh1If -2*rhD1f )
                else:
                    out = C*( -(l+3)*r2hIf -(l+1)*r3h1If -2*r3hD1f )

                if ut.symm1 == 1:
                    offd = 1

            elif offdiag == 2:  # l+2 terms (quadrupole)

                C1 = 3*l*(3 + l)*np.sqrt((2 + l - m)*(1 + l + m))*np.sqrt((1 + l - m)*(2 + l + m))
                C2 = (3 + 2*l)*(5 + 2*l)

                out = ( hIf*(-5 - l) - 3*rhD1f - rh1If*(2 + l) )*C1/C2

                offd = 1

        elif component == 'utor':

            if offdiag == -1:  # l-1 terms (quadrupole)

                out = 18j * rhIf * m * np.sqrt(l**2-m**2)/(1-2*l)

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 0:  # l terms (dipole)

                if not cdipole:
                    out = -2j*par.m* rhIf
                else:
                    out = -2j*par.m* r3hIf

            elif offdiag == 1:  # l+1 terms (quadrupole)

                out = -18j * rhIf * m * np.sqrt((1+l-m)*(1+l+m))/(3+2*l)

                if ut.symm1 == -1:
                    offd = 1


    elif section == 'g':  # --------------------------------------------- 1curl  r² 𝐫⋅∇×( ∇×(𝐁₀×𝐮) )  (×r³ if dipole)

        if component == 'upol':

            if offdiag == -1:  # l-1 terms (quadrupole)

                C = (3j * m * np.sqrt(l**2 - m**2))/(-1 + 2*l)

                out = C*( -2*h1Ig*(-3 + l) - 2*hD1g*(-3 + l) - 2*qhIg*(3 + l**2) + 6*rhD2g - 2*rh1D1g*(-3 + l) + rh2Ig*(-1 + l)*l )

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 0:  # l terms (dipole)

                if not cdipole:
                    out = 2j*par.m*( hD1g + rh1D1g -(l**2+l+1)*qhIg + h1Ig + (L/2)*rh2Ig + rhD2g )  # qh=h/r
                else:
                    out = 2j*par.m*( r3hD1g + r4h1D1g -(l**2+l+1)*r2hIg + r3h1Ig + (L/2)*r4h2Ig + r4hD2g )

            elif offdiag == 1:  # l+1 terms (quadrupole)

                C = (3j*m*np.sqrt((1 + l - m)*(1 + l + m)))/(3 + 2*l)

                out = C*( 2*(4+l)*( h1Ig + hD1g + rh1D1g ) - 2*qhIg*(4 + 2*l + l**2) + 6*rhD2g + rh2Ig*(2 + 3*l + l**2) )

                if ut.symm1 == 1:
                    offd = 1

        elif component == 'utor':

            if offdiag == -2:  # l-2 terms (quadrupole)

                C = (3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)

                out = C*( hIg*l - 3*rhD1g + rh1Ig*(-3 + l) )

                offd = -1

            elif offdiag == -1:  # l-1 terms (dipole)

                C  = (2*l+1)*np.sqrt( l*(l**2-1)*(l**2-par.m**2)/(4*l**2-1) )
                C1 = np.sqrt( (l**2-1)/(4*l**3-l) )
                C2 = np.sqrt( l*(l**2-1)/(4*l**2-1) )

                if not cdipole:
                    out = C*( C2* hIg -2*C1* rhD1g + (C2-2*C1)* rh1Ig )
                else:
                    out = C*( C2* r3hIg -2*C1* r4hD1g + (C2-2*C1)* r4h1Ig )

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 0:  # l terms (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))

                out = C*( -hIg*L - 2*rhD1g*(-3 + l + l**2) - 3*rh1Ig*(-2 + l + l**2) )

            elif offdiag == 1:  # l+1 terms

                C = l*(l+2)*np.sqrt(1+2*l+l**2-par.m**2)/(3+2*l)

                if not cdipole:
                    out = C * ( -2*rhD1g - (l+1)* hIg - (l+3) * rh1Ig )
                else:
                    out = C * ( -2*r4hD1g - (l+1)* r3hIg - (l+3) * r4h1Ig )

                if ut.symm1 == -1:
                    offd = 1

            elif offdiag == 2:  # l+2 terms (quadrupole)

                C = (3*l*(3 + l)*np.sqrt(2 + 3*l + l**2 - m - m**2)*np.sqrt(2 + 3*l + l**2 + m - m**2))/(15 + 16*l + 4*l**2)

                out = C*( -hIg*(1 + l) - 3*rhD1g - rh1Ig*(4 + l) )

                offd = 1

    return [ out, offd ]



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------- NEW (2026-10-30): toroidal B0, and the totals
# ----------------------------------------------------------------------------------------------------------------------
# Ported from the anelastic_log branch (its operators.py, 2026-10-28, derived there with tests/magnetic/derive.py and fitted
# in l and m), at uniform density and without its 1/L³ row scaling, in this branch's labels ('bto' = cnorm_tor k^(j), see
# submatrices.py). B0_tor = cnorm_tor ∇×(k Y_n⁰ 𝐫), B_φ = -cnorm_tor k ∂θY_n⁰. Degree 1 (equatorially symmetric) couples:
# rows f to P at l; g to P at l ± 1 and T at l; u to F at l and G at l ± 1; v to F at l ± 1 and G at l. Degree 2
# (antisymmetric, as the dipole) couples: f to P at l ± 1; g to P at l, l ± 2 and T at l ± 1; u to F at l ± 1 and G at l,
# l ± 2; v to F at l, l ± 2 and G at l ± 1.


def moff(row, col, dl):
    '''
    Block offset (units of N1) of the column block of degree l + dl for the rows of section row ('u', 'v', 'f', 'g') and the
    unknown col ('P', 'T', 'F', 'G'), from whether the two use the l's of the poloidal flow (top) or of the toroidal flow.
    The same offsets as the poloidal B0 terms above.
    '''
    top = {'u': True, 'v': False, 'P': True, 'T': False, 'f': ut.symmB0 == 1, 'g': ut.symmB0 != 1, 'F': ut.symmB0 == 1, 'G': ut.symmB0 != 1}
    if top[row] == top[col]:
        return int(np.trunc(dl/2))   # dl = ±2 is one block, dl = 0 none
    if top[row]:
        return -1 if (dl == -1 and ut.symm1 == 1) else (1 if (dl == 1 and ut.symm1 == -1) else 0)
    return -1 if (dl == -1 and ut.symm1 == -1) else (1 if (dl == 1 and ut.symm1 == 1) else 0)



def quad_factors(l, m):
    '''
    The angular factors of a degree-2 background field, by dl = -2..2: products of the coefficients of the recurrence
    cos θ Y_l = p_l Y_(l+1) + q_l Y_(l-1) that lead from degree l + dl (the column) to l (the row), and i m where a
    φ-derivative enters
    '''
    return { 0: 1/((2*l - 1)*(2*l + 3)),
            -1: 1j*m*np.sqrt(l**2 - m**2)/(2*l - 1),
             1: 1j*m*np.sqrt((l+1)**2 - m**2)/(2*l + 3),
            -2: np.sqrt((l**2 - m**2)*((l-1)**2 - m**2))/((2*l - 1)*(2*l - 3)),
             2: np.sqrt(((l+1)**2 - m**2)*((l+2)**2 - m**2))/((2*l + 3)*(2*l + 5)) }



def induction_tor(l, section, component, offdiag):
    '''
    The induction term ∇×(𝐁₀×𝐮) of a toroidal B0 in the rows f (r² 𝐫⋅) and g (r² 𝐫⋅∇×), columns upol (P) or utor (T) of
    degree l + offdiag: [ block, column block offset ], 0 for no coupling. Degree 2: induction_tor2.
    '''
    if par.B0_tor_l == 2:
        return induction_tor2(l, section, component, offdiag)
    out = 0
    m = par.m
    L = l*(l+1)
    Cm = np.sqrt(l**2 - m**2)/(2*l - 1)
    Cp = np.sqrt((l+1)**2 - m**2)/(2*l + 3)
    col = 'P' if component == 'upol' else 'T'

    if section == 'f' and component == 'upol' and offdiag == 0:
        out = -1j*m*L*rbtoIf
    elif section == 'g' and component == 'upol' and offdiag == -1:
        out = -Cm*l*(l-1)*(l+1)*( btoIg - rbto1Ig )
    elif section == 'g' and component == 'upol' and offdiag == 1:
        out = Cp*l*(l+1)*(l+2)*( btoIg - rbto1Ig )
    elif section == 'g' and component == 'utor' and offdiag == 0:
        out = -1j*m*L*rbtoIg

    return [ out, moff(section, col, offdiag) ]



def induction_tor2(l, section, component, offdiag):
    '''
    As induction_tor, for a toroidal B0 of degree 2 (quad_factors times polynomials in l; the same-degree terms carry
    l(l+1) - 3m²)
    '''
    out = 0
    m = par.m
    L = l*(l+1)
    A = quad_factors(l, m).get(offdiag, 0)
    col = 'P' if component == 'upol' else 'T'

    if section == 'f' and component == 'upol' and offdiag in [-1, 1]:
        out = A*3*(l*(l-1) if offdiag == -1 else (l+1)*(l+2))*rbtoIf
    elif section == 'g' and component == 'upol':
        if offdiag == -2:
            out = A*( 3*l*(l-2)*(l+1)*btoIg - 3*(l-2)*(l-1)*(l+1)*rbto1Ig + 3*(l-2)*(l+1)*rbtoD1g )
        elif offdiag == 0:
            out = A*(L - 3*m**2)*( -3*L*btoIg - 3*L*rbto1Ig - 6*L*rbtoD1g )
        elif offdiag == 2:
            out = A*( -3*l*(l+1)*(l+3)*btoIg + 3*l*(l+2)*(l+3)*rbto1Ig + 3*l*(l+3)*rbtoD1g )
    elif section == 'g' and component == 'utor' and offdiag in [-1, 1]:
        out = A*3*L*rbtoIg

    return [ -out, moff(section, col, offdiag) ]



def lorentz_tor(l, section, component, offdiag):
    '''
    Minus the Lorentz force, -[(∇×𝐛)×𝐁₀ + (∇×𝐁₀)×𝐛] (the A side, as coriolis is 2ẑ×𝐮), of a toroidal B0 in the rows
    u (r⁴ 𝐫⋅∇×∇×) and v (r² 𝐫⋅∇×), columns bpol (F) or
    btor (G) of degree l + offdiag: [ (OmgTau*Le)² × block, column block offset ]. Degree 2: lorentz_tor2.
    '''
    if par.B0_tor_l == 2:
        return lorentz_tor2(l, section, component, offdiag)
    out = 0
    m = par.m
    L = l*(l+1)
    Cm = np.sqrt(l**2 - m**2)/(2*l - 1)
    Cp = np.sqrt((l+1)**2 - m**2)/(2*l + 3)
    col = 'F' if component == 'bpol' else 'G'

    if section == 'u' and component == 'bpol' and offdiag == 0:
        out = -1j*m*( (L**2 - 2*L - 2)*rbtoIu + 2*r2bto1Iu - 2*(L-1)*r2btoD1u + L*r3bto2Iu + 2*r3bto1D1u - (L-2)*r3btoD2u )
    elif section == 'u' and component == 'btor' and offdiag == -1:
        out = -Cm*( 2*l*(l**2-1)*r2btoIu - 2*(l**2-1)*r3bto1Iu - 2*(l**2-1)*r3btoD1u )
    elif section == 'u' and component == 'btor' and offdiag == 1:
        out = Cp*( 2*l*(l+1)*(l+2)*r2btoIu + 2*l*(l+2)*r3bto1Iu + 2*l*(l+2)*r3btoD1u )
    elif section == 'v' and component == 'bpol' and offdiag == -1:
        out = -Cm*( (l-2)*(l**2-1)*btoIv + l*(l**2-1)*rbto1Iv - 2*(l**2-1)*rbtoD1v )
    elif section == 'v' and component == 'bpol' and offdiag == 1:
        out = Cp*( l*(l+2)*(l+3)*btoIv + l*(l+1)*(l+2)*rbto1Iv + 2*l*(l+2)*rbtoD1v )
    elif section == 'v' and component == 'btor' and offdiag == 0:
        out = -1j*m*(L-2)*rbtoIv

    return [ (par.OmgTau*par.Le)**2 * out, moff(section, col, offdiag) ]



def lorentz_tor2(l, section, component, offdiag):
    '''
    As lorentz_tor, for a toroidal B0 of degree 2 (fitted as induction_tor2)
    '''
    out = 0
    m = par.m
    L = l*(l+1)
    A = quad_factors(l, m).get(offdiag, 0)
    col = 'F' if component == 'bpol' else 'G'

    if section == 'u' and component == 'bpol' and offdiag in [-1, 1]:
        if offdiag == -1:
            c = [ 3*(l**4 - 3*l**2 - 6), -6*(l-3), 3*l*(l-1), -6*(l-1)*(l+3), -6*(l-3), -3*(l-2)*(l+3) ]
        else:
            c = [ 3*(l**4 + 4*l**3 + 3*l**2 - 2*l - 8), 6*(l+4), 3*(l+1)*(l+2), -6*(l-2)*(l+2), 6*(l+4), -3*(l-2)*(l+3) ]
        out = A*( c[0]*rbtoIu + c[1]*r2bto1Iu + c[2]*r3bto2Iu + c[3]*r2btoD1u + c[4]*r3bto1D1u + c[5]*r3btoD2u )
    elif section == 'u' and component == 'btor' and offdiag in [-2, 0, 2]:
        c = { -2: [ 6*l*(l-2)*(l+1), -6*(l-2)*(l+1), -6*(l-2)*(l+1) ],
               0: [ -6*L*(L - 3*m**2), -6*(2*L-3)*(L - 3*m**2), -6*(2*L-3)*(L - 3*m**2) ],
               2: [ -6*l*(l+1)*(l+3), -6*l*(l+3), -6*l*(l+3) ] }[offdiag]
        out = A*( c[0]*r2btoIu + c[1]*r3bto1Iu + c[2]*r3btoD1u )
    elif section == 'v' and component == 'bpol' and offdiag in [-2, 0, 2]:
        c = { -2: [ 3*(l-4)*(l-2)*(l+1), 3*(l-2)*(l-1)*(l+1), -9*(l-2)*(l+1) ],
               0: [ -3*(L-6)*(L - 3*m**2), 3*L*(L - 3*m**2), -6*(L-3)*(L - 3*m**2) ],
               2: [ -3*l*(l+3)*(l+5), -3*l*(l+2)*(l+3), -9*l*(l+3) ] }[offdiag]
        out = A*( c[0]*btoIv + c[1]*rbto1Iv + c[2]*rbtoD1v )
    elif section == 'v' and component == 'btor' and offdiag in [-1, 1]:
        out = A*3*((l-3)*(l+2) if offdiag == -1 else (l-1)*(l+4))*rbtoIv

    return [ -(par.OmgTau*par.Le)**2 * out, moff(section, col, offdiag) ]



def _total(pol, tor, l, section, component, offdiag):
    '''
    The poloidal and the toroidal B0 parts of a coupling, added: [ block, offset ], block 0 if neither couples (the offsets
    agree where both do)
    '''
    out = pol(l, section, component, offdiag) if par.B0 is not None else [0, 0]
    if par.B0_tor is not None:
        t = tor(l, section, component, offdiag)
        if not np.isscalar(t[0]):
            out = t if np.isscalar(out[0]) else [ out[0] + t[0], out[1] ]
    return out



def lorentz(l, section, component, offdiag):  # NEW (2026-10-30): poloidal + toroidal B0
    return _total(lorentz_pol, lorentz_tor, l, section, component, offdiag)



def induction(l, section, component, offdiag):  # NEW (2026-10-30): poloidal + toroidal B0
    return _total(induction_pol, induction_tor, l, section, component, offdiag)



def magnetic_diffusion(l, section, component, offdiag):
    '''
    The magnetic diffusion term −Eₘ ∇×(η∇×𝐛)
    '''
    cdipole = (par.B0 == 'dipole') and (par.ricb > 0)

    out = 0
    L= l*(l+1)

    if offdiag == 0:

        if section == 'f' and component == 'bpol':  #  -r²𝐫⋅∇×(η∇×𝐛)   (×r² if dipole)

            if not cdipole:
                #out = L*( -L*If + 2*r1D1f + r2D2f )
                out = L*( -L*etaIf + 2*retaD1f + r2etaD2f )
                #out = L*( -L*r0_eho0_D0 + 2*r1_eho0_D1 + r2_eho0_D2 )
            else:
                out = L*( -L*r2etaIf + 2*r3etaD1f + r4etaD2f )

        elif section == 'g' and component == 'btor':  # -r²𝐫⋅∇×(∇×(η∇×𝐛))  (×r³ if dipole)

            if not cdipole:
                #out = L*( -L*Ig + 2*r1D1g + r2D2g )
                out = L*( -L*etaIg + 2*retaD1g + r2etaD2g + reta1Ig + r2eta1D1g)
                #out = L*( -L*r0_eho0_D0 + 2*r1_eho0_D1 + r1_eho1_D0 + r2_eho0_D2 + r2_eho1_D1 )
            else:
                out = L*( -L*r3etaIg + 2*r4etaD1g + r5etaD2g + r4eta1Ig + r5eta1D1g )

        elif section == 'fic' and component == 'bpol_ic':  # for a conductive inner core

            out = L*( -L*r0_D0f_ic + 2*r1_D1f_ic + r2_D2f_ic )

        elif section == 'gic' and component == 'btor_ic':  # for a conductive inner core

            out = L*( -L*r0_D0g_ic + 2*r1_D1g_ic + r2_D2g_ic )

    if section in ['f', 'g']:
        out2 = out * par.Em
    elif section in ['fic', 'gic']:
        out2 = out * par.Em /( par.mu_i2o * par.sigma_i2o )

    return par.OmgTau * out2



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



def thermal_advection(l, section, component, offdiag):  # -u_r * dT/dr

    L = l*(l+1)
    rcmb = 1
    gap = rcmb - par.ricb

    if ((section == 'h') and (component == 'upol')) and (offdiag == 0) :

        if par.heating == 'internal':
            conv = r2Ih  # dT/dr = -beta*r. Heat equation is times r**2
        elif par.heating == 'differential':
            conv = Ih * par.ricb/gap  # dT/dr = -beta * r**2. Heat equation is times r**3
        elif par.heating == 'two zone' or par.heating == 'user defined':
            conv = NrIh  # dT/dr specified in ut.twozone or ut.BVprof. Heat equation is times r**2

    return L * conv



def thermal_diffusion(l, section, component, offdiag):

    L = l*(l+1)

    if section == 'h' and offdiag == 0 :

        if par.heating == 'differential':
            difus = - L*r1Ih + 2*r2D1h + r3D2h  # eq. times r**3
        else:
            difus = - L*Ih + 2*r1D1h + r2D2h  # eq. times r**2

    return difus * par.OmgTau * par.Ek / par.Prandtl



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------- Composition equation operators
# ----------------------------------------------------------------------------------------------------------------------



def composition(l, section, component, offdiag):

    out = 0
    if (section == 'i') and (offdiag == 0) :

        if par.comp_background == 'differential' :
            out = r3Ii
        else:
            out = r2Ii

    return out



def compositional_advection(l, section, component, offdiag):

    L = l*(l+1)
    rcmb = 1
    gap = rcmb - par.ricb

    if ((section == 'i') and (component == 'upol')) and (offdiag == 0) :

        if par.comp_background == 'differential':
            conv = Ii * par.ricb/gap  # Composition eq. times r**3
        else:
            conv = r2Ii  # Composition eq. times r**2

    return L * conv



def compositional_diffusion(l, section, component, offdiag):

    L = l*(l+1)

    if section == 'i' and offdiag == 0 :

        if par.comp_background == 'differential' :
            difus = - L*r1Ii + 2*r2D1i + r3D2i  # eq. times r**3
        else:
            difus = - L*Ii + 2*r1D1i + r2D2i  # eq. times r**2

    return difus * par.OmgTau * par.Ek / par.Schmidt


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

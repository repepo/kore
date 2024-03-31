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



def u(l, section, component, offdiag):  # ------------------------------------------------------------------- velocity u

    out = 0

    if offdiag == 0:

        L = l*(l+1)

        if section == 'u' and component == 'upol':

            if (par.magnetic == 1 and par.B0 == 'dipole'):
                out = L*( L*r4_D0_u - 2*r5_D1_u - r6_D2_u )  # r6* r.2curl(u)   r⁷ r̂⋅∇×∇×𝐮
            else:
                out = L*( L*r2_D0_u - 2*r3_D1_u - r4_D2_u )  # r4* r.2curl(u)   r⁵ r̂⋅∇×∇×𝐮

        elif section == 'v' and component == 'utor':

            if (par.magnetic == 1 and par.B0 == 'dipole'):
                out = L*r5_D0_v                          # r5* r.1curl(u)    r⁶ r̂⋅∇×𝐮
            else:
                out = L*r2_D0_v                          # r2* r.1curl(u)    r³ r̂⋅∇×𝐮

    return out



def coriolis(l, section, component, offdiag):  # ------------------------------------------------- Coriolis force 2z x u

    out  = 0
    offd = 0
    L = l*(l+1)

    if section == 'u':  # ------------------------------------------------------- 2curl

        if component == 'upol':

            if offdiag == 0:

                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2j*par.m*( -L*r4_D0_u + 2*r5_D1_u + r6_D2_u )  # r6* r.2curl(2z x u)
                else:
                    out = 2j*par.m*( -L*r2_D0_u + 2*r3_D1_u + r4_D2_u )  # r4* r.2curl(2z x u)

        elif component == 'utor':

            if offdiag == -1:

                C = (l**2-1)*np.sqrt(l**2-par.m**2) / (2*l-1.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( (l-1)*r5_D0_u - r6_D1_u )              # r6* r.2curl(2z x u)
                else:
                    out = 2*C*( (l-1)*r3_D0_u - r4_D1_u )              # r4* r.2curl(2z x u)

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 1:

                C = l*(l+2.)*np.sqrt((l+par.m+1.)*(l-par.m+1)) / (2.*l+3.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( -(l+2)*r5_D0_u - r6_D1_u )             # r6* r.2curl(2z x u)
                else:
                    out = 2*C*( -(l+2)*r3_D0_u - r4_D1_u )             # r4* r.2curl(2z x u)

                if ut.symm1 == -1:
                    offd = 1

    if section == 'v':  # ------------------------------------------------------- 1curl

        if component == 'upol':

            if offdiag == -1:

                C = (l**2-1)*np.sqrt(l**2-par.m**2) / (2*l-1.)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( (l-1)*r4_D0_v - r5_D1_v )              # r5* r.1curl(2z x u)
                else:
                    out = 2*C*( (l-1)*r1_D0_v - r2_D1_v )              # r2* r.1curl(2z x u)

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 1:

                C = l*(l+2)*np.sqrt((l+par.m+1.)*(l-par.m+1)) / (2*l+3)
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = 2*C*( -(l+2)*r4_D0_v - r5_D1_v )             # r5* r.1curl(2z x u)
                else:
                    out = 2*C*( -(l+2)*r1_D0_v - r2_D1_v )             # r2* r.1curl(2z x u)

                if ut.symm1 == 1:
                    offd = 1

        elif component == 'utor':

            if offdiag == 0:

                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = -2j*par.m*r5_D0_v                          # r5* r.1curl(2z x u)
                else:
                    out = -2j*par.m*r2_D0_v                          # r2* r.1curl(2z x u)

    return [ par.Gaspard * out, offd ]



def poincare(l, section, component, offdiag):  # Coming soon

    out = 0

    return out



def viscous_diffusion(l, section, component, offdiag):  # ----------------------------------- viscous force Ek*nabla^2 u

    out = 0
    L= l*(l+1)

    inviscid = (par.Ek == 0)

    if (offdiag == 0)&(not inviscid):

        if section == 'u' and component == 'upol':

            if par.anelastic:
                out = L * ( (-L*(l+2)*(l-1)*r0_D0_u - (L+2)*r1_lho1_D0_u - 2*(L-1)*r2_lho2_D0_u + r3_lho3_D0_u)
                           -(L-2)*r2_lho1_D1_u + 6*r3_lho2_D1_u + r4_lho3_D1_u
                           + 2*L*r2_D2_u + 5*r3_lho1_D2_u + 2*r4_lho2_D2_u
                           - 4*r3_D3_u + r4_lho1_D3_u
                           - r4_D4_u )

                if par.variable_viscosity:

                    out = L * ( -L*(l+2)*(l-1)*r0_vsc0_D0_u - (L+2)*r1_vsc0_lho1_D0_u - 2 * (L+1) * r1_vsc1_D0_u
                                -2*(L-1)*(r2_vsc0_lho2_D0_u + r2_vsc1_lho1_D0_u) - (l+2)*(l-1)*r2_vsc2_D0_u
                                + r3_vsc0_lho3_D0_u + 2*r3_vsc1_lho2_D0_u + r3_vsc2_lho1_D0_u

                                + (2-L)*r2_vsc0_lho1_D1_u + 6 * (r3_vsc0_lho2_D1_u + r3_vsc1_lho1_D1_u)
                                + r4_vsc0_lho3_D1_u + 2*r4_vsc1_lho2_D1_u + r4_vsc2_lho1_D1_u

                                + 2*L*r2_vsc0_D2_u + 5*r3_vsc0_lho1_D2_u - 4*r3_vsc1_D2_u + 2*r4_vsc0_lho2_D2_u
                                + 2*r4_vsc1_lho1_D2_u - r4_vsc2_D2_u

                                -4*r3_vsc0_D3_u + r4_vsc0_lho1_D3_u - 2*r4_vsc1_D3_u

                                - r4_vsc0_D4_u
                                )
            else:

                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = L*( -L*(l+2)*(l-1)*r2_D0_u + 2*L*r4_D2_u - 4*r5_D3_u - r6_D4_u )  # r6* r.2curl( nabla^2 u )
                else:
                    out = L*( -L*(l+2)*(l-1)*r0_D0_u + 2*L*r2_D2_u - 4*r3_D3_u - r4_D4_u )    # r4* r.2curl( nabla^2 u )

        elif section == 'v' and component == 'utor':

            if par.anelastic:
                out = L * ( -L*r0_D0_v - 3*r1_lho1_D0_v - r2_lho2_D0_v
                            + 2*r1_D1_v-r2_lho1_D1_v
                            +r2_D2_v)

                if par.variable_viscosity:

                    out = L * ( -L*r0_vsc0_D0_v - 3*r1_vsc0_lho1_D0_v - r2_vsc1_lho1_D0_v

                            + 2*r1_vsc0_D1_v - r2_vsc0_lho1_D1_v + r2_vsc1_D1_v

                            + r2_vsc0_D2_v
                            )
            else:
                if (par.magnetic == 1 and par.B0 == 'dipole'):
                    out = L*( -L*r3_D0_v + 2*r4_D1_v + r5_D2_v )                          # r5* r.1curl( nabla^2 u )
                else:
                    out = L*( -L*r0_D0_v + 2*r1_D1_v + r2_D2_v )                            # r2* r.1curl( nabla^2 u )

    return par.ViscosD * out



def lorentz(l, section, component, offdiag):  # ---------------------------------------------------------- Lorentz force

    out = 0
    offd = 0
    m = par.m
    L = l*(l+1)
    cdipole = ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0))  # boolean

    if section == 'u':    # ------------------------------------------------------- 2curl, section u

        if component == 'bpol':


            if offdiag == -2:  # (l-2) terms  (quadrupole)

                C = (3*(-2 - l + l**2)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)
                out1 = r0_h0_D0_u*(2*l + 3*l**2 + l**3) - r1_h0_D1_u*(6 - 7*l + 3*l**2) + r1_h1_D0_u*(2 + l - 6*l**2 + l**3)
                out2 = - r2_h0_D2_u*(-6 + l) - 2*r2_h1_D1_u*(-2 + l) + r2_h2_D0_u*(-2 + l) - r3_h2_D1_u + 3*r3_h0_D3_u
                out3 = - r3_h1_D2_u*(-3 + l) + r3_h3_D0_u*(-1 + l)
                out = C*(out1+out2+out3)

                offd = -1


            elif offdiag == -1:  # (l-1) terms

                C = np.sqrt(l**2-par.m**2)*(l**2-1)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :  # r4*r.2curl
                    out1 = -2*(l**2+2)*r1_h0_D1_u -2*(l-2)*r2_h1_D1_u - (l-4)*r2_h0_D2_u - (l-2)*r3_h1_D2_u
                    out2 = L*(l+2)*r0_h0_D0_u + L*(l-4)*r1_h1_D0_u + l*r2_h2_D0_u + l*r3_h3_D0_u + 2*r3_h0_D3_u
                elif par.B0 == 'dipole':
                    # same but +r2
                    out1 = -2*(l**2+2)*r3_h0_D1_u -2*(l-2)*r4_h1_D1_u - (l-4)*r4_h0_D2_u - (l-2)*r5_h1_D2_u
                    out2 = L*(l+2)*r2_h0_D0_u + L*(l-4)*r3_h1_D0_u + l*r4_h2_D0_u + l*r5_h3_D0_u + 2*r5_h0_D3_u
                out = C*(out1+out2)

                if ut.symm1 == 1:
                    offd = -1


            elif offdiag == 0:  # l terms  (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l + 4*l**2)
                out = C*( 3*r0_h0_D0_u*l*(1 + l)*(-2 + l + l**2) - 3*r1_h1_D0_u*L**2 + 2*r1_h0_D1_u*(6 - 4*l - 5*l**2 - 2*l**3 - l**4) \
                          + 3*r2_h2_D0_u*L + r2_h0_D2_u*(-12 + 5*l + 5*l**2) + 2*r2_h1_D1_u*(-6 + 5*l + 5*l**2) + 2*r3_h2_D1_u*L \
                          + r3_h3_D0_u*L + 2*r3_h0_D3_u*(-3 + l + l**2) + 3*r3_h1_D2_u*(-2 + l + l**2) )


            elif offdiag == 1:  # (l+1) terms

                C = np.sqrt((1+l+par.m)*(1+l-par.m))*l*(l+2)/(2*l+3)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :  # r4*r.2curl
                    out1 = -2*(l**2+2*l+3)*r1_h0_D1_u + 2*(l+3)*r2_h1_D1_u + (l+5)*r2_h0_D2_u + (l+3)*r3_h1_D2_u
                    out2 = -L*(l-1)*r0_h0_D0_u - L*(l+5)*r1_h1_D0_u - (l+1)*r2_h2_D0_u - (l+1)*r3_h3_D0_u + 2*r3_h0_D3_u
                elif par.B0 == 'dipole':
                    # same but +r2
                    out1 = -2*(l**2+2*l+3)*r3_h0_D1_u + 2*(l+3)*r4_h1_D1_u + (l+5)*r4_h0_D2_u + (l+3)*r5_h1_D2_u
                    out2 = -L*(l-1)*r2_h0_D0_u - L*(l+5)*r3_h1_D0_u - (l+1)*r4_h2_D0_u - (l+1)*r5_h3_D0_u + 2*r5_h0_D3_u
                out = C*(out1+out2)

                if ut.symm1 == -1:
                    offd = 1


            elif offdiag == 2:  # (l+2) terms  (quadrupole)

                C = (3*l*(3 + l)*np.sqrt((1 + l - m)*(2 + l + m))*np.sqrt(2 + 3*l + l**2 + m - m**2))/(15 + 16*l + 4*l**2)
                out = C*( r0_h0_D0_u*(l - l**3) - r1_h0_D1_u*(16 + 13*l + 3*l**2) - r1_h1_D0_u*(6 + 16*l + 9*l**2 + l**3) + 2*r2_h1_D1_u*(3 + l) \
                          - r2_h2_D0_u*(3 + l) + r2_h0_D2_u*(7 + l) - r3_h2_D1_u + 3*r3_h0_D3_u + r3_h1_D2_u*(4 + l) - r3_h3_D0_u*(2 + l) )

                offd = 1


        elif component == 'btor':


            if offdiag == -1:  # (l-1) terms  (quadrupole)

                C = (6j*m*np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( -r1_h0_D0_u*(3-3*l-2*l**2) - r2_h0_D1_u*(l-3) - r2_h1_D0_u*(-3+2*l+l**2) + 3*r3_h0_D2_u - r3_h1_D1_u*(l-3) - r3_h2_D0_u*l )

                if ut.symm1 == 1:
                    offd = -1


            elif offdiag == 0:  # l terms

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :  # r4*r.2curl
                    out = 2j*par.m*( -r1_h0_D0_u - (l**2+l-1)*r2_h1_D0_u + r2_h0_D1_u + r3_h1_D1_u + r3_h0_D2_u )
                elif par.B0 == 'dipole':
                    # same but +r2
                    out = 2j*par.m*( -r3_h0_D0_u - (l**2+l-1)*r4_h1_D0_u + r4_h0_D1_u + r5_h1_D1_u + r5_h0_D2_u )


            elif offdiag == 1:  # (l+1) terms  (quadrupole)

                C = (6j*m*np.sqrt(1 + 2*l + l**2 - m**2))/(3 + 2*l)
                out = C*( r1_h0_D0_u*(-4+l+2*l**2) + r2_h0_D1_u*(4+l) - r2_h1_D0_u*(-4+l**2) + 3*r3_h0_D2_u + r3_h2_D0_u*(1+l) + r3_h1_D1_u*(4+l) )

                if ut.symm1 == -1:
                    offd = 1


    elif section == 'v':  # ------------------------------------------------------- 1curl, section v


        if component == 'bpol':


            if offdiag == -1:  # (l-1) terms  (quadrupole)

                C = (3j*m*np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( 12*r0_h0_D1_v - 2*r0_h1_D0_v*(-1 + l)*l + 6*r1_h0_D2_v - r1_h2_D0_v*(-1 + l)*l )

                if ut.symm1 == -1:
                    offd = -1


            elif offdiag == 0:  # l terms (dipole)

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = 1j*par.m*( 4*r0_h0_D1_v - L*( 2*r0_h1_D0_v + r1_h2_D0_v ) + 2*r1_h0_D2_v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = 1j*par.m*( 4*r3_h0_D1_v - L*( 2*r3_h1_D0_v + r4_h2_D0_v ) + 2*r4_h0_D2_v )  # r5*r.1curl


            elif offdiag == 1:  # (l+1) terms  (quadrupole)

                C = (3j*m*np.sqrt((1 + l - m)*(1 + l + m)))/(3 + 2*l)
                out = C*( 12*r0_h0_D1_v - 2*r0_h1_D0_v*(1 + l)*(2 + l) + 6*r1_h0_D2_v - r1_h2_D0_v*(1 + l)*(2 + l) )

                if ut.symm1 == 1:
                    ofd = 1


        elif component == 'btor':


            if offdiag == -2:  # (l-2) terms  (quadrupole)

                C = (3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)
                out = C*( r0_h0_D0_v*(-4 + l) - 3*r1_h0_D1_v + r1_h1_D0_v*(-1 + l) )

                offd = -1


            elif offdiag == -1:

                C = np.sqrt((l-par.m)*(l+par.m))*(l**2-1)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( (l-2)*r0_h0_D0_v + l*r1_h1_D0_v -2*r1_h0_D1_v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l-2)*r3_h0_D0_v + l*r4_h1_D0_v -2*r4_h0_D1_v )  # r5*r.1curl

                if ut.symm1 == -1:
                    offd = -1


            elif offdiag == 0:  # l terms  (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))
                out = C*( r0_h0_D0_v*(6 - l - l**2) + r1_h1_D0_v*L - 2*r1_h0_D1_v*(-3 + l + l**2) )


            elif offdiag == 1:

                C = -np.sqrt((l+par.m+1)*(l+1-par.m))*l*(l+2)/(2*l+3)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( (l+3)*r0_h0_D0_v + (l+1)*r1_h1_D0_v + 2*r1_h0_D1_v )  # r2*r.1curl
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l+3)*r3_h0_D0_v + (l+1)*r4_h1_D0_v + 2*r4_h0_D1_v )  # r5*r.1curl

                if ut.symm1 == 1:
                    offd = 1


            elif offdiag == 2:  # (l+2) terms  (quadrupole)

                C = (3*l*(3 + l)*np.sqrt((2 + l - m)*(1 + l + m))*np.sqrt((1 + l - m)*(2 + l + m)))/((3 + 2*l)*(5 + 2*l))
                out = C*( -r0_h0_D0_v*(5 + l) - 3*r1_h0_D1_v - r1_h1_D0_v*(2 + l) )

                offd = 1


    return [ par.Hendrik * out, offd ]



def buoyancy(l, section, component, offdiag):  # -------------------------------------------------------- buoyancy force

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        if par.anelastic:
            #buoy = r3_buo0_D0_u
            buoy = r3_rog0_D0_u
            #buoy = -1 * r3_bvs0_D0_u
            
        else:
            if (par.magnetic == 1) and (par.B0 == 'dipole') :
                buoy = r6_D0_u
            else:
                buoy = r4_D0_u

    out = L * buoy

    return par.Beyonce * out



def comp_buoyancy(l, section, component, offdiag):  # ------------------------------------- compositional buoyancy force

    out = 0
    L = l*(l+1)

    if (section == 'u') and (offdiag == 0) :

        if (par.magnetic == 1) and (par.B0 == 'dipole') :
            buoy = r6_D0_u
        else:
            buoy = r4_D0_u

    out = L * buoy

    return par.OmgTau**2 * par.BV2_comp * out



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
                if not par.anelastic:
                    out = L* r2_D0_f
                else:
                    out = L* r2_rho0_D0_f
            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                if not par.anelastic:
                    out = L* r4_D0_f
                else:
                    out = L* r4_rho0_D0_f

        elif section == 'g' and component == 'btor':  # r² 𝐫⋅∇×𝐛   (×r³ if dipole)
            if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                if not par.anelastic:
                    out = L* r2_D0_g
                else:
                    out = L* r2_rho0_D0_g
            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                if not par.anelastic:
                    out = L* r5_D0_g
                else:
                    out = L* r5_rho0_D0_g

    return out



def induction(l, section, component, offdiag):
    '''
    The induction term ∇×(𝐁₀×𝐮)
    '''

    out = 0
    offd = 0
    m = par.m
    l = np.float128(l)  # to avoid overflow errors at high N
    L = l*(l+1)
    cdipole = ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0))  # boolean


    if section == 'f':  # ---------------------------------------------------- nocurl  r² 𝐫⋅∇×(𝐁₀×𝐮)  (×r² if classic dipole)

        if component == 'upol':

            if offdiag == -2:  # l-2 terms (quadrupole)

                C1 = 3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m))
                C2 = 3 - 8*l + 4*l**2
                out = ( r0_h0_D0_f*(-4 + l) - 3*r1_h0_D1_f + r1_h1_D0_f*(-1 + l) )*C1/C2

                offd = -1

            elif offdiag == -1:  # l-1 terms (dipole)

                C = np.sqrt(l**2-par.m**2)*(l**2-1)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1'] :
                    out = C*( (l-2)*r0_h0_D0_f + l*r1_h1_D0_f - 2*r1_h0_D1_f )
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( (l-2)*r2_h0_D0_f + l*r3_h1_D0_f - 2*r3_h0_D1_f )

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 0:  # l terms (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))
                out = C*( r0_h0_D0_f*(6 - l - l**2) + r1_h1_D0_f*l*(1 + l) - 2*r1_h0_D1_f*(-3 + l + l**2) )

            elif offdiag == 1:  # l+1 terms (dipole)

                C = np.sqrt((l+1)**2-par.m**2)*l*(l+2)/(2*l+3)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( -(l+3)*r0_h0_D0_f -(l+1)*r1_h1_D0_f -2*r1_h0_D1_f )
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( -(l+3)*r2_h0_D0_f -(l+1)*r3_h1_D0_f -2*r3_h0_D1_f )

                if ut.symm1 == 1:
                    offd = 1

            elif offdiag == 2:  # l+2 terms (quadrupole)

                C1 = 3*l*(3 + l)*np.sqrt((2 + l - m)*(1 + l + m))*np.sqrt((1 + l - m)*(2 + l + m))
                C2 = (3 + 2*l)*(5 + 2*l)
                out = ( r0_h0_D0_f*(-5 - l) - 3*r1_h0_D1_f - r1_h1_D0_f*(2 + l) )*C1/C2

                offd = 1

        elif component == 'utor':

            if offdiag == -1:  # l-1 terms (quadrupole)

                out = 18j * r1_h0_D0_f * m * np.sqrt(l**2-m**2)/(1-2*l)

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 0:  # l terms (dipole)

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = -2j*m* r1_h0_D0_f
                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = -2j*m* r3_h0_D0_f

            elif offdiag == 1:  # l+1 terms (quadrupole)

                out = -18j * r1_h0_D0_f * m * np.sqrt((1+l-m)*(1+l+m))/(3+2*l)

                if ut.symm1 == -1:
                    offd = 1


    elif section == 'g':  # --------------------------------------------- 1curl  r² 𝐫⋅∇×( ∇×(𝐁₀×𝐮) )  (×r³ if dipole)

        if component == 'upol':

            if offdiag == -1:  # l-1 terms (quadrupole)

                C = (3j * m * np.sqrt(l**2 - m**2))/(-1 + 2*l)
                out = C*( -2*r0_h1_D0_g*(-3 + l) - 2*r0_h0_D1_g*(-3 + l) - 2*q1_h0_D0_g*(3 + l**2) + 6*r1_h0_D2_g - 2*r1_h1_D1_g*(-3 + l) + r1_h2_D0_g*(-1 + l)*l )

                if ut.symm1 == -1:
                    offd = -1

            elif offdiag == 0:  # l terms (dipole)

                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = 2j*m*( r0_h0_D1_g + r1_h1_D1_g -(L+1)*q1_h0_D0_g + r0_h1_D0_g + (L/2)*r1_h2_D0_g + r1_h0_D2_g )  # qh=h/r
                    if par.anelastic:
                        out += 2j*m*( - (L/2)*r1_h1_lho1_D0_g - (1/2)*(L+2)*r0_h0_lho1_D0_g - r1_h0_lho1_D1_g )

                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = 2j*m*( r3_h0_D1_g + r4_h1_D1_g -(l**2+l+1)*r2_h0_D0_g + r3_h1_D0_g + (L/2)*r4_h2_D0_g + r4_h0_D2_g )
                    if par.anelastic:
                        out += 2j*m*( -(L/2)*r4_h1_lho1_D0_g - (1/2)*(l**2+l+2)*r3_h0_lho1_D0_g - r4_h0_lho1_D1_g)



            elif offdiag == 1:  # l+1 terms (quadrupole)

                C = (3j*m*np.sqrt((1 + l - m)*(1 + l + m)))/(3 + 2*l)
                out = C*( 2*(4+l)*( r0_h1_D0_g + r0_h0_D1_g + r1_h1_D1_g ) - 2*q1_h0_D0_g*(4 + 2*l + l**2) + 6*r1_h0_D2_g + r1_h2_D0_g*(2 + 3*l + l**2) )

                if ut.symm1 == 1:
                    offd = 1

        elif component == 'utor':

            if offdiag == -2:  # l-2 terms (quadrupole)

                C = (3*(-2 + l)*(1 + l)*np.sqrt((l - m)*(-1 + l + m))*np.sqrt((-1 + l - m)*(l + m)))/(3 - 8*l + 4*l**2)
                out = C*( r0_h0_D0_g*l - 3*r1_h0_D1_g + r1_h1_D0_g*(-3 + l) )

                offd = -1

            elif offdiag == -1:  # l-1 terms (dipole)

                C = (l**2-1)*np.sqrt( l**2-m**2)/(2*l-1)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C*( l* r0_h0_D0_g -2* r1_h0_D1_g + (l-2)* r1_h1_D0_g )
                    if par.anelastic:
                        out += 2*C* r1_h0_lho1_D0_g

                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C*( l* r3_h0_D0_g -2* r4_h0_D1_g + (l-2)* r4_h1_D0_g )
                    if par.anelastic:
                        out += 2*C* r4_h0_lho1_D0_g

                if ut.symm1 == 1:
                    offd = -1

            elif offdiag == 0:  # l terms (quadrupole)

                C = (3*(l + l**2 - 3*m**2))/(-3 + 4*l*(1 + l))
                out = C*( -r0_h0_D0_g*L - 2*r1_h0_D1_g*(-3 + l + l**2) - 3*r1_h1_D0_g*(-2 + l + l**2) )

            elif offdiag == 1:  # l+1 terms

                C = l*(l+2)*np.sqrt((l+1)**2-m**2)/(3+2*l)
                if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :
                    out = C * ( -2* r1_h0_D1_g - (l+1)* r0_h0_D0_g - (l+3) * r1_h1_D0_g )
                    if par.anelastic:
                        out += 2*C* r1_h0_lho1_D0_g

                elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
                    out = C * ( -2* r4_h0_D1_g - (l+1)* r3_h0_D0_g - (l+3) * r4_h1_D0_g )
                    if par.anelastic:
                        out += 2*C* r4_h0_lho1_D0_g

                if ut.symm1 == -1:
                    offd = 1

            elif offdiag == 2:  # l+2 terms (quadrupole)

                C = (3*l*(3 + l)*np.sqrt(2 + 3*l + l**2 - m - m**2)*np.sqrt(2 + 3*l + l**2 + m - m**2))/(15 + 16*l + 4*l**2)
                out = C*( -r0_h0_D0_g*(1 + l) - 3*r1_h0_D1_g - r1_h1_D0_g*(4 + l) )

                offd = 1

    return [ out, offd ]



def magnetic_diffusion(l, section, component, offdiag):
    '''
    The magnetic difussion term Eₘ∇²𝐛
    '''

    out = 0
    L = l*(l+1)

    if offdiag == 0:

        if section == 'f' and component == 'bpol':  #  r² 𝐫⋅∇²𝐛

            if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :

                if par.anelastic:
                    out = L*( -L*r0_eho0_D0_f + 2*r1_eho0_D1_f + r2_eho0_D2_f )
                else:
                    out = L*( -L*r0_eta0_D0_f + 2*r1_eta0_D1_f + r2_eta0_D2_f )

            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :  # extra ×r² if dipole

                if par.anelastic:
                    out = L*( -L*r2_eho0_D0_f + 2*r3_eho0_D1_f + r4_eho0_D2_f )
                else:
                    out = L*( -L*r2_eta0_D0_f + 2*r3_eta0_D1_f + r4_eta0_D2_f )

        elif section == 'g' and component == 'btor':  # r² 𝐫⋅∇×(∇²𝐛)

            if par.B0 in ['axial', 'G21 dipole', 'FDM', 'Luo_S1', 'Luo_S2'] :

                if par.anelastic:
                    out = L*( 2*r1_eho0_D1_g - L* r0_eho0_D0_g + r2_eho0_D2_g + r1_eta1_rho0_D0_g + r2_eta1_rho0_D1_g )
                else:
                    out = L*( 2*r1_eta0_D1_g - L* r0_eta0_D0_g + r2_eta0_D2_g + r1_eta1_D0_g + r2_eta1_D1_g )

            elif ((par.B0 == 'dipole') and (par.ricb > 0)) :  # extra ×r^3 if dipole

                if par.anelastic:
                    out = L*( 2*r4_eho0_D1_g - L* r3_eho0_D0_g + r5_eho0_D2_g - r4_eta1_rho0_D0_g - r5_eta1_rho0_D1_g )
                else:
                    out = L*( 2*r4_eta0_D1_g - L* r3_eta0_D0_g + r5_eta0_D2_g - r4_eta1_D0_g - r5_eta1_D1_g )


    return par.MagnetD * out



# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------- Heat equation operators
# ----------------------------------------------------------------------------------------------------------------------



def theta(l, section, component, offdiag):
    '''
    This is the temperature perturbation in the Bussinesq case,
    or the specific entropy perturbation in the anelastic case. 
    '''

    if (section == 'h') and (offdiag == 0) :

        if par.anelastic:
            #out = r2_rho0_D0_h
            out = r2_roT0_D0_h
            #out = r1_D0_h
        else:
            if par.heating == 'differential' :
                out = r3_D0_h
            else:
                out = r2_D0_h

    return out



def thermal_advection(l, section, component, offdiag):  # -u_r * dT/dr

    out = 0
    L = l*(l+1)
    rcmb = 1
    gap = rcmb - par.ricb

    if ((section == 'h') and (component == 'upol')) and (offdiag == 0) :

        if par.anelastic:
            #conv = -r1_drS0_D0_h  # (r*S0')*D0s
            conv = -r1_tds0_D0_h
            #conv = r0_D0_h
           
        else:
            if par.heating == 'internal':
                conv = r2_D0_h  # dT/dr = -beta*r. Heat equation is times r**2
            elif par.heating == 'differential':
                conv = r0_D0_h * par.ricb/gap  # dT/dr = -beta * r**2. Heat equation is times r**3
            elif par.heating == 'two zone' or par.heating == 'user defined':
                conv = r0_drS0_D0_h  # dT/dr or dS/dr specified in rap.twozone or rap.BVprof. Heat equation is times r**2

        out = L * conv

    return out



def thermal_diffusion(l, section, component, offdiag):

    L = l*(l+1)

    if section == 'h' and offdiag == 0 :

        if not par.anelastic:

            if par.heating == 'differential':
                difus = - L*r1_D0_h + 2*r2_D1_h + r3_D2_h  # eq. times r**3
            else:
                difus = - L*r0_D0_h + 2*r1_D1_h + r2_D2_h  # eq. times r**2

        else:

            #difus = - L*r0_kho0_D0_h + 2*r1_kho0_D1_h + r2_kho0_D2_h + r2_kho0_lnT1_D1_h + r2_kho1_D1_h
            difus = -L*r0_krT0_D0_h + 2*r1_krT0_D1_h + r2_krT1_D1_h + r2_krT0_D2_h

    return difus * par.ThermaD



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------- Composition equation operators
# ----------------------------------------------------------------------------------------------------------------------



def composition(l, section, component, offdiag):

    out = 0
    if (section == 'i') and (offdiag == 0) :

        if par.comp_background == 'differential' :
            out = r3_D0_i
        else:
            out = r2_D0_i

    return out



def compositional_advection(l, section, component, offdiag):

    out = 0
    L = l*(l+1)
    rcmb = 1
    gap = rcmb - par.ricb

    if ((section == 'i') and (component == 'upol')) and (offdiag == 0) :

        if par.comp_background == 'differential':
            conv = r0_D0_i * par.ricb/gap  # Composition eq. times r**3
        else:
            conv = r2_D0_i  # Composition eq. times r**2

        out = L * conv

    return out



def compositional_diffusion(l, section, component, offdiag):

    L = l*(l+1)

    if section == 'i' and offdiag == 0 :

        if par.comp_background == 'differential' :
            difus = - L*r1_D0_i + 2*r2_D1_i + r3_D2_i  # eq. times r**3
        else:
            difus = - L*r0_D0_i + 2*r1_D1_i + r2_D2_i  # eq. times r**2


    return difus * par.OmgTau * par.Ek / par.Schmidt


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import scipy.sparse as ss
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
import utils4pp as upp
import radial_profiles as rap
from parameters import par   # FIX (2026-11-08): par is the object in parameters.py

'''
Computes the radial profile of kinetic energy, internal energy dissipation,
and kinetic energy dissipation (theta,phi integrated).

FIX (2026-11-08): the kinetic energy is ρ0 |u|² with the anelastic velocity, as spin_doctor's KE (it left out the
density), compared with KE (it was KE + KP, from an older flow.dat), in a full sphere too; from conductive_IC (9da232d),
it runs without %pylab.

Use as 
%run -i tools/plot_rad_con_tor_profile.py sol nR R1 R2
solnum is the solution muber (0 if forced problem)
nR     is the number of points in radius
R1     from radius R1
R2     to radius R2
'''


solnum = int(sys.argv[1])
nR = int(sys.argv[2]) # number of radial points
R1 = float(sys.argv[3]) # From radius R1
R2 = float(sys.argv[4]) # To radius R2

lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = ut.rcmb
n    = ut.n


# xk are the colocation points, from -1 to 1
i = np.arange(0,nR)
xk = np.cos( (i+0.5)*np.pi/nR )
# rk are the radial colocation points, from Ra to Rb
r = 0.5*(R2-R1)*xk + 0.5*(R1+R2)
x1 = ut.xcheb(r, ricb, rcmb)   # FIX (2026-11-08): the Chebyshev variable of the shell, or of [-rcmb, rcmb] in a full sphere
r2 = r**2
rho0 = np.exp(rap.logrhoX(r, 0))   # NEW (2026-11-08): the background density, 1 in a uniform fluid, and the derivatives
lho1, lho2, lho3 = [ rap.logrhoX(r, k) for k in (1, 2, 3) ]   # of its log, as utils4pp.cheb2space_pol uses them

Inv_r = ss.diags(1/r,0)



# matrix with Chebishev polynomials at every x point for all degrees:
chx = ch.chebvander(x1,par.N-1) # this matrix has nR rows and N-1 cols


u = np.loadtxt('flow.dat')          # flow data
if len(u.shape)==1:   
    u = u.reshape((-1,len(u)))
K  = u[:,0]                     # FIX (2026-11-08): KE, the kinetic energy (KP and KT are columns 2 and 3)

# w = np.loadtxt('eigenvalues.dat')
# if len(w.shape)==1:   
#     w = w.reshape((-1,len(w)))


_sol = ut.load_field('flow', solnum)   # NEW (2026-11-08): flow.npy or the text files
_sol = upp.expand_sol(_sol, symm)      # FIX (2026-11-08): N coefficients per l in a full sphere too (one parity is stored)
reu, imu = _sol.real, _sol.imag
n = int(_sol.size/2)                   # the poloidal part, N coefficients per l


Plj0 = reu[:n]    + 1j*imu[:n]      #  N elements on each l block
Tlj0 = reu[n:n+n] + 1j*imu[n:n+n]   #  N elements on each l block
Plj   = np.reshape(Plj0,(-1,N))
Tlj   = np.reshape(Tlj0,(-1,N))


# l-indices for u and b:
lmm = 2*np.shape(Plj)[0] -1 # this should be =lmax-m
s = int(symm*0.5+0.5) # s=0 if u is antisymm, s=1 if u is symm
if m>0:
    lup = np.arange( m+1-s, m+1-s +lmm, 2) # u pol
    lut = np.arange( m+s  , m+s   +lmm, 2) # u tor
elif m==0:
    lup = np.arange( 1+s, 1+s +lmm, 2) # u pol
    lut = np.arange( 2-s, 2-s +lmm, 2) # u tor


Lup = ss.diags(lup*(lup+1),0)
Lut = ss.diags(lut*(lut+1),0)
Inv_Lup = ss.diags(1/(lup*(lup+1)),0)
Inv_Lut = ss.diags(1/(lut*(lut+1)),0)


d1Plj = np.zeros(np.shape(Plj),dtype=complex)
d2Plj = np.zeros(np.shape(Plj),dtype=complex)
d3Plj = np.zeros(np.shape(Plj),dtype=complex)

d1Tlj = np.zeros(np.shape(Tlj),dtype=complex)
d2Tlj = np.zeros(np.shape(Tlj),dtype=complex)

P0 = np.zeros((int((lmm+1)/2), nR),dtype=complex)
P1 = np.zeros((int((lmm+1)/2), nR),dtype=complex)
P2 = np.zeros((int((lmm+1)/2), nR),dtype=complex)
P3 = np.zeros((int((lmm+1)/2), nR),dtype=complex)

T0 = np.zeros((int((lmm+1)/2), nR),dtype=complex)
T1 = np.zeros((int((lmm+1)/2), nR),dtype=complex)
T2 = np.zeros((int((lmm+1)/2), nR),dtype=complex)

Q0 = np.zeros((int((lmm+1)/2), nR),dtype=complex)
S0 = np.zeros((int((lmm+1)/2), nR),dtype=complex)


np.matmul( Plj, chx.T, P0 )
np.matmul( Tlj, chx.T, T0 )

for k in range(np.size(lup)):
    d1Plj[k,:] = ut.Dcheb(   Plj[k,:], ricb, rcmb)
    d2Plj[k,:] = ut.Dcheb( d1Plj[k,:], ricb, rcmb)
    d3Plj[k,:] = ut.Dcheb( d2Plj[k,:], ricb, rcmb)
np.matmul(d1Plj, chx.T, P1)
np.matmul(d2Plj, chx.T, P2)
np.matmul(d3Plj, chx.T, P3) 
    
for k in range(np.size(lup)):
    d1Tlj[k,:] = ut.Dcheb(   Tlj[k,:], ricb, rcmb)
    d2Tlj[k,:] = ut.Dcheb( d1Tlj[k,:], ricb, rcmb)
np.matmul(d1Tlj, chx.T, T1)
np.matmul(d2Tlj, chx.T, T2)

Q0 = Lup * P0 * Inv_r
Q1 = ( Lup * P1 -   Q0 ) * Inv_r
Q2 = ( Lup * P2 - 2*Q1 ) * Inv_r

# FIX (2026-11-08): the consoidal parts of ρ0𝐮 = ∇×∇×(ρ0 P 𝐫), as utils4pp.cheb2space_pol (they left out the density)
S0 = P1 + P0*(lho1 + 1/r)
S1 = P2 + P0*(lho2 - 1/r2) + P1*(lho1 + 1/r)
S2 = P3 + P0*(lho3 + 2/r**3) + 2*P1*(lho2 - 1/r2) + P2*(lho1 + 1/r)
   

ke_pol = np.zeros((int((lmm+1)/2), nR),dtype=complex)
ke_rad = np.zeros((int((lmm+1)/2), nR),dtype=complex)
ke_con = np.zeros((int((lmm+1)/2), nR),dtype=complex)
ke_tor = np.zeros((int((lmm+1)/2), nR),dtype=complex)

#di_pol = np.zeros((int((lmm+1)/2), nR),dtype=complex)
#di_tor = np.zeros((int((lmm+1)/2), nR),dtype=complex)

#dk_pol = np.zeros((int((lmm+1)/2), nR),dtype=complex)
#dk_tor = np.zeros((int((lmm+1)/2), nR),dtype=complex)


## Poloidal components
for k,l in enumerate(lup):
    
    L = l*(l+1)
    
    q0 = Q0[k,:]
    q1 = Q1[k,:]
    q2 = Q2[k,:] 
    
    s0 = S0[k,:]
    s1 = S1[k,:]
    s2 = S2[k,:]
    
    f0 = 4*np.pi/(2*l+1)    
        
    # kinetic energy
    f1 = rho0*r2*np.absolute( q0 )**2
    f2 = rho0*r2*L*np.absolute( s0 )**2
    ke_pol[k,:] = f0*(f1+f2)
    ke_rad[k,:] = f0*(f1)
    ke_con[k,:] = f0*(f2)
    
    # Dint: Internal energy dissipation 
    #f1 = L*np.absolute(q0 + r*s1 - s0)**2
    #f2 = 3*np.absolute(r*q1)**2
    #f3 = L*(l-1)*(l+2)*np.absolute(s0)**2
    #di_pol[k,:] = 2*f0*( f1+f2+f3 )
    
    # Dkin: Kinetic energy dissipation
    #f1 = L * r2 * np.conj(s0) * s2
    #f2 = 2 * r * L * np.conj(s0) * s1
    #f3 = -(L**2)*( np.conj(s0)*s0 ) - (l**2+l+2) * ( np.conj(q0)*q0 )
    #f4 = 2 * r * np.conj(q0)*q1 + r2 * np.conj(q0) * q2
    #f5 = 2 * L *( np.conj(q0)*s0 + q0*np.conj(s0) )
    #dk_pol[k,:] = 2*f0*( f1+f2+f3+f4+f5 )


    
## Toroidal components
for k,l in enumerate(lut):
    
    L = l*(l+1)
    
    t0 = T0[k,:]
    t1 = T1[k,:]
    t2 = T2[k,:]
    
    f0 = 4*np.pi/(2*l+1)
    
    # kinetic energy
    f1 = rho0*(r2)*L*np.absolute(t0)**2
    ke_tor[k,:] = f0*f1
    
    # Dint: Internal energy dissipation
    #f1 = L*np.absolute( r*t1-t0 )**2
    #f2 = L*(l-1)*(l+2)*np.absolute( t0 )**2
    #di_tor[k,:] = 2*f0*( f1+f2 )

    # Dkin: Kinetic energy dissipation
    #f1 = L * r2 * np.conj(t0) * t2 
    #f2 = 2 * r * L * np.conj(t0) * t1
    #f3 = -(L**2)*( np.conj(t0)*t0 )
    #dk_tor[k,:] = 2*f0*(f1+f2+f3) 

ke    = np.real(sum(ke_pol+ke_tor,0))
krad = np.real(sum(ke_rad,0))
kcon = np.real(sum(ke_con,0))
ktor = np.real(sum(ke_tor,0))

#dint = np.real(sum(di_pol+di_tor,0)*par.Ek)
#dkin = np.real(sum(dk_pol+dk_tor,0)*par.Ek)

totK = K[solnum]

# FIX (2026-11-08): Fejér's first rule on the Chebyshev points, exact for polynomials of degree < nR (np.trapz left out the
# ends, and numpy 2 renamed it)
th = (i + 0.5)*np.pi/nR
jw = np.arange(1, nR//2 + 1)
wk = (2/nR)*(1 - 2*np.sum(np.cos(2*np.outer(th, jw))/(4*jw**2 - 1), axis=1))
print('Estimated kinetic energy:', 0.5*(R2-R1)*np.sum(wk*ke))
print('   Actual kinetic energy:', K[solnum])



# plt.figure();

# plt.plot(r,ke/totK,'--',lw=1,color='gray', label=r'Total kinetic energy')
# plt.plot(r,krad/totK, label=r'Radial')
# plt.plot(r,kcon/totK, label=r'Consoidal')
# plt.plot(r,ktor/totK, label=r'Toroidal')

# plt.yscale('log')
# plt.xlabel(r'$r$',size=12)
# plt.legend()
# plt.tight_layout()
# plt.show()


fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(5,5))

ax[0].plot(r,ke/totK,'--',lw=1,color='gray', label=r'Total kinetic energy')
ax[0].plot(r,krad/totK, label=r'Radial')
ax[0].plot(r,kcon/totK, label=r'Consoidal')
ax[0].plot(r,ktor/totK, label=r'Toroidal')

ax[0].set_ylim(-1,18)
ax[0].set_xlabel(r'$r/R_\odot$',size=12)
ax[0].set_ylabel(r'$\left<K\right>_{\theta\phi}$',size=12)
ax[0].legend()


k = np.argmin(np.abs(r-0.99))

ax[1].plot(lup,np.abs(ke_rad[:,k])/ke[k],'.-',lw=0.3,label=r'Radial')
ax[1].plot(lup,np.abs(ke_con[:,k])/ke[k],'.-',lw=0.3,label=r'Consoidal')
ax[1].plot(lut,np.abs(ke_tor[:,k])/ke[k],'.-',lw=0.3,label=r'Toroidal')

ax[1].set_yscale('log')
ax[1].set_xlabel(r'Spherical harmonic degree $l$',size=12)
ax[1].set_ylabel(r'Spectral energy at $r=0.99R_\odot$',size=12)
ax[1].set_xlim(7,35)     
ax[1].set_ylim(1e-5,1) 
#ax[1].text(8,2e-5,r'$r=0.99R_\odot$',size=14) 


ax[1].legend()



plt.tight_layout()
plt.show()

import numpy as np
import scipy.sparse as ss
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
import parameters as par

'''
Computes the radial profile of kinetic energy, internal energy dissipation,
and kinetic energy dissipation (theta,phi integrated).

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
rcmb = 1
n    = ut.n


# xk are the colocation points, from -1 to 1
i = np.arange(0,nR)
xk = np.cos( (i+0.5)*np.pi/nR )
x1 = ( (R2-R1)*xk + (R1+R2) - (ricb+rcmb) )/(rcmb-ricb)
# rk are the radial colocation points, from Ra to Rb
r = 0.5*(rcmb-ricb)*( x1 + 1 ) + ricb
r2 = r**2

Inv_r = ss.diags(1/r,0)



# matrix with Chebishev polynomials at every x point for all degrees:
chx = ch.chebvander(x1,par.N-1) # this matrix has nR rows and N-1 cols


u = np.loadtxt('flow.dat')          # flow data
if len(u.shape)==1:   
    u = u.reshape((-1,len(u)))
KP = u[:,0]                     # Poloidal kinetic energy
KT = u[:,1]                     # Toroidal kinetic energy
K  = KP + KT

# w = np.loadtxt('eigenvalues.dat')
# if len(w.shape)==1:   
#     w = w.reshape((-1,len(w)))


reu = np.loadtxt('real_flow.field',usecols=solnum)
imu = np.loadtxt('imag_flow.field',usecols=solnum)


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

S0 = P1 + P0 * Inv_r
S1 = P2 + Inv_Lup * Q1
S2 = P3 + Inv_Lup * Q2
   

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
    f1 = r2*np.absolute( q0 )**2
    f2 = r2*L*np.absolute( s0 )**2
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
    f1 = (r2)*L*np.absolute(t0)**2
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

k = np.argsort(r)
tmp = ke
print('Estimated kinetic energy:', np.trapz(tmp[k],r[k]))
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


fig, ax = subplots(nrows=2,ncols=1,figsize=(5,5))

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



tight_layout()
show()

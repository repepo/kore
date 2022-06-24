import numpy as np
import scipy.sparse as ss
import numpy.polynomial.chebyshev as ch
import scipy.integrate as si
sys.path.insert(1,'bin/')
import utils as ut
import parameters as par

'''

'''

solnum = int(sys.argv[1])

lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = 1
n    = ut.n


gap = rcmb-ricb
r = loadtxt('rmesh.orig')[::4]
k = (r>=0.71)&(r<=0.985)
r = r[k]/0.985

nR = np.size(r)
theta = (pi/180)*( arange(49)*(15/8) )  
Ntheta = np.size(theta)

err = loadtxt('err2d.hmiv72d.ave')
err = err[k,:]
rot = loadtxt('rot2d.hmiv72d.ave')
rot = rot[k,:]

x = 2.*(r-ricb)/gap - 1. 

chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols

phi = 0. # select meridional cut

a = np.loadtxt('real_flow.field',usecols=solnum)
b = np.loadtxt('imag_flow.field',usecols=solnum)

if m > 0 :
    symm1 = symm
    if symm == 1:
        m_top = m
        m_bot = m+1                     # equatorially symmetric case (symm=1)
        lmax_top = lmax
        lmax_bot = lmax+1
    elif symm == -1:
        m_top = m+1
        m_bot = m               # equatorially antisymmetric case (symm=-1)
        lmax_top = lmax+1
        lmax_bot = lmax
elif m == 0 :
    symm1 = -symm 
    if symm == 1:
        m_top = 2
        m_bot = 1                       # equatorially symmetric case (symm=1)
        lmax_top = lmax+2
        lmax_bot = lmax+1
    elif symm == -1:
        m_top = 1
        m_bot = 2               # equatorially antisymmetric case (symm=-1)
        lmax_top = lmax+1
        lmax_bot = lmax+2


Plj0 = a[:n] + 1j*b[:n]         #  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n]   #  N elements on each l block

Plj  = np.reshape(Plj0,(int((lmax-m+1)/2),N))
Tlj  = np.reshape(Tlj0,(int((lmax-m+1)/2),N))
dPlj = np.zeros(np.shape(Plj),dtype=complex)

Plr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
dP  = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
rP  = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Qlr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Slr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Tlr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

np.matmul( Plj, chx.T, Plr )
np.matmul( Tlj, chx.T, Tlr )

rI = ss.diags(r**-1,0)

ll = np.arange(m_top,lmax_top,2)
L = ss.diags(ll*(ll+1),0)

for k in range(np.size(ll)):
    dPlj[k,:] = ut.Dcheb(Plj[k,:], ricb, rcmb)

np.matmul(dPlj, chx.T, dP)

rP  = Plr * ss.diags(r**-1,0)
Qlr = ss.diags(ll*(ll+1),0) * rP
Slr = rP + dP


m1 = max(m,1)
if m == 0 :
    lmax1 = lmax+1
else:
    lmax1 = lmax 
l1 = np.arange(m1,lmax1+1) # vector with all l's allowed whether for T or P


clm = np.zeros((lmax-m+2,1))
for i,l in enumerate(l1):
    clm[i] = np.sqrt((l-m)*(l+m))


# start index for l. Do not confuse with indices for the Cheb expansion!
idP = int( (1-symm1)/2 )
idT = int( (1+symm1)/2 )

plx = idP+lmax-m+1
tlx = idT+lmax-m+1

Wa = np.zeros((nR,Ntheta))
Wb = np.zeros((nR,Ntheta))

y0a = np.zeros_like(Wa)
y1a = np.zeros_like(Wa)
y00a = np.zeros_like(r)
y11a = np.zeros_like(r)

y0b = np.zeros_like(Wb)
y1b = np.zeros_like(Wb)
y00b = np.zeros_like(r)
y11b = np.zeros_like(r)

for kr in range(0,nR):

    for kt in range(Ntheta):
        
        s = r[kr]*sin(theta[kt])
        if (err[kr,kt]<0.9) and (s>0.22):

            ylm = np.r_[ut.Ylm_full(lmax, m, theta[kt], phi),0]
        
            ur2 = np.absolute(np.dot( Qlr[:,kr], ylm[idP:plx:2] ))**2        
    
            tmp1 = np.dot(   -(l1[idP:plx:2]+1) * Slr[:,kr]/np.tan(theta[kt]), ylm[idP:plx:2]     )
            tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,kr]/np.sin(theta[kt]), ylm[idP+1:plx+1:2] )
            tmp3 = np.dot(                 1j*m * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT:tlx:2]     )
            ut2 = np.absolute(tmp1+tmp2+tmp3)**2
            
            tmp1 = np.dot(     (l1[idT:tlx:2]+1) * Tlr[:,kr]/np.tan(theta[kt]), ylm[idT:tlx:2]     )
            tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT+1:tlx+1:2] )
            tmp3 = np.dot(                  1j*m * Slr[:,kr]/np.sin(theta[kt]), ylm[idP:plx:2]     )
            up2 = np.absolute(tmp1+tmp2+tmp3)**2
    
            Wa[kr,kt] = np.sqrt(ur2 + ur2 + up2)
            Wb[kr,kt] = ur2 + ur2 + up2
            
            y0a[kr,kt] = Wa[kr,kt] * (r[kr]**2)*np.sin(theta[kt])
            y0b[kr,kt] = Wb[kr,kt] * (r[kr]**2)*np.sin(theta[kt])
            
            y1a[kr,kt] = rot[kr,kt] * y0a[kr,kt]
            y1b[kr,kt] = rot[kr,kt] * y0b[kr,kt]
              
    k = Wa[kr,:]>0
    
    y00a[kr] = si.simpson( y0a[kr,k], theta[k], even='last' )
    y00b[kr] = si.simpson( y0b[kr,k], theta[k], even='last' )
    y11a[kr] = si.simpson( y1a[kr,k], theta[k], even='last' )
    y11b[kr] = si.simpson( y1b[kr,k], theta[k], even='last' )


z0a = si.simpson( y00a, r, even='last' )
za  = si.simpson( y11a, r, even='last' )

z0b = si.simpson( y00b, r, even='last' )
zb  = si.simpson( y11b, r, even='last' )

print('Average rotation, amplitude weighted = ', za/z0a)
print('Average rotation, energy weighted    = ', zb/z0b)


















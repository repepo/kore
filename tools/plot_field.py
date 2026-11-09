import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch
import multiprocessing as mp

sys.path.insert(1,'bin/')

import utils as ut
from parameters import par
import utils4pp as upp
import radial_profiles as rap

'''
Script to plot meridional cuts of a solution field
Use as:

python3 plot_field.py nR ntheta theta0 theta1 field opt nsol

nR     : number of points in radius
ntheta : number of points in the theta direction
theta0 : starting colatitude
theta1 : final colatitude
field  : 'u' (or 'vel') the flow velocity, 'mf' the mass flux, 'vis' the viscous force over the density, 'cor' the Coriolis
         force, 'buo' the buoyancy force (of the entropy and the composition), 'curl_u', 'curl_vis', 'curl_cor' and
         'curl_buo' their curls, 'b' the magnetic field
opt    : 'raw' for real part (phase and phi dependent!), or 'abs' for the magnitude
nsol   : solution index

FIX (2026-11-08): 'u', 'b', 'buo' and the curls work ('u' was not passed on as 'vel', 'b' had no worker, 'buo' needed a
flow, 'curl_buo' was spelt 'curlbuo', and a curl came with its r-derivative), the buoyancy includes the composition, and
forced runs (no eigenvalues0.dat), fewer colatitudes than cores and numpy < 2.1 (no np.unstack) work too.
'''

plt.rc('text', usetex=True)

solnum = int(sys.argv[7])

csol, bsol = 0, None   # NEW (2026-11-08): the composition (for the buoyancy) and the magnetic field (for 'b')

if sys.argv[5] in ['u', 'vel', 'mf', 'vis', 'vif', 'cor', 'curl_u', 'curl_vel', 'curl_cor', 'curl_vis', 'curl_vif']:

    _sol = ut.load_field('flow', solnum)   # NEW (2026-11-08): flow.npy or the text files
    ru, iu = _sol.real, _sol.imag
    vsymm = par.symm
    usol = upp.expand_reshape_sol( ru + 1j*iu, vsymm)
    tsol = None
    cmap = 'rainbow'

elif sys.argv[5] in ['buo', 'curl_buo']:

    vsymm = par.symm
    tsol = None
    if par.thermal:
        _sol = ut.load_field('thermal', solnum)   # NEW (2026-11-08): thermal.npy or the text files
        rt, it = _sol.real, _sol.imag
        tsol = upp.expand_reshape_sol( rt + 1j*it, vsymm)
    if par.compositional:   # NEW (2026-11-08): the composition's buoyancy
        csol = upp.expand_reshape_sol( ut.load_field('composition', solnum), vsymm)
    usol = upp.zero_flow()   # FIX (2026-11-08): the buoyancy needs no flow (set_solution needs one, None stopped it)
    cmap = 'rainbow'

elif sys.argv[5] in ['test1', 'test2']:

    _sol = ut.load_field('flow', solnum)   # NEW (2026-11-08): flow.npy or the text files
    ru, iu = _sol.real, _sol.imag
    vsymm = par.symm
    usol = upp.expand_reshape_sol( ru + 1j*iu, vsymm)
    tsol = None
    if par.thermal:
        _sol = ut.load_field('thermal', solnum)   # NEW (2026-11-08): thermal.npy or the text files
        rt, it = _sol.real, _sol.imag
        tsol = upp.expand_reshape_sol( rt + 1j*it, vsymm)
    if par.compositional:   # NEW (2026-11-08): the composition's buoyancy
        csol = upp.expand_reshape_sol( ut.load_field('composition', solnum), vsymm)
    cmap = 'rainbow'

elif sys.argv[5] == 'b':
    _sol = ut.load_field('magnetic', solnum)   # NEW (2026-11-08): magnetic.npy or the text files
    a0, b0 = _sol.real, _sol.imag
    vsymm = ut.bsymm
    bsol = upp.expand_reshape_sol( a0 + 1j*b0, vsymm)   # FIX (2026-11-08): drawn by utils4pp.worker_4plot
    usol, tsol = upp.zero_flow(), None
    cmap = 'plasma'
    tl0 = r'\mathbf{b}'

# FIX (2026-11-08): a forced run writes no eigenvalues0.dat (sigma is used by 'test1' alone)
sigma = np.loadtxt('eigenvalues0.dat').reshape((-1,2))[solnum,0] if par.forcing == 0 else 0.

if sys.argv[5] in ['u', 'vel']:
    tl0 = r'\mathbf{u}'
elif sys.argv[5] == 'mf':
    tl0 = r'\rho\,\mathbf{u}'
elif sys.argv[5] in ['vis','vif']:
    tl0 = r'\mathbf{F_\nu/\rho}'
elif sys.argv[5] == 'cor':
    tl0 = r'2\mathbf{\hat z}\times \mathbf{u}'
elif sys.argv[5] == 'buo':
    tl0 = r'\mathbf{F_g}'
elif sys.argv[5] in ['curl_u', 'curl_vel']:
    tl0 = r'\nabla\times\mathbf{u}'
elif sys.argv[5] in ['curl_vis','curl_vif']:
    tl0 = r'\nabla\times(\mathbf{F_\nu/\rho})'
elif sys.argv[5] == 'curl_cor':
    tl0 = r'\nabla\times(2\mathbf{\hat z}\times \mathbf{u})'
elif sys.argv[5] == 'curl_buo':
    tl0 = r'\nabla\times(\mathbf{F_g})'
elif sys.argv[5] == 'test1':
    tl0 = r'\nabla\times (\sigma\,\mathbf{u})'
elif sys.argv[5] == 'test2':
    tl0 = r'\nabla\times (\mathbf{F_g}+\mathbf{F_\nu}-\mathbf{F_\Omega})'

if sys.argv[6] == 'raw':
    bra1 = r'$\mathbf{\hat r}\cdot\mathrm{Re}\left('
    bra2 = r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left('
    bra3 = r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left('
    ket = r'\right)$'

elif sys.argv[6] == 'abs':
    bra1 = r'$\mathbf{\hat r}\cdot\left|'
    bra2 = r'$\mathbf{\hat \theta}\cdot\left|'
    bra3 = r'$\mathbf{\hat \phi}\cdot\left|'
    ket = r'\right|$'
                    
titlelabels = [bra1 + tl0 + ket, bra2 + tl0 + ket, bra3 + tl0 + ket]

lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = 1
n    = ut.n
n0   = ut.n0

nR = int(sys.argv[1]) # number of radial points
Ntheta = int(sys.argv[2]) # number of points in the theta direction

ncpus   = max(1, min(mp.cpu_count(), Ntheta))   # FIX (2026-11-08): at least one theta point per process (none if Ntheta < ncpus/2)
ntht_pp = int(np.round(Ntheta/ncpus))  # number of theta points per cpu
totheta = ntht_pp * ncpus     # the actual number of total theta points  

# setup radial grid
gap = rcmb-ricb
r = np.linspace(ricb,rcmb,nR)
if ricb == 0:
	r = r[1:]
	nR = nR - 1

phi = 0. # select meridional cut

ll0 = ut.ell(m,lmax,vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll    = ll0[2]


# --------------------------------------------------------------
if sys.argv[5] == 'test2':
    out10 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_buo', csol=csol)
    out11 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_vis', csol=csol)
    out12 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_cor', csol=csol)
    #out13 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curlvel')
    out1 = out10 + out11 - out12 # - sigma*out13
elif sys.argv[5] == 'test1':
    out1 = sigma * upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_vel')
else:
    # FIX (2026-11-08): the names of utils4pp.worker_4plot ('u' is 'vel', 'vif' is 'vis'), and the composition and the field
    fld = {'u': 'vel', 'vif': 'vis', 'curl_u': 'curl_vel', 'curl_vif': 'curl_vis'}.get(sys.argv[5], sys.argv[5])
    out1 = upp.diagnose_4plot(ncpus, usol, tsol, r, fld, csol=csol, bsol=bsol)
if out1.ndim == 4:   # FIX (2026-11-08): the values of a curl (utils4pp.curl gives their r-derivatives too, as spin_doctor uses)
    out1 = out1[:, :, 0, :]
# --------------------------------------------------------------




if sys.argv[5][:4] in ['curl', 'test']:
    vsymm = -1*vsymm  # change the symmetry if we are plotting the curl of something

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1

Qlr = out1[idP:plx:2,0,:]
Slr = out1[idP:plx:2,1,:]
Tlr = out1[idT:tlx:2,2,:]

ll0 = ut.ell(m,lmax,vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll    = ll0[2]

# setup the latitudinal grid
theta = np.linspace(float(sys.argv[3])*np.pi/180, float(sys.argv[4])*np.pi/180, totheta+2)
theta = theta[1:-1]
theta2 = np.reshape(theta,(-1,ncpus)).copy()   # FIX (2026-11-08): reshape(..., copy=True) needs numpy >= 2.1

clm = np.zeros((lmax-m+2,1))
for i,l in enumerate(ll):
	clm[i] = np.sqrt((l-m)*(l+m))

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1


def pieceofcake( theta_pp ):

    ntht_pp = len(theta_pp)
    out = np.zeros( (nR*ntht_pp,5), dtype=complex ) 

    k=0
    for kt in range(ntht_pp):

        tht = theta_pp[kt]
        ylm = np.r_[ut.Ylm_full(lmax, m, tht, phi),0]	
        for kr,rr in enumerate(r):
            
            out[k,0]   = rr*np.sin(tht)  #s
            out[k,1]   = rr*np.cos(tht)  #z
            
            out[k,2] = np.dot( Qlr[:,kr], ylm[idP:plx:2] )  #ur	

            tmp1 = np.dot(           -(llpol+1) * Slr[:,kr]/np.tan(tht), ylm[idP:plx:2]     )
            tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,kr]/np.sin(tht), ylm[idP+1:plx+1:2] )
            tmp3 = np.dot(                 1j*m * Tlr[:,kr]/np.sin(tht), ylm[idT:tlx:2]     )
            out[k,3] = tmp1+tmp2+tmp3  #utheta
            
            tmp1 = np.dot(             (lltor+1) * Tlr[:,kr]/np.tan(tht), ylm[idT:tlx:2]     )
            tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,kr]/np.sin(tht), ylm[idT+1:tlx+1:2] )
            tmp3 = np.dot(                  1j*m * Slr[:,kr]/np.sin(tht), ylm[idP:plx:2]     )
            out[k,4] = tmp1+tmp2+tmp3  #uphi
            	
            k=k+1

    return out


pool  = mp.Pool(processes=ncpus)
popov = [ pool.apply_async( pieceofcake, [theta2[:,i]]) for i in range(ncpus) ]
out0 = [ pp.get() for pp in popov ]
pool.close()
pool.join()

out1 = np.vstack(out0)
(s0,z0,ur,utheta,uphi) = out1.T   # FIX (2026-11-08): np.unstack(out1, axis=1), which numpy < 2.1 lacks
s = np.real(s0)
z = np.real(z0)

# Mask the inner core
a = 1.
c = 1.
id_in = np.where((s**2/(a**2)) + (z**2/(c**2)) < 1.)
if par.centrifugal:
    # NEW (2026-11-05): a deformed body (centrifugal = 1): the points, and the flow (u, mf), where they are in the body
    # (centrifugal.to_physical); the walls are the level surfaces r = 1 and r = ricb of the reference radius. The field b is
    # a flux of the reference space like the flow (b = F b̃/J), so it maps the same way (NEW 2026-11-05, phase 1b)
    import centrifugal as cf
    sref, zref = s, z
    rref, tref = np.sqrt(s**2 + z**2), np.arctan2(s, z)
    if sys.argv[5] in ['u', 'vel', 'mf', 'b']:
        s, z, ur, utheta, uphi = cf.to_physical(rref, tref, ur, utheta, uphi)
    else:   # other fields are drawn at their points in the body, with their components in the reference space
        s, z = cf.to_physical(rref, tref)
s1 = s[id_in]
z1 = z[id_in]
triang = tri.Triangulation(s1, z1)
if par.centrifugal:   # NEW (2026-11-05): the inner core is inside the level surface ricb (triangle midpoints in the reference space)
    smid, zmid = sref[id_in][triang.triangles].mean(axis=1), zref[id_in][triang.triangles].mean(axis=1)
    mask = np.where(smid**2 + zmid**2 <= ricb**2, 1, 0)
else:
    xmid = s1[triang.triangles].mean(axis=1)
    x2 = xmid*xmid
    ymid = z1[triang.triangles].mean(axis=1)
    y2 = ymid*ymid
    mask = np.where( (x2 + y2 <= ricb**2), 1, 0)
triang.set_mask(mask)


fig=plt.figure(figsize=(14,7))
# ------------------------------------------------------------------- ur
ax1=fig.add_subplot(131)
ax1.set_title(titlelabels[0],size=20)
#ax1.text(0.1,0,titlelabels[0],size=20)
if sys.argv[6] == 'raw':
    im1=ax1.tricontourf( triang, np.real(ur[id_in]), 70, cmap=cmap)
elif sys.argv[6] == 'abs':
    im1=ax1.tricontourf( triang, np.absolute(ur[id_in]), 70, cmap=cmap)
#for c in im1.collections:
#              c.set_edgecolor('face')
im1.set_edgecolor('face')
ax1.set_aspect('equal')
plt.colorbar(im1,aspect=70)

# --------------------------------------------------------------- utheta
ax2=fig.add_subplot(132)
ax2.set_title(titlelabels[1],size=20)
#ax2.text(0.1,0,titlelabels[1],size=20)
if sys.argv[6] == 'raw':
    im2=ax2.tricontourf( triang, np.real(utheta[id_in]), 70, cmap=cmap)
elif sys.argv[6] == 'abs':
    im2=ax2.tricontourf( triang, np.absolute(utheta[id_in]), 70, cmap=cmap)
#for c in im2.collections:
#              c.set_edgecolor('face')
im2.set_edgecolor('face')
ax2.set_aspect('equal')
plt.colorbar(im2,aspect=70)

# ----------------------------------------------------------------- uphi
ax3=fig.add_subplot(133)
ax3.set_title(titlelabels[2],size=20)
#ax3.text(0.1,0,titlelabels[2],size=20)
if sys.argv[6] == 'raw':
    im3=ax3.tricontourf( triang, np.real(uphi[id_in]), 70, cmap=cmap)
elif sys.argv[6] == 'abs':
    im3=ax3.tricontourf( triang, np.absolute(uphi[id_in]), 70, cmap=cmap)
#for c in im3.collections:
#              c.set_edgecolor('face')
im2.set_edgecolor('face')
ax3.set_aspect('equal')
plt.colorbar(im3,aspect=70)

# ----------------------------------------------------------------------
plt.tight_layout()
plt.show()

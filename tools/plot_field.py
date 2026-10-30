import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch


def set_contour_edges_to_face(contour_set):
    '''
    Avoid white seams in filled contours when Matplotlib exposes collections.
    Newer Matplotlib versions may not provide this public attribute.
    '''
    for collection in getattr(contour_set, 'collections', []):
        collection.set_edgecolor('face')


sys.path.insert(1,'bin/')

import utils as ut
import parameters as par
import utils_pp as upp

'''
Script to plot meridional cuts of a solution field
Use as:

python3 plot_field.py nsol nR ntheta theta0 theta1 field opt

nsol   : solution number
nR     : number of points in radius
ntheta : number of points in the theta direction
theta0 : starting colatitude
theta1 : final colatitude
field  : field to plot:
         'u'  for fluid velocity in [ricb, rcmb]
         'b'  for fluid-core magnetic field in [ricb, rcmb]
         'bm' for resolved-mantle magnetic field in [rcmb, r_surface]
opt    : 'raw' for real part (phase and phi dependent!), or 'abs' for the magnitude
         'phase' for the complex phase in radians, masked where the amplitude is tiny
'''

plt.rc('text', usetex=True)

solnum = int(sys.argv[1])
field = sys.argv[6]
plot_opt = sys.argv[7]

if field == 'u':
    a0 = np.loadtxt('real_flow.field',usecols=solnum)
    b0 = np.loadtxt('imag_flow.field',usecols=solnum)
    vsymm = par.symm
    radial_domain = (par.ricb, ut.rcmb)
    expand_origin = True
    if plot_opt == 'raw':
        titlelabels = [r'$\mathbf{\hat r}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$', \
                       r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$', \
                       r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$']
    elif plot_opt == 'abs':
        titlelabels = [r'$\mathbf{\hat r}\cdot\left|\mathbf{u_0}\right|$', \
                       r'$\mathbf{\hat \theta}\cdot\left|\mathbf{u_0}\right|$', \
                       r'$\mathbf{\hat \phi}\cdot\left|\mathbf{u_0}\right|$'] 
    elif plot_opt == 'phase':
        titlelabels = [r'$\arg(u_r)$', r'$\arg(u_\theta)$', r'$\arg(u_\phi)$']
    else:
        raise ValueError("opt must be 'raw', 'abs', or 'phase'")
    cmap = 'rainbow'
elif field == 'b':
    a0 = np.loadtxt('real_magnetic.field',usecols=solnum)
    b0 = np.loadtxt('imag_magnetic.field',usecols=solnum)
    radial_domain = (par.ricb, ut.rcmb)
    expand_origin = True
    if plot_opt == 'raw':
        titlelabels = [r'$\mathbf{\hat r}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                       r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                       r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$']
    elif plot_opt == 'abs':
        titlelabels = [r'$\mathbf{\hat r}\cdot\left|\mathbf{b_0}\right|$', \
                       r'$\mathbf{\hat \theta}\cdot\left|\mathbf{b_0}\right|$', \
                       r'$\mathbf{\hat \phi}\cdot\left|\mathbf{b_0}\right|$'] 
    elif plot_opt == 'phase':
        titlelabels = [r'$\arg(b_r)$', r'$\arg(b_\theta)$', r'$\arg(b_\phi)$']
    else:
        raise ValueError("opt must be 'raw', 'abs', or 'phase'")
    vsymm = ut.bsymm
    cmap = 'plasma'
elif field == 'bm':
    if not ut.resolved_mantle:
        raise ValueError("field='bm' requires mantle='resolved' in bin/parameters.py")
    a0 = np.loadtxt('real_magnetic_mantle.field',usecols=solnum)
    b0 = np.loadtxt('imag_magnetic_mantle.field',usecols=solnum)
    radial_domain = (ut.rcmb, par.r_surface)
    expand_origin = False
    if plot_opt == 'raw':
        titlelabels = [r'$\mathbf{\hat r}\cdot\mathrm{Re}\left(\mathbf{b_m}\right)$', \
                       r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left(\mathbf{b_m}\right)$', \
                       r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{b_m}\right)$']
    elif plot_opt == 'abs':
        titlelabels = [r'$\mathbf{\hat r}\cdot\left|\mathbf{b_m}\right|$', \
                       r'$\mathbf{\hat \theta}\cdot\left|\mathbf{b_m}\right|$', \
                       r'$\mathbf{\hat \phi}\cdot\left|\mathbf{b_m}\right|$']
    elif plot_opt == 'phase':
        titlelabels = [r'$\arg(b_{m,r})$', r'$\arg(b_{m,\theta})$', r'$\arg(b_{m,\phi})$']
    else:
        raise ValueError("opt must be 'raw', 'abs', or 'phase'")
    vsymm = ut.bsymm
    cmap = 'plasma'
else:
    raise ValueError("field must be 'u', 'b', or 'bm'")

if plot_opt == 'phase':
    cmap = 'twilight_shifted'

lmax = par.lmax
m    = par.m
Ra, Rb = radial_domain

nR = int(sys.argv[2]) # number of radial points
Ntheta = int(sys.argv[3]) # number of points in the theta direction

# setup radial grid
r = np.linspace(Ra,Rb,nR)
if Ra == 0:
    r = r[1:]
    nR = nR - 1
x = upp.xcheb(r,Ra,Rb)


phi = 0. # select meridional cut


# expand solution in case ricb=0 for fluid-core fields
if expand_origin:
    aib = upp.expand_sol(a0+1j*b0,vsymm)
else:
    aib = a0+1j*b0
a = np.real(aib)
b = np.imag(aib)

ll0 = ut.ell(m,lmax,vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll    = ll0[2]

ncoeff = int(np.size(aib)/2)
N = int(ncoeff/np.size(llpol))

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x,N-1) # this matrix has nR rows and N cols

Plj0 = a[:ncoeff] + 1j*b[:ncoeff]
Tlj0 = a[ncoeff:ncoeff+ncoeff] + 1j*b[ncoeff:ncoeff+ncoeff]

Plj  = np.reshape(Plj0,(np.size(llpol),N))
Tlj  = np.reshape(Tlj0,(np.size(llpol),N))
dPlj = np.zeros(np.shape(Plj),dtype=complex)

Plr = np.zeros((np.size(llpol), nR),dtype=complex)
dP  = np.zeros((np.size(llpol), nR),dtype=complex)
rP  = np.zeros((np.size(llpol), nR),dtype=complex)
Qlr = np.zeros((np.size(llpol), nR),dtype=complex)
Slr = np.zeros((np.size(llpol), nR),dtype=complex)
Tlr = np.zeros((np.size(llpol), nR),dtype=complex)

np.matmul( Plj, chx.T, Plr )
np.matmul( Tlj, chx.T, Tlr )

for k in range(np.size(llpol)):
    dPlj[k,:] = ut.Dcheb(Plj[k,:], Ra, Rb)

np.matmul(dPlj, chx.T, dP)

rP  = Plr / r[np.newaxis,:]
Qlr = (llpol*(llpol+1))[:,np.newaxis] * rP
Slr = rP + dP

# setup the latitudinal grid
theta = np.linspace(float(sys.argv[4])*np.pi/180,float(sys.argv[5])*np.pi/180,Ntheta+2)
theta = theta[1:-1]

s = np.zeros( nR*Ntheta )
z = np.zeros( nR*Ntheta )

#ur2 = np.zeros( (nR)*Ntheta )
#ut2 = np.zeros( (nR)*Ntheta )
#up2 = np.zeros( (nR)*Ntheta )

ur     = np.zeros( (nR)*Ntheta, dtype=complex)
utheta = np.zeros( (nR)*Ntheta, dtype=complex)
uphi   = np.zeros( (nR)*Ntheta, dtype=complex)

clm = np.zeros((lmax-m+2,1))
for i,l in enumerate(ll):
	clm[i] = np.sqrt((l-m)*(l+m))

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1


k=0
for kt in range(Ntheta):

	ylm = np.r_[ut.Ylm_full(lmax, m, theta[kt], phi),0]	
	for kr in range(0,nR):
		
		s[k]   = r[kr]*np.sin(theta[kt])
		z[k]   = r[kr]*np.cos(theta[kt])
		
		ur[k] = np.dot( Qlr[:,kr], ylm[idP:plx:2] )
		#ur2[k] = absolute(dot( Qlm[:,kr], ylm[idP:plx:2] ))**2		

		tmp1 = np.dot(           -(llpol+1) * Slr[:,kr]/np.tan(theta[kt]), ylm[idP:plx:2]     )
		tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,kr]/np.sin(theta[kt]), ylm[idP+1:plx+1:2] )
		tmp3 = np.dot(                 1j*m * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT:tlx:2]     )
		utheta[k] = tmp1+tmp2+tmp3
		#ut2[k] = absolute(tmp1+tmp2+tmp3)**2
		
		tmp1 = np.dot(             (lltor+1) * Tlr[:,kr]/np.tan(theta[kt]), ylm[idT:tlx:2]     )
		tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT+1:tlx+1:2] )
		tmp3 = np.dot(                  1j*m * Slr[:,kr]/np.sin(theta[kt]), ylm[idP:plx:2]     )
		uphi[k] = tmp1+tmp2+tmp3
		#up2[k] = absolute(tmp1+tmp2+tmp3)**2
			
		#uz[k] = ur[k]*cos(theta[kt]) - ut[k]*sin(theta[kt])		
		k=k+1

print(f"{field}: radial domain = [{Ra:g}, {Rb:g}], Chebyshev N = {N}")
print(f"{field}: max |radial| = {np.max(np.absolute(ur)):.6e}")
print(f"{field}: max |theta|  = {np.max(np.absolute(utheta)):.6e}")
print(f"{field}: max |phi|    = {np.max(np.absolute(uphi)):.6e}")

# Mask the part outside the selected spherical shell
id_in = np.arange(np.size(s))
s1 = s
z1 = z
triang = tri.Triangulation(s1, z1)
xmid = s1[triang.triangles].mean(axis=1)
x2 = xmid*xmid
ymid = z1[triang.triangles].mean(axis=1)
y2 = ymid*ymid
radius2_mid = x2 + y2
eps = 1e-12
mask = np.where((radius2_mid < Ra**2 - eps) | (radius2_mid > Rb**2 + eps), 1, 0)
triang.set_mask(mask)
print(f"{field}: unmasked triangles = {np.count_nonzero(mask == 0)} / {np.size(mask)}")


fig=plt.figure(figsize=(14,7))
phase_levels = np.linspace(-np.pi, np.pi, 81)

def phase_value(component):
    amplitude = np.absolute(component)
    cutoff = 1e-8*np.nanmax(amplitude)
    return np.ma.masked_where(amplitude < cutoff, np.angle(component))

# ------------------------------------------------------------------- ur
ax1=fig.add_subplot(131)
ax1.set_title(titlelabels[0],size=20)
#ax1.text(0.1,0,titlelabels[0],size=20)
if plot_opt == 'raw':
    im1=ax1.tricontourf( triang, np.real(ur[id_in]), 70, cmap=cmap)
elif plot_opt == 'abs':
    im1=ax1.tricontourf( triang, np.absolute(ur[id_in]), 70, cmap=cmap)
elif plot_opt == 'phase':
    im1=ax1.tricontourf( triang, phase_value(ur[id_in]), phase_levels, cmap=cmap, vmin=-np.pi, vmax=np.pi)
set_contour_edges_to_face(im1)
ax1.set_aspect('equal')
plt.colorbar(im1,aspect=70)

# --------------------------------------------------------------- utheta
ax2=fig.add_subplot(132)
ax2.set_title(titlelabels[1],size=20)
#ax2.text(0.1,0,titlelabels[1],size=20)
if plot_opt == 'raw':
    im2=ax2.tricontourf( triang, np.real(utheta[id_in]), 70, cmap=cmap)
elif plot_opt == 'abs':
    im2=ax2.tricontourf( triang, np.absolute(utheta[id_in]), 70, cmap=cmap)
elif plot_opt == 'phase':
    im2=ax2.tricontourf( triang, phase_value(utheta[id_in]), phase_levels, cmap=cmap, vmin=-np.pi, vmax=np.pi)
set_contour_edges_to_face(im2)
ax2.set_aspect('equal')
plt.colorbar(im2,aspect=70)

# ----------------------------------------------------------------- uphi
ax3=fig.add_subplot(133)
ax3.set_title(titlelabels[2],size=20)
#ax3.text(0.1,0,titlelabels[2],size=20)
if plot_opt == 'raw':
    im3=ax3.tricontourf( triang, np.real(uphi[id_in]), 70, cmap=cmap)
elif plot_opt == 'abs':
    im3=ax3.tricontourf( triang, np.absolute(uphi[id_in]), 70, cmap=cmap)
elif plot_opt == 'phase':
    im3=ax3.tricontourf( triang, phase_value(uphi[id_in]), phase_levels, cmap=cmap, vmin=-np.pi, vmax=np.pi)
set_contour_edges_to_face(im3)
ax3.set_aspect('equal')
plt.colorbar(im3,aspect=70)

# ----------------------------------------------------------------------
plt.tight_layout()
plt.show()

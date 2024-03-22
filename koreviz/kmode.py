import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch
import scipy.sparse as ss
import shtns
from glob import glob
#import cmasher as cmr
from .plotlib import add_colorbar,default_cmap,radContour,merContour,eqContour
from .libkoreviz import spec2spat_vec,spec2spat_scal
import sys
import os

class kmode:

    def __init__(self, vort=False,datadir='./',field='u', solnum=0,
                 nr=None, nphi=None, nthreads=4, phase=0, transform=True):

        sys.path.insert(0,datadir+'bin')

        import parameters as par
        import utils as ut
        import utils4pp as upp

        self.solnum = solnum
        self.field  = field

        self.m      = par.m
        self.symm   = par.symm

        self.ricb   = par.ricb
        self.rcmb   = 1

        self.lmax   = par.lmax
        self.N      = par.N
        self.n0     = ut.n0

        self.ut     = ut
        self.par    = par

        if nr is None:
            self.nr = par.N + 2
        else:
            self.nr = nr

        if self.m == 0:

            if nphi is None:
                self.nphi = par.lmax * 3 # Orszag's 1/3 rule
            else:
                self.nphi = nphi
            self.ntheta = self.nphi // 2

        else:

            if nphi is None:
                self.nphi = par.lmax * 3 // self.m  # Orszag's 1/3 rule
            else:
                self.nphi = nphi // self.m
            self.ntheta = (self.nphi * self.m) // 2

        # set up the evenly spaced radial grid
        r = np.linspace(self.ricb, self.rcmb, self.nr)

        if self.ricb == 0:
            r = r[1:]
            self.nr = self.nr -1
        x = upp.xcheb(r, self.ricb, self.rcmb)

        self.r = r

        # matrix with Chebyshev polynomials at every x point for all degrees:
        chx = ch.chebvander(x,self.N-1) # this matrix has nr rows and N-1 cols

        # read fields from disk
        if field == 'u':
            a = np.loadtxt(datadir + 'real_flow.field',usecols=solnum)
            b = np.loadtxt(datadir + 'imag_flow.field',usecols=solnum)
            vsymm = par.symm
            vec = True
        elif field == 'b':
            a = np.loadtxt(datadir + 'real_magnetic.field',usecols=solnum)
            b = np.loadtxt(datadir + 'imag_magnetic.field',usecols=solnum)
            vsymm = ut.bsymm
            vec = True
        elif field in ['t','temp','temperature']:
            if len(glob('*_temperature.field')) > 0:
                a = np.loadtxt(datadir + 'real_temperature.field',usecols=solnum)
                b = np.loadtxt(datadir + 'imag_temperature.field',usecols=solnum)
            elif len(glob('*_temp.field')) > 0:
                a = np.loadtxt(datadir + 'real_temp.field',usecols=solnum)
                b = np.loadtxt(datadir + 'imag_temp.field',usecols=solnum)
            field='temperature'
            vsymm = par.symm
            vec = False
        elif field in ['comp','composition']:
            a = np.loadtxt(datadir + 'real_composition.field',usecols=solnum)
            b = np.loadtxt(datadir + 'imag_composition.field',usecols=solnum)
            field='composition'
            vsymm = par.symm
            vec = False

        # expand solution in case ricb=0, multiply by complex phase factor
        aib = upp.expand_sol(a+1j*b,vsymm)*(np.cos(phase)+1j*np.sin(phase))
        a = np.real(aib)
        b = np.imag(aib)

        if vec:

            sol = spec2spat_vec(self,ut,par,chx,a,b,vsymm,nthreads,
                               vort=vort,transform=transform)

            self.Q,self.S,self.T = sol[:3]

            if transform:
                if not vort:

                    exec('self.'+field+'r'     + '= sol[3]')
                    exec('self.'+field+'theta' + '= sol[4]')
                    exec('self.'+field+'phi'   + '= sol[5]')

                else:
                    if field == 'u':
                        self.vort_r, self.vort_t, self.vort_p = sol[3:]
                    elif field == 'b':
                        self.jr, self.jtheta, self.jphi = sol[3:]

            del sol

        else:
            sol = spec2spat_scal(self,ut,par,chx,a,b,vsymm,nthreads,transform=transform)
            self.Q = sol[0]
            if transform:
                scal = sol[1]
                exec('self.'+field + '= scal')
                del scal
            del sol


    def get_data(self,field):

        field = field.lower()

        if field in ['ur','vr','urad','vrad']:
            data = self.ur
            titl = r'$u_r$'

        if field in ['up','vp','uphi','vphi']:
            data = self.uphi
            titl = r'$u_\phi$'

        if field in ['ut','vt','utheta','vtheta']:
            data = self.utheta
            titl = r'$u_\theta$'

        if field in ['br','brad']:
            data = self.br
            titl = r'$B_r$'

        if field in ['bp','bphi']:
            data = self.bphi
            titl = r'$B_\phi$'

        if field in ['bt','btheta']:
            data = self.btheta
            titl = r'$B_\theta$'

        if field in ['t','temp','temperature']:
            data = self.temperature
            titl = r'Temperature'

        if field in ['c','comp','composition']:
            data = self.composition
            titl = r'Composition'

        if field in ['energy','ener','ke','e']:
            data = 0.5 * (self.ur**2 + self.utheta**2 + self.uphi**2)
            titl = r'Kinetic Energy'

        if field in ['vortz']:
            th3D = np.zeros_like(self.vort_r)
            for k in range(self.ntheta):
                th3D[:,k,:] = self.theta[k]

            data = self.vort_r * np.cos(th3D) - self.vort_t * np.sin(th3D)
            titl = r'$\omega_z$'

        return data, titl

    def surf(self, field='ur', r=0.5, levels=48, cmap=None,
             colbar=True, titl=True, clim=[0,0]):
        # Surface plot at constant radius

        if self.m == 0:
            data = np.zeros([ self.ntheta, self.nphi + 1])
            ir = np.argmin(abs(self.r-r))
            dat_tmp,titl = self.get_data(field=field)
            data[:,:-1] = dat_tmp[ir,...]
            data[:, -1] = data[:,0]
        else:
            data = np.zeros([ self.ntheta, self.nphi*self.m + 1])
            ir = np.argmin(abs(self.r-r))
            dat_tmp,titl = self.get_data(field=field)
            data[:,:-1] = np.tile( dat_tmp[ir,...], self.m )
            data[:, -1] = data[:,0]

        plt.figure(figsize=(12,6))

        if cmap is None:
            cmap = default_cmap(field)

        cont = radContour( self.theta, self.phi, data.T,
                          levels=levels, cmap=cmap, clim=clim)

        if titl:
            titl = titl + r' at $r/r_o = %.2f$' %(self.r[ir]/self.rcmb)
            plt.title(titl,fontsize=30)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=40)

        plt.tight_layout()
        plt.show()


    def merid(self, field='ur', azim=0, levels=48, cmap=None,
              colbar=True, titl=True, clim=[0,0]):
        # Meridional cross section

        iphi = np.argmin(abs( self.phi - (azim*np.pi/180) )) % self.nphi
        dat_tmp,titl = self.get_data(field)
        data = dat_tmp[:,:,iphi]

        if field in ['energy','ener','e','ke']:
            #cmap = cmr.tropical_r
            data = np.log10(data)

        plt.figure(figsize=(6,9))

        if cmap is None:
            cmap = default_cmap(field)

        cont = merContour( self.r, self.theta, data.T,
                           levels=levels, cmap=cmap, clim=clim)

        if titl:
            titl = titl + r' at $\phi=%.1f^\circ$' %(self.phi[iphi] * 180/np.pi)
            plt.title(titl,fontsize=20)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
        plt.tight_layout()
        plt.show()


    def equat(self, field='ur', levels=48, cmap=None,
              colbar=True, titl=True, clim=[0,0]):
        # Equatorial cross section

        if self.m == 0:
            data = np.zeros([ self.nr, self.nphi + 1])
            itheta = np.argmin( abs( self.theta - np.pi/2 ) )
            dat_tmp,titl = self.get_data(field)
            data[:,:-1] = dat_tmp[:,itheta,:]
            data[:, -1] = data[:,0]
        else:
            data = np.zeros([ self.nr, self.nphi*self.m + 1])
            itheta = np.argmin( abs( self.theta - np.pi/2 ) )
            dat_tmp,titl = self.get_data(field)
            print(np.shape(data),np.shape(dat_tmp))
            data[:,:-1] = np.tile( dat_tmp[:,itheta,:], self.m )
            data[:, -1] = data[:,0]

        plt.figure(figsize=(11,9))

        if cmap is None:
            cmap = default_cmap(field)

        cont = eqContour(self.r, self.phi, data.T,
                         levels=levels, cmap=cmap, clim=clim)

        if titl:
            titl = titl + ' at equator'
            plt.title(titl,fontsize=20)

        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
        plt.tight_layout()
        plt.show()


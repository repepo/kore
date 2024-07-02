import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch
from glob import glob
from .plotlib import add_colorbar,default_cmap,radContour,merContour,eqContour
from .libkoreviz import spec2spat_vec,spec2spat_scal
import sys


class kmode:

    def __init__(self, vort=False,datadir='.',field='u', solnum=0,
                 nr=None, nphi=None, nthreads=4, phase=0, transform=True):

        sys.path.insert(0,datadir+'/bin')

        import parameters as par
        import utils as ut
        import utils4pp as upp
        import radial_profiles as rap

        self.solnum   = solnum
        self.lmax     = par.lmax
        self.m        = par.m
        self.symm     = par.symm
        self.N        = par.N
        self.ricb     = par.ricb
        self.rcmb     = ut.rcmb
        self.n        = ut.n
        self.n0       = ut.n0
        self.field    = field
        gap           = self.rcmb - self.ricb
        self.ut       = ut
        self.par      = par
        self.nthreads = nthreads

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
            self.nphi = max(128,self.nphi)

        else:

            if nphi is None:
                self.nphi = par.lmax * 3 // self.m  # Orszag's 1/3 rule
            else:
                self.nphi = nphi // self.m

            self.nphi = max(128//self.m,self.nphi)
            self.ntheta = (self.nphi * self.m) // 2

        # set the radial grid
        if self.ricb > 0:
            i = np.arange(0,self.nr-2)
            xk = np.r_[ 1, np.cos( (i+0.5)*np.pi/self.nr ), -1]  # include endpoints ricb and rcmb
        elif self.ricb==0:
            i = np.arange(0,self.nr-1)
            xk = np.r_[ 1, np.cos( (i+0.5)*np.pi/self.nr )    ]  # include rcmb but not the origin if ricb=0
        r = 0.5*gap*(xk+1) + self.ricb
        x0 = upp.xcheb(r,self.ricb,self.rcmb)
        self.r = r

        # matrix with Chebyshev polynomials at every x point for all degrees:
        chx = ch.chebvander(x0,par.N-1) # this matrix has nr rows and N-1 cols

        # read fields from disk
        if field == 'u':
            a = np.loadtxt('real_flow.field',usecols=solnum)
            b = np.loadtxt('imag_flow.field',usecols=solnum)
            vsymm = par.symm
            vec = True
        elif field == 'b':
            a = np.loadtxt('real_magnetic.field',usecols=solnum)
            b = np.loadtxt('imag_magnetic.field',usecols=solnum)
            vsymm = ut.bsymm
            vec = True
        elif field in ['t','temp','temperature']:
            if len(glob('*_temperature.field')) > 0:
                a = np.loadtxt('real_temperature.field',usecols=solnum)
                b = np.loadtxt('imag_temperature.field',usecols=solnum)
            elif len(glob('*_temp.field')) > 0:
                a = np.loadtxt('real_temp.field',usecols=solnum)
                b = np.loadtxt('imag_temp.field',usecols=solnum)
            field='temperature'
            vsymm = par.symm
            vec = False
        elif field in ['comp','composition']:
            a = np.loadtxt('real_composition.field',usecols=solnum)
            b = np.loadtxt('imag_composition.field',usecols=solnum)
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

            self.Qlm,self.Slm,self.Plm,self.Tlm = sol[:4]

            if transform:
                if not vort:
                    if field == 'u':
                        self.ur, self.utheta, self.uphi = sol[4:]
                        if par.anelastic: #Comment the block out if you want momentum/mass flux
                            self.rho = rap.density(self.r)
                            for irho in range(self.nr):
                                self.ur[irho,...]     /= self.rho[irho]
                                self.utheta[irho,...] /= self.rho[irho]
                                self.uphi[irho,...]   /= self.rho[irho]
                    elif field == 'b':
                        self.br, self.btheta, self.bphi = sol[4:]
                else:
                    if field == 'u':
                        self.vort_r, self.vort_t, self.vort_p = sol[4:]
                    elif field == 'b':
                        self.jr, self.jtheta, self.jphi = sol[4:]

            del sol

        else:
            sol = spec2spat_scal(self,ut,par,chx,a,b,vsymm,nthreads,transform=transform)
            self.Qlm = sol[0]
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

    def potextra(self,brcmb,rcmb,rout):

        if self.field != 'b':
            print("Potential extrapolation only valid when field='b'")
            return 0

        try:
            import shtns
        except ImportError:
            print("Potential extrapolation requires the SHTns library")
            print("It can be obtained here: https://bitbucket.org/nschaeff/shtns")

        self.nrout = len(rout)
        polar_opt = 1e-10

        norm=shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE
        lmax2 = int( self.lmax + 1 - np.sign(self.m) )
        mmax = int( np.sign(self.m) )
        mres = max(1,self.m)
        sh   = shtns.sht( lmax2, mmax=mmax, mres=mres, norm=norm,
                         nthreads=self.nthreads )
        ntheta, nphi = sh.set_grid(self.ntheta, self.nphi,
                                       polar_opt=polar_opt)

        L = sh.l * (sh.l + 1)

        brlm = sh.analys(brcmb)
        bpolcmb = np.zeros_like(brlm)
        bpolcmb[1:] = rcmb**2 * brlm[1:]/L[1:]
        btor = np.zeros_like(brlm)

        brout = np.zeros([ntheta,nphi,self.nrout])
        btout = np.zeros([ntheta,nphi,self.nrout])
        bpout = np.zeros([ntheta,nphi,self.nrout])

        for k,radius in enumerate(rout):
            print(("%d/%d" %(k,self.nrout)))

            radratio = rcmb/radius
            bpol = bpolcmb * radratio**(sh.l)
            brlm = bpol * L/radius**2
            brout[...,k] = sh.synth(brlm)

            slm = -sh.l/radius * bpol

            btout[...,k], bpout[...,k] = sh.synth(slm,btor)

        brout = np.transpose(brout,(2,0,1))
        btout = np.transpose(btout,(2,0,1))
        bpout = np.transpose(bpout,(2,0,1))

        return brout, btout, bpout

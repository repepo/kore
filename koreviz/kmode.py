import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch
from glob import glob
from .plotlib import add_colorbar,default_cmap,radContour,merContour,eqContour
from .libkoreviz import spec2spat_vec,spec2spat_scal,ell_idx,potential_field,shtns
import sys
import os


class kmode:

    def __init__(self, vort=False,datadir='.',field='u', solnum=0,
                 nr=None, nphi=None, nthreads=4, phase=0, transform=True,
                 ic=False, nr_ic=None):
        '''
        ic    : with field='b' and an inner core, also compute the magnetic field inside the inner core
                (br_ic, btheta_ic, bphi_ic on the radii r_ic), see inner_core_field
        nr_ic : number of radii in the inner core (default nr)
        '''

        sys.path.insert(0,datadir+'/bin')

        from parameters import par
        import utils as ut
        import utils4pp as upp

        self.solnum   = solnum
        self.lmax     = par.lmax
        self.m        = par.m
        self.symm     = par.symm
        self.N        = par.N
        self.ricb     = par.ricb
        self.rcmb     = ut.rcmb
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

        # multiply by the complex phase factor, then reshape to (l, Chebyshev degree) and expand the ricb = 0
        # solution from N/2 to N coefficients (done once, here)
        sol = (a + 1j*b)*np.exp(1j*phase)

        # spectral coefficients: one row per degree (self.lpol for the poloidal/scalar part, self.ltor for the
        # toroidal part), Chebyshev degree along the columns
        ll, idp, idt = ell_idx(self.m, self.lmax, vsymm)
        coef = upp.expand_reshape_sol(sol,vsymm)
        if vec:
            Plj, Tlj = coef
            self.Tlj, self.ltor = Tlj, ll[idt]
        else:
            Plj = coef
        self.Plj, self.lpol = Plj, ll[idp]

        if shtns is None:
            print("SHTns not found: kmode keeps only the spectral coefficients, no fields on the grid.")
            print("potextra falls back to scipy. SHTns: https://bitbucket.org/nschaeff/shtns")
            from scipy.special import roots_legendre
            self.ntheta += self.ntheta % 2
            self.theta = np.arccos(roots_legendre(self.ntheta)[0][::-1])  # Gauss grid, north to south as SHTns
            self.phi   = np.linspace(0., 2*np.pi, self.nphi*max(1,self.m)+1, endpoint=True)
            return

        if vec:

            sol = spec2spat_vec(self,ut,chx,Plj,Tlj,vsymm,nthreads,
                               vort=vort,transform=transform)

            self.Qlm,self.Slm,self.Plm,self.Tlm = sol[:4]

            if transform:
                if not vort:
                    if field == 'u':
                        self.ur, self.utheta, self.uphi = sol[4:]
                    elif field == 'b':
                        self.br, self.btheta, self.bphi = sol[4:]
                        if ic and self.ricb > 0:
                            self.inner_core_field(nr_ic=nr_ic, phase=phase)
                else:
                    if field == 'u':
                        self.vort_r, self.vort_t, self.vort_p = sol[4:]
                    elif field == 'b':
                        self.jr, self.jtheta, self.jphi = sol[4:]

            del sol

        else:
            sol = spec2spat_scal(self,chx,Plj,vsymm,
                                 nthreads,transform=transform)
            self.Qlm = sol[0]
            if transform:
                scal = sol[1]
                exec('self.'+field + '= scal')
                del scal
            del sol


    def get_data(self,field,ic=False):

        if ic:  # magnetic field inside the inner core
            if not hasattr(self, 'br_ic'):
                raise ValueError("no inner core field: use kmode(field='b', ic=True) with ricb > 0")
            comp = {'br':'br', 'brad':'br', 'bt':'btheta', 'btheta':'btheta', 'bp':'bphi', 'bphi':'bphi'}
            if field.lower() not in comp:
                raise ValueError('the inner core field has only br, btheta and bphi')
            data, titl, field = self.get_data(field)
            return getattr(self, comp[field.lower()] + '_ic'), titl, field

        if self.field in ['t','temp','temperature']:
            field = 'temperature'
        elif self.field in ['c','comp','composition']:
            field = 'composition'

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

        return data, titl, field


    def surf(self, field='ur', r=0.5, levels=48, cmap=None,
             colbar=True, titl=True, clim=[0,0]):
        # Surface plot at constant radius. Below the ICB, the inner core field is used (if computed, see ic)

        ic = r < self.ricb and hasattr(self, 'br_ic')
        rgrid = self.r_ic if ic else self.r
        ir = np.argmin(abs(rgrid-r))
        dat_tmp,titl,field = self.get_data(field=field, ic=ic)

        data = np.zeros([ self.ntheta, self.nphi*max(1,self.m) + 1])
        data[:,:-1] = np.tile( dat_tmp[ir,...], max(1,self.m) )
        data[:, -1] = data[:,0]

        plt.figure(figsize=(12,6))

        if cmap is None:
            cmap = default_cmap(field)

        cont = radContour( self.theta, self.phi, data.T,
                          levels=levels, cmap=cmap, clim=clim)

        if titl:
            titl = titl + r' at $r/r_o = %.2f$' %(rgrid[ir]/self.rcmb)
            plt.title(titl,fontsize=30)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=40)

        plt.tight_layout()
        plt.show()


    def merid(self, field='ur', azim=0, levels=48, cmap=None,
              colbar=True, titl=True, clim=[0,0], ic=False):
        # Meridional cross section. ic=True also shows the inner core field (see kmode's ic)

        iphi = np.argmin(abs( self.phi - (azim*np.pi/180) )) % self.nphi
        dat_tmp,titl,field = self.get_data(field)
        data = dat_tmp[:,:,iphi]
        r = self.r
        if ic:  # append the inner core, without its first radius (the ICB, already the last of r)
            dat_ic = self.get_data(field, ic=True)[0]
            data = np.concatenate( (data, dat_ic[1:,:,iphi]), axis=0 )
            r = np.r_[ r, self.r_ic[1:] ]

        if field in ['energy','ener','e','ke']:
            #cmap = cmr.tropical_r
            data = np.log10(data)

        plt.figure(figsize=(6,9))

        if cmap is None:
            cmap = default_cmap(field)

        cont = merContour( r, self.theta, data.T,
                           levels=levels, cmap=cmap, clim=clim)
        if ic:
            plt.plot(self.ricb*np.sin(self.theta), self.ricb*np.cos(self.theta), 'k', lw=0.6)

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
              colbar=True, titl=True, clim=[0,0], ic=False):
        # Equatorial cross section. ic=True also shows the inner core field (see kmode's ic)

        itheta = np.argmin( abs( self.theta - np.pi/2 ) )
        dat_tmp,titl,field = self.get_data(field)
        deq = dat_tmp[:,itheta,:]
        r = self.r
        if ic:  # append the inner core, without its first radius (the ICB, already the last of r)
            dat_ic = self.get_data(field, ic=True)[0]
            deq = np.concatenate( (deq, dat_ic[1:,itheta,:]), axis=0 )
            r = np.r_[ r, self.r_ic[1:] ]

        data = np.zeros([ len(r), self.nphi*max(1,self.m) + 1])
        data[:,:-1] = np.tile( deq, max(1,self.m) )
        data[:, -1] = data[:,0]

        plt.figure(figsize=(11,9))

        if cmap is None:
            cmap = default_cmap(field)

        cont = eqContour(r, self.phi, data.T,
                         levels=levels, cmap=cmap, clim=clim)
        if ic:
            plt.plot(self.ricb*np.cos(self.phi), self.ricb*np.sin(self.phi), 'k', lw=0.6)

        if titl:
            titl = titl + ' at equator'
            plt.title(titl,fontsize=20)

        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
        plt.tight_layout()
        plt.show()


    def inner_core_field(self, nr_ic=None, phase=0):
        '''
        Magnetic field inside the inner core (field='b', ricb > 0), on the radii r_ic (from the ICB down to,
        but not including, the centre), set as br_ic, btheta_ic, bphi_ic with the same angular grid as the
        field in the fluid. Depends on par.innercore:
            'conducting, Chebys'        : the inner core solution (real/imag_magnetic_ic.field)
            'insulator', 'TWA'          : the potential field matching the fluid field at the ICB (see potextra)
            'perfect conductor, ...'    : zero (no field perturbation inside a perfect conductor)
            'conducting, Bessel'        : not computed (its ICB condition in assemble.bc_b_icb needs checking first)
        '''
        from types import SimpleNamespace
        par, ut = self.par, self.ut

        if self.field != 'b' or self.ricb == 0:
            raise ValueError("the inner core field needs field='b' and ricb > 0")

        nr_ic = self.nr if nr_ic is None else nr_ic
        i = np.arange(0, nr_ic-1)
        xk = np.r_[ 1, np.cos( (i+0.5)*np.pi/nr_ic ) ]  # include the ICB but not the centre
        self.r_ic = 0.5*self.ricb*(xk+1)
        shape = (nr_ic, self.ntheta, self.nphi)

        if par.innercore == 'conducting, Chebys':

            # same layout as a full sphere of radius ricb: Chebyshev polynomials of x = r/ricb, of the parity
            # of the induced field b (as in bc_variables), N_cic/2 coefficients per l, poloidal then toroidal
            a = np.loadtxt('real_magnetic_ic.field', usecols=self.solnum)
            b = np.loadtxt('imag_magnetic_ic.field', usecols=self.solnum)
            sol = (a + 1j*b)*np.exp(1j*phase)
            nb, Nic = int((par.lmax_cic - self.m + 1)/2), ut.Nic
            s  = int( (ut.bsymm + 1)/2 )
            iP = (self.m + 1 - s)%2
            iT = (self.m + s)%2
            Pic = np.zeros((nb, par.N_cic), dtype=complex)
            Tic = np.zeros((nb, par.N_cic), dtype=complex)
            Pic[:, iP::2] = np.reshape(sol[:nb*Nic], (nb, Nic))
            Tic[:, iT::2] = np.reshape(sol[nb*Nic:2*nb*Nic], (nb, Nic))

            if self.ntheta <= par.lmax_cic + 1:
                raise ValueError('lmax_cic = %d needs more points in theta than %d: increase nphi' % (par.lmax_cic, self.ntheta))
            chx = ch.chebvander(self.r_ic/self.ricb, par.N_cic-1)
            M = SimpleNamespace(lmax=par.lmax_cic, m=self.m, nr=nr_ic, r=self.r_ic, ricb=0, rcmb=self.ricb,
                                ntheta=self.ntheta, nphi=self.nphi)
            out = spec2spat_vec(M, ut, chx, Pic, Tic, ut.bsymm, self.nthreads)
            assert np.allclose(M.theta, self.theta) and M.nphi == self.nphi
            self.br_ic, self.btheta_ic, self.bphi_ic = out[4:]

        elif par.innercore in ['insulator', 'TWA']:
            self.br_ic, self.btheta_ic, self.bphi_ic = self.potextra(self.r_ic)

        elif 'perfect' in par.innercore:
            self.br_ic, self.btheta_ic, self.bphi_ic = np.zeros(shape), np.zeros(shape), np.zeros(shape)

        else:
            print("innercore = '%s': the field inside the inner core is not computed" % par.innercore)
            del self.r_ic


    def potextra(self, rout, backend='auto'):
        '''
        Potential field of the magnetic field (field='b') outside the fluid shell, at the radii rout, on the grid
        of this mode: r >= rcmb outside the CMB (insulating mantle), and 0 < r <= ricb inside an inner core that is
        insulating inside (innercore = 'insulator' or 'TWA'). rout may mix both, but no radius may lie in the
        fluid. The poloidal scalars on the boundaries come directly from the Chebyshev coefficients
        (T_k(1) = 1 at the CMB, T_k(-1) = (-1)**k at the ICB). See libkoreviz.potential_field.
        backend: 'shtns', 'scipy' or 'auto' (SHTns if available, else scipy).
        Returns br, btheta, bphi at rout, each with shape (len(rout), ntheta, nphi).
        '''

        if self.field != 'b':
            raise ValueError("Potential extrapolation only valid when field='b'")

        rout = np.atleast_1d(rout)
        Pcmb = self.Plj @ ch.chebvander( 1.0, self.N-1).ravel()  # P_l(rcmb)
        Picb = None
        if self.ricb > 0 and np.any(rout <= self.ricb*(1 + 1e-12)):
            if self.par.innercore not in ['insulator', 'TWA']:
                raise ValueError("the field inside a '%s' inner core is not a potential field" % self.par.innercore)
            Picb = self.Plj @ ch.chebvander(-1.0, self.N-1).ravel()  # P_l(ricb)

        return potential_field(Pcmb, Picb, self.lpol, self.m, self.rcmb, self.ricb, rout, self.theta, self.nphi,
                               nthreads=self.nthreads, backend=backend)

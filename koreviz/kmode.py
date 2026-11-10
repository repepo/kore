import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch
from glob import glob
#import cmasher as cmr
from .plotlib import add_colorbar,default_cmap,radContour,merContour,eqContour
from .libkoreviz import spec2spat_vec,spec2spat_scal
import sys
import os


# NEW (2026-11-08, from conductive_IC): the solution files as kore writes them, name.npy (field_format = 'npy', the default)
# or real_name.field and imag_name.field (field_format = 'text'), in the current folder
def read_field(names, solnum):
    '''real and imaginary parts of solution solnum of the first of the fields names whose files exist'''
    for name in names:
        if os.path.isfile(name + '.npy'):
            x = np.load(name + '.npy', mmap_mode='r')[:, solnum]
            return np.array(x.real), np.array(x.imag)
        if os.path.isfile('real_' + name + '.field') and os.path.isfile('imag_' + name + '.field'):
            return (np.loadtxt('real_' + name + '.field', usecols=solnum),
                    np.loadtxt('imag_' + name + '.field', usecols=solnum))
    sys.exit('kmode: no solution file of %s (name.npy, or real_name.field and imag_name.field)' % ' or '.join(names))


class kmode:

    def __init__(self, vort=False,datadir='.',field='u', solnum=0,
                 nr=None, nphi=None, nthreads=4, phase=0, transform=True,
                 ic=False, nr_ic=None, mantle=False, nr_m=None, rmax=None):
        '''
        NEW (2026-11-10, after conductive_IC's ic option): with field = 'b',
        ic     : also the field inside the inner core (inner_core_field): br_ic, btheta_ic, bphi_ic on the radii r_ic
        nr_ic  : number of radii in the inner core (default nr)
        mantle : also the field beyond the outer wall (mantle_field): br_m, btheta_m, bphi_m on the radii r_m
        nr_m   : number of radii there (default nr), up to rmax (default r_surface with a resolved mantle, else 1.5 rcmb)
        The plots surf, merid and equat then take ic=True, mantle=True.
        '''

        sys.path.insert(0,datadir+'/bin')

        from parameters import par   # FIX (2026-11-08): par is the object in parameters.py
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
            a, b = read_field(['flow'], solnum)
            vsymm = par.symm
            vec = True
        elif field == 'b':
            a, b = read_field(['magnetic'], solnum)
            vsymm = ut.bsymm
            vec = True
        elif field in ['t','temp','temperature']:
            a, b = read_field(['thermal', 'temperature', 'temp'], solnum)   # NEW (2026-11-08): kore writes 'thermal'
            field='temperature'
            vsymm = par.symm
            vec = False
        elif field in ['comp','composition']:
            a, b = read_field(['composition'], solnum)
            field='composition'
            vsymm = par.symm
            vec = False

        # multiply by the complex phase factor. FIX (2026-11-10): spec2spat_vec and spec2spat_scal expand a full-sphere solution
        # (ricb = 0) from N1 to N coefficients themselves; expanded here too, it was cut and expanded twice
        aib = (a+1j*b)*(np.cos(phase)+1j*np.sin(phase))
        self.phase = phase
        self._sol, self._vsymm, self.upp = aib, vsymm, upp   # NEW (2026-11-10): for inner_core_field, mantle_field
        a = np.real(aib)
        b = np.imag(aib)

        if vec:

            sol = spec2spat_vec(self,ut,par,chx,a,b,vsymm,nthreads,
                               vort=vort,transform=transform)

            self.Q,self.S,self.T = sol[:3]

            if transform:
                if not vort:
                    if field == 'u':
                        self.ur, self.utheta, self.uphi = sol[3:]
                        if par.anelastic: #Comment the block out if you want momentum/mass flux
                            self.rho = rap.density(self.r)
                            for irho in range(self.nr):
                                self.ur[irho,...]     /= self.rho[irho]
                                self.utheta[irho,...] /= self.rho[irho]
                                self.uphi[irho,...]   /= self.rho[irho]
                    elif field == 'b':
                        self.br, self.btheta, self.bphi = sol[3:]
                        if ic:
                            self.inner_core_field(nr_ic)
                        if mantle:
                            self.mantle_field(nr_m, rmax)
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


    def get_data(self,field,region=''):
        '''region '' (the fluid), '_ic' (the inner core) or '_m' (beyond the outer wall): the magnetic field only'''

        field = field.lower()
        if region:
            comp = {'br':'br', 'brad':'br', 'bt':'btheta', 'btheta':'btheta', 'bp':'bphi', 'bphi':'bphi'}
            if field not in comp or not hasattr(self, 'br' + region):
                raise ValueError("%s: only br, btheta, bphi, with kmode(field='b', %s=True)" %
                                 (field, 'ic' if region == '_ic' else 'mantle'))
            titl = self.get_data(field)[1]
            return getattr(self, comp[field] + region), titl

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
        # Surface plot at constant radius. NEW (2026-11-10): below the ICB the inner core's field, beyond the CMB the
        # field there, where kmode computed them (ic, mantle)

        rgrid, reg = self.r, ''
        if r < self.ricb and hasattr(self, 'br_ic'):
            rgrid, reg = self.r_ic, '_ic'
        elif r > self.rcmb and hasattr(self, 'br_m'):
            rgrid, reg = self.r_m, '_m'
        ir = np.argmin(abs(rgrid-r))
        dat_tmp,titl = self.get_data(field, reg)
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


    def _walls(self, ic, mantle, curve):
        '''NEW (2026-11-10): the walls between the regions drawn, as thin lines'''
        for rw in ([self.ricb] if ic else []) + ([self.rcmb] if mantle else []):
            curve(rw)


    def merid(self, field='ur', azim=0, levels=48, cmap=None,
              colbar=True, titl=True, clim=[0,0], ic=False, mantle=False):
        # Meridional cross section. NEW (2026-11-10): ic=True, mantle=True add the field inside the inner core and beyond
        # the outer wall (kmode(field='b', ic=True, mantle=True))

        iphi = np.argmin(abs( self.phi - (azim*np.pi/180) )) % self.nphi
        dat_tmp, r, titl = self._regions(field, ic, mantle)
        data = dat_tmp[:,:,iphi]

        if field in ['energy','ener','e','ke']:
            #cmap = cmr.tropical_r
            data = np.log10(data)

        plt.figure(figsize=(6,9))

        if cmap is None:
            cmap = default_cmap(field)

        cont = merContour( r, self.theta, data.T,
                           levels=levels, cmap=cmap, clim=clim)
        self._walls(ic, mantle, lambda rw: plt.plot(rw*np.sin(self.theta), rw*np.cos(self.theta), 'k', lw=0.6))

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
              colbar=True, titl=True, clim=[0,0], ic=False, mantle=False):
        # Equatorial cross section. NEW (2026-11-10): ic=True, mantle=True as in merid

        itheta = np.argmin( abs( self.theta - np.pi/2 ) )
        dat_tmp, r, titl = self._regions(field, ic, mantle)
        data = np.zeros([ len(r), self.nphi*max(1,self.m) + 1])
        data[:,:-1] = np.tile( dat_tmp[:,itheta,:], max(1,self.m) )
        data[:, -1] = data[:,0]

        plt.figure(figsize=(11,9))

        if cmap is None:
            cmap = default_cmap(field)

        cont = eqContour(r, self.phi, data.T,
                         levels=levels, cmap=cmap, clim=clim)
        self._walls(ic, mantle, lambda rw: plt.plot(rw*np.cos(self.phi), rw*np.sin(self.phi), 'k', lw=0.6))

        if titl:
            titl = titl + ' at equator'
            plt.title(titl,fontsize=20)

        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # NEW (2026-11-10, after conductive_IC's inner_core_field and potential_field): the magnetic field inside the inner core
    # and beyond the outer wall, on the angular grid of the fluid's field (the same SHTns synthesis), so that the plots can
    # join them. Each region's field is l(l+1) f/r, f' + f/r and g per degree, as in the fluid (b = ∇×∇×(f r) + ∇×(g r)).

    def _lsets(self):
        '''the poloidal and toroidal degrees of b, with their rows in spec2spat's (l, r) arrays'''
        sb = int(self._vsymm*0.5 + 0.5)
        lm1 = self.lmax - self.m + 1
        if self.m > 0:
            idp, idt, ll = np.arange(1-sb, lm1, 2), np.arange(sb, lm1, 2), np.arange(self.m, self.lmax+1)
        else:
            idp, idt, ll = np.arange(sb, lm1, 2), np.arange(1-sb, lm1, 2), np.arange(1, self.lmax+2)
        return ll, idp, idt

    def _synth_b(self, r, f, f1, g):
        '''br, btheta, bphi (len(r), ntheta, nphi) from f, f' (poloidal degrees) and g (toroidal degrees), each (degrees, len(r))'''
        ll, idp, idt = self._lsets()
        lm1 = self.lmax - self.m + 1
        Q = np.zeros((lm1, len(r)), complex); S = np.zeros_like(Q); T = np.zeros_like(Q)
        L = (ll*(ll + 1))[idp, None]
        Q[idp] = L*f/r
        S[idp] = f1 + f/r
        T[idt] = g
        lmax2 = int(self.lmax + 1 - np.sign(self.m))
        nlm = (np.sign(self.m) + 1)*(lmax2 + 1) - self.m
        pad = (lambda a: np.r_[np.zeros((1, len(r))), a]) if self.m == 0 else (lambda a: a)
        out = np.zeros((3, len(r), self.ntheta, self.nphi))
        for ir in range(len(r)):
            q, sl, t = (np.zeros(nlm, complex) for _ in range(3))
            q[np.sign(self.m)*(lmax2+1):]  = pad(Q)[:, ir]
            sl[np.sign(self.m)*(lmax2+1):] = pad(S)[:, ir]
            t[np.sign(self.m)*(lmax2+1):]  = pad(T)[:, ir]
            out[:, ir] = self.sh.synth(q, sl, t)
        return out[0], out[1], out[2]

    def _wall_values(self):
        '''f, f' and g of the fluid's field at the outer wall and at the inner wall (per degree)'''
        Plj, Tlj = self.upp.expand_reshape_sol(self._sol, self._vsymm)
        out = {}
        for side, x in (('o', 1.0), ('i', -1.0)):
            v = ch.chebvander(x, self.N - 1).ravel()
            dP = np.array([self.ut.Dcheb(Plj[k], self.ricb, self.rcmb) for k in range(Plj.shape[0])])
            out[side] = (Plj @ v, dP @ v, Tlj @ v)
        return out

    def inner_core_field(self, nr_ic=None):
        '''
        The field inside the inner core (field = 'b', ricb > 0) on the radii r_ic, from the ICB down to (not including) the
        centre: br_ic, btheta_ic, bphi_ic. innercore 'conducting, Chebys': the core's own solution (magnetic_ic);
        'conducting, Bessel' (forced runs): f(R) j_l(kr)/j_l(kR) and mu_i2o g(R) j_l(kr)/j_l(kR) (utils.bessel_ic);
        'insulator' and 'TWA' (insulating beneath the layer): the potential field f(R) (r/R)^l.
        '''
        par, ut = self.par, self.ut
        if self.field != 'b' or self.ricb == 0:
            raise ValueError("the inner core field needs field='b' and ricb > 0")
        nr_ic = self.nr if nr_ic is None else nr_ic
        R = self.ricb
        i = np.arange(0, nr_ic - 1)
        r = 0.5*R*(np.r_[1, np.cos((i + 0.5)*np.pi/nr_ic)] + 1)   # the ICB first, not the centre
        ll, idp, idt = self._lsets()
        lp, lt = ll[idp], ll[idt]
        if par.innercore == 'conducting, Chebys':
            a, b = read_field(['magnetic_ic'], self.solnum)
            sol = (a + 1j*b)*np.exp(1j*self.phase)
            nb, Nic = len(lp), ut.Nic
            sb = int(self._vsymm*0.5 + 0.5)
            ixf, ixg = (self.m + 1 - sb) % 2, (self.m + sb) % 2   # the parities in x = r/R (as utils4pp.em_budget)
            Fc = np.zeros((nb, par.N_cic), complex); Gc = np.zeros_like(Fc)
            Fc[:, ixf::2] = sol[:nb*Nic].reshape(nb, Nic)
            Gc[:, ixg::2] = sol[nb*Nic:2*nb*Nic].reshape(nb, Nic)
            V = ch.chebvander(r/R, par.N_cic - 1)
            f = Fc @ V.T
            f1 = np.array([ch.chebder(c) for c in Fc]) @ ch.chebvander(r/R, par.N_cic - 2).T/R
            g = Gc @ V.T
        else:
            fR, _, gR = self._wall_values()['i']
            if par.innercore in ('insulator', 'TWA'):
                f  = fR[:, None]*(r/R)**lp[:, None]
                f1 = fR[:, None]*lp[:, None]*r**(lp[:, None] - 1)/R**lp[:, None]
                g  = np.zeros((len(lt), len(r)))
            elif par.innercore == 'conducting, Bessel':
                from scipy.special import spherical_jn
                kf = [ut.bessel_ic(l)[0] for l in lp]
                kg = [ut.bessel_ic(l)[0] for l in lt]
                f  = np.array([fR[c]*spherical_jn(l, k*r)/spherical_jn(l, k*R) for c, (l, k) in enumerate(zip(lp, kf))])
                f1 = np.array([fR[c]*k*spherical_jn(l, k*r, derivative=True)/spherical_jn(l, k*R)
                               for c, (l, k) in enumerate(zip(lp, kf))])
                g  = np.array([par.mu_i2o*gR[c]*spherical_jn(l, k*r)/spherical_jn(l, k*R) for c, (l, k) in enumerate(zip(lt, kg))])
            else:
                raise ValueError("innercore = '%s': no field inside the inner core" % par.innercore)
        self.r_ic = r
        self.br_ic, self.btheta_ic, self.bphi_ic = self._synth_b(r, f, f1, g)

    def mantle_field(self, nr_m=None, rmax=None):
        '''
        The field beyond the outer wall (field = 'b') on the radii r_m, from rmax down to (not including) the CMB:
        br_m, btheta_m, bphi_m. mantle 'resolved': the layer's own solution (magnetic_mantle) up to r_surface, and beyond
        it the potential field f(r_surface) (r_surface/r)^(l+1); 'insulator' and 'TWA': the potential field
        f(rcmb) (rcmb/r)^(l+1) (b_r is continuous through a thin layer). rmax defaults to r_surface with a resolved
        mantle and to 1.5 rcmb otherwise.
        '''
        par, ut = self.par, self.ut
        if self.field != 'b':
            raise ValueError("the field beyond the outer wall needs field='b'")
        nr_m = self.nr if nr_m is None else nr_m
        R1 = self.rcmb
        resolved = par.mantle == 'resolved'
        R2 = par.r_surface if resolved else R1
        rmax = (R2 if resolved else 1.5*R1) if rmax is None else rmax
        ll, idp, idt = self._lsets()
        lp, lt = ll[idp], ll[idt]
        i = np.arange(0, nr_m)
        r = R1 + (rmax - R1)*0.5*(np.cos((i + 0.5)*np.pi/nr_m) + 1)   # rmax side first, the CMB excluded
        r = np.r_[rmax, r] if rmax > R1 else r
        f = np.zeros((len(lp), len(r)), complex); f1 = np.zeros_like(f); g = np.zeros((len(lt), len(r)), complex)
        if resolved:
            a, b = read_field(['magnetic_mantle'], self.solnum)
            sol = (a + 1j*b)*np.exp(1j*self.phase)
            nb, Nm = len(lp), par.N_mantle
            Fc, Gc = sol[:nb*Nm].reshape(nb, Nm), sol[nb*Nm:2*nb*Nm].reshape(nb, Nm)
            inside = r <= R2*(1 + 1e-12)
            x = 2*(r[inside] - R1)/(R2 - R1) - 1
            f[:, inside]  = Fc @ ch.chebvander(x, Nm - 1).T
            f1[:, inside] = np.array([ch.chebder(c) for c in Fc]) @ ch.chebvander(x, Nm - 2).T*2/(R2 - R1)
            g[:, inside]  = Gc @ ch.chebvander(x, Nm - 1).T
            fo = Fc.sum(axis=1)   # f(r_surface)
        elif par.mantle in ('insulator', 'TWA'):
            fo = self._wall_values()['o'][0]
        else:
            raise ValueError("mantle = '%s': no field beyond the outer wall" % par.mantle)
        out = r > R2*(1 + 1e-12)
        if np.any(out):
            ro = r[out]
            f[:, out]  = fo[:, None]*(R2/ro)**(lp[:, None] + 1)
            f1[:, out] = -fo[:, None]*(lp[:, None] + 1)*R2**(lp[:, None] + 1)*ro**(-lp[:, None] - 2)
        self.r_m = r
        self.br_m, self.btheta_m, self.bphi_m = self._synth_b(r, f, f1, g)

    def _regions(self, field, ic, mantle):
        '''the data and radii of the regions asked for, outermost first'''
        parts = [('_m', 'r_m')] if mantle else []
        parts += [('', 'r')] + ([('_ic', 'r_ic')] if ic else [])
        data, rr = [], []
        for reg, rname in parts:
            d = self.get_data(field, reg)[0]
            r = getattr(self, rname)
            if reg == '_ic':   # the ICB is the fluid's last radius
                d, r = d[1:], r[1:]
            data.append(d); rr.append(r)
        return np.concatenate(data, axis=0), np.concatenate(rr), self.get_data(field)[1]


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

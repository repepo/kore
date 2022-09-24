import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch
import scipy.sparse as ss
import shtns
import cmasher as cmr
from .plotlib import *
import sys

sys.path.insert(1,'bin/')
import utils as ut
import parameters as par
import utils_pp as upp


class kmode:
    
    def __init__(self, field='u', solnum=0, nr=par.N+2, ntheta=par.lmax+3, nphi=10, nthreads=4 ):

        self.solnum = solnum
        self.nr     = nr
        self.lmax   = par.lmax
        self.m      = par.m
        self.symm   = par.symm
        self.N      = par.N
        self.ricb   = par.ricb
        self.rcmb   = 1
        self.n      = ut.n
        self.field  = field        
        m           = par.m
        lmax        = par.lmax
        ricb        = par.ricb
        rcmb        = 1
        gap         = rcmb - ricb
        n           = ut.n
        n0          = ut.n0
        
        # set the radial grid
        if ricb > 0:
            i = np.arange(0,nr-2)
            xk = np.r_[ 1, np.cos( (i+0.5)*np.pi/nr ), -1]  # include endpoints ricb and rcmb
        elif ricb==0:
            i = np.arange(0,nr-1)
            xk = np.r_[ 1, np.cos( (i+0.5)*np.pi/nr )    ]  # include rcmb but not the origin if ricb=0
        r = 0.5*gap*(xk+1) + ricb;
        x0 = upp.xcheb(r,ricb,rcmb)
        self.r = r
        
        # matrix with Chebyshev polynomials at every x point for all degrees:
        chx = ch.chebvander(x0,par.N-1) # this matrix has nr rows and N-1 cols
        
        # read fields from disk
        if field == 'u':
            a0 = np.loadtxt('real_flow.field',usecols=solnum)
            b0 = np.loadtxt('imag_flow.field',usecols=solnum)
            vsymm = par.symm
        elif field == 'b':
            a0 = np.loadtxt('real_magnetic.field',usecols=solnum)
            b0 = np.loadtxt('imag_magnetic.field',usecols=solnum)
            vsymm = ut.bsymm
        elif field == 't':
            a0 = np.loadtxt('real_temperature.field',usecols=solnum)
            b0 = np.loadtxt('imag_temperature.field',usecols=solnum)
            vsymm = par.symm
        
        # expand solution in case ricb=0
        aib = upp.expand_sol(a0+1j*b0,vsymm)
        a = np.real(aib)
        b = np.imag(aib)
        
        # Rearrange and separate poloidal and toroidal parts
        Plj0 = a[:n0] + 1j*b[:n0]         #  N elements on each l block
        Tlj0 = a[n0:n0+n0] + 1j*b[n0:n0+n0]   #  N elements on each l block
        
        lm1  = lmax-m+1    
        Plj  = np.reshape(Plj0,(int(lm1/2),par.N))
        Tlj  = np.reshape(Tlj0,(int(lm1/2),par.N))

        # init arrays
        Plr  = np.zeros( (lm1, nr), dtype=complex )
        Qlr  = np.zeros( (lm1, nr), dtype=complex )
        Slr  = np.zeros( (lm1, nr), dtype=complex )
        Tlr  = np.zeros( (lm1, nr), dtype=complex )
        dP   = np.zeros( (lm1, nr), dtype=complex )
        rP   = np.zeros( (lm1, nr), dtype=complex )
        dPlj = np.zeros(  np.shape(Plj), dtype=complex )
        
        # # These are the l values (ll) and indices (idp,idt)
        # sy = int(vsymm*0.5+0.5) # sy=0 if antisymm, sy=1 if symm
        # idp = np.arange( (np.sign(m)+sy  )%2, lm1, 2, dtype=int)
        # idt = np.arange( (np.sign(m)+sy+1)%2, lm1, 2, dtype=int)
        # ll = ut.ell(m,lmax,vsymm)[2]
        
        sy = int(vsymm*0.5+0.5) # s=0 if antisymm, s=1 if symm
        if m>0:
            idp = np.arange( 1-sy, lm1, 2)
            idt = np.arange( sy  , lm1, 2)
            ll  = np.arange( m, lmax+1 )
        elif m==0:
            idp = np.arange( sy  , lm1, 2)
            idt = np.arange( 1-sy, lm1, 2)
            ll  = np.arange( m+1, lmax+2 )

        # populate Plr and Tlr
        Plr[idp,:] = np.matmul( Plj, chx.T)
        Tlr[idt,:] = np.matmul( Tlj, chx.T)
        
        # print(np.amax(np.abs(Plr)))
        # print(np.amax(np.abs(Tlr)))
        
        # populate dPlj and dP
        for k in range(int(lm1/2)):
            dPlj[k,:] = ut.Dcheb(Plj[k,:], ricb, rcmb)
        dP[idp,:] = np.matmul(dPlj, chx.T)
        
        # populate Qlr and Slr
        rI = ss.diags(r**-1,0)
        L  = ss.diags(ll*(ll+1),0)
        rP  = Plr * rI  # P/r
        Qlr = L * rP    # l(l+1)*P/r
        Slr = rP + dP   # P' + P/r
        
        # Now in these Q, S, T arrays, the first lmax+1 indices are for m=0
        # and the remaining lmax+1-m are for mres.
        # (this is the SHTns way with m=mres when m is not zero)
        lmax2 = int( lmax + 1 - np.sign(m) )  # the true max value of l
        nlm = ( np.sign(m)+1 ) * (lmax2+1) - m
        self.Q = np.zeros([nr, nlm], dtype=complex)
        self.S = np.zeros([nr, nlm], dtype=complex)
        self.T = np.zeros([nr, nlm], dtype=complex)
        
        if m == 0 :  #pad with zeros for the l=0 component
            ql = np.r_[ np.zeros((1,nr)) ,Qlr ]
            sl = np.r_[ np.zeros((1,nr)) ,Slr ]
            tl = np.r_[ np.zeros((1,nr)) ,Tlr ]
        else :
            ql = Qlr
            sl = Slr
            tl = Tlr
            
        self.Q[:, np.sign(m)*(lmax2+1):] = ql.T
        self.S[:, np.sign(m)*(lmax2+1):] = sl.T
        self.T[:, np.sign(m)*(lmax2+1):] = tl.T
        self.ell = np.arange(m,lmax+1)
        
        self.Plr = Plr
        self.Plj = Plj
        self.dP  = dP
        self.rI  = rI
        self.L   = L
        self.rP  = rP
        self.Tlr = Tlr
        self.Tlj = Tlj
        
        # SHTns init
        #norm = shtns.sht_schmidt | shtns.SHT_NO_CS_PHASE
        norm = shtns.sht_schmidt
        self.mmax = int( np.sign(self.m) )
        self.mres = max(1,self.m)
        self.sh   = shtns.sht( lmax2, mmax=self.mmax, mres=self.mres, norm=norm, nthreads=nthreads )
        ntheta, nphi = self.sh.set_grid( ntheta+ntheta%2, nphi, shtns.sht_reg_poles, polar_opt=1e-10)
        self.theta = np.arccos(self.sh.cos_theta)
        self.phi   = np.linspace(0., 2*np.pi, nphi*self.mres+1, endpoint=True)

        # init the spatial component arrays
        self.ur     = np.zeros([nr, ntheta, nphi] )
        self.utheta = np.zeros([nr, ntheta, nphi] )
        self.uphi   = np.zeros([nr, ntheta, nphi] )
        self.ntheta = ntheta
        self.nphi   = nphi
        
        # the final call to shtns for each radius
        for ir in range(nr):
            self.sh.SHqst_to_spat( self.Q[ir,:], self.S[ir,:], self.T[ir,:], self.ur[ir,...], self.utheta[ir,...],  self.uphi[ir,...])
            
        

    def get_data(self,component): # Accesory function to select the desired vector component
        
        if   component in ['radial','rad','r']:
            data = self.ur
        elif component in ['theta','tht','t']:
            data = self.utheta
        elif component in ['phi','p']:
            data = self.uphi
        elif component in ['energy','ener','e']:
            data = (1/2)*(self.ur**2 + self.utheta**2 + self.uphi**2)
        elif component in ['amplitude','amp','a']:
            data = np.sqrt( (1/2)*(self.ur**2 + self.utheta**2 + self.uphi**2) )
        
        return data

        
 
    def surf(self, comp='rad', r=(par.ricb+1)/2, levels=48, cmap=cmr.prinsenvlag, colbar=True):
        # Surface plot at constant radius

        data = np.zeros([ self.ntheta, self.nphi*self.mres + 1])
        ir = np.argmin(abs(self.r-r))
        data[:,:-1] = np.tile( self.get_data(comp)[ir,...], self.mres )
        data[:, -1] = data[:,0]
 
        plt.figure(figsize=(12,6))
        cont = radContour( self.theta, self.phi, data.T, levels=levels, cmap=cmap)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=40)
        plt.tight_layout()
        plt.show()    
        

        
    def merid(self, comp='rad', azim=0, levels=48, cmap=cmr.prinsenvlag, colbar=True,limits=[0,0]):
        # Meridional cross section
        
        iphi = np.argmin(abs( self.phi - (azim*np.pi/180) )) % self.nphi
        data = self.get_data(comp)[:,:,iphi]
        
        if comp in ['energy','ener','e', 'amplitude','amp','a']:
            cmap = cmr.tropical_r
            #data = np.log10(data)

        plt.figure(figsize=(6,9))
        cont = merContour( self.r, self.theta, data.T, levels=levels, cmap=cmap,limits=limits)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
            cbar.ax.tick_params(labelsize=13) 
        plt.tight_layout()
        plt.show()



    def equat(self, comp='rad', levels=48, cmap=cmr.prinsenvlag, colbar=True):
        # Equatorial cross section
        
        data = np.zeros([ self.nr, self.nphi*self.mres + 1])
        itheta = np.argmin( abs( self.theta - np.pi/2 ) )
        data[:,:-1] = np.tile( self.get_data(comp)[:,itheta,:], self.mres )
        data[:, -1] = data[:,0]
        
        plt.figure(figsize=(11,9))
        cont = eqContour(self.r, self.phi, data.T, levels=levels, cmap=cmap)                
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
        plt.tight_layout()
        plt.show()


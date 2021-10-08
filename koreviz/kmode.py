import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch
import scipy.sparse as ss
import shtns
#import cmasher as cmr
from .plotlib import *
from .libkoreviz import *
import sys


class kmode:

    def __init__(self, datadir='.',field='u', solnum=0, nr=None, nphi=None, nthreads=4 ):

        sys.path.insert(0,datadir)

        import parameters as par
        import utils as ut

        self.solnum = solnum
        self.lmax   = par.lmax
        self.m      = par.m
        self.symm   = par.symm
        self.N      = par.N
        self.ricb   = par.ricb
        self.rcmb   = ut.rcmb
        self.n      = ut.n
        self.field  = field
        gap         = self.rcmb - self.ricb
        self.ut     = ut
        self.par    = par

        if nr is None:
            self.nr = par.N
        else:
            self.nr = nr

        if nphi is None:
            self.nphi = par.lmax * 3 # Orszag's 1/3 rule
        else:
            self.nphi = nphi

        self.ntheta = self.nphi // 2

        # set the radial grid
        i = np.arange(0,self.nr)
        x = np.cos( (i+0.5)*np.pi/self.nr )
        r = 0.5*gap*(x+1) + self.ricb;
        self.r = r
        if self.ricb == 0 :
            x0 = 0.5 + x/2
        else :
            x0 = x

        # matrix with Chebyshev polynomials at every x point for all degrees:
        chx = ch.chebvander(x0,par.N-1) # this matrix has nr rows and N-1 cols

        print(chx)

        # read fields from disk
        if field == 'u':
            a = np.loadtxt('real_flow.field',usecols=solnum)
            b = np.loadtxt('imag_flow.field',usecols=solnum)
            vsymm = par.symm
            vec = True
        elif field == 'b':
            a = np.loadtxt('real_magnetic.field',usecols=solnum)
            b = np.loadtxt('imag_magnetic.field',usecols=solnum)
            vsymm = -par.symm # because external mag field is antisymmetric wrt the equator (if dipole or axial)
            vec = True
        elif field in ['t','temp','temperature']:
            a = np.loadtxt('real_temperature.field',usecols=solnum)
            b = np.loadtxt('imag_temperature.field',usecols=solnum)
            vsymm = par.symm
            vec = False

        if vec:
            [ self.Q,self.S,self.T,
            vecR, vecTheta,vecPhi ] = spec2spat_vec(self,ut,par,chx,a,b,vsymm,nthreads)

            exec('self.'+field+'r'     + '= vecR')
            exec('self.'+field+'theta' + '= vecTheta')
            exec('self.'+field+'phi'   + '= vecPhi')

            del vecR
            del vecTheta
            del vecPhi

        else:
            self.Q,scal = spec2spat_scal(self,ut,par,chx,a,b,vsymm,nthreads)
            exec('self.'+field + '= scal')
            del scal


    def get_data(self,component): # Accesory function to select the desired vector component

        if   component in ['radial','rad','r']:
            data = self.ur
        elif component in ['theta','tht','t']:
            data = self.utheta
        elif component in ['phi','p']:
            data = self.uphi
        elif component in ['energy','ener','e']:
            data = (1/2)*(self.ur**2 + self.utheta**2 + self.uphi**2)

        return data



    def surf(self, comp='rad', r=0.5, levels=48, cmap='seismic', colbar=True):
        # Surface plot at constant radius

        data = np.zeros([ self.ntheta, self.nphi*self.m + 1])
        ir = np.argmin(abs(self.r-r))
        data[:,:-1] = np.tile( self.get_data(comp)[ir,...], self.m )
        data[:, -1] = data[:,0]

        plt.figure(figsize=(12,6))
        cont = radContour( self.theta, self.phi, data.T, levels=levels, cmap=cmap)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=40)
        plt.tight_layout()
        plt.show()



    def merid(self, comp='rad', azim=0, levels=48, cmap='seismic', colbar=True):
        # Meridional cross section

        iphi = np.argmin(abs( self.phi - (azim*np.pi/180) )) % self.nphi
        data = self.get_data(comp)[:,:,iphi]

        if comp in ['energy','ener','e']:
            cmap = cmr.tropical_r
            data = np.log10(data)

        plt.figure(figsize=(6,9))
        cont = merContour( self.r, self.theta, data.T, levels=levels, cmap=cmap)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
        plt.tight_layout()
        plt.show()



    def equat(self, comp='rad', levels=48, cmap='seismic', colbar=True):
        # Equatorial cross section

        data = np.zeros([ self.nr, self.nphi*self.m + 1])
        itheta = np.argmin( abs( self.theta - np.pi/2 ) )
        data[:,:-1] = np.tile( self.get_data(comp)[:,itheta,:], self.m )
        data[:, -1] = data[:,0]

        plt.figure(figsize=(11,9))
        cont = eqContour(self.r, self.phi, data.T, levels=levels, cmap=cmap)
        plt.axis('equal')
        plt.axis('off')
        if colbar:
            cbar = add_colorbar(cont,aspect=60)
        plt.tight_layout()
        plt.show()


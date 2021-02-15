import matplotlib.pyplot as plt
from .sht import *
from .plotlib import *
import sys
import os

class kmode(sol):
    def __init__(self,datDir,solnum=0,nr=100,nphi=None,ntheta=None):

        if datDir[-1] != '/':
            datDir += '/'
        if sys.path[0] != datDir:
            sys.path.insert(0,datDir)

        global ut, par
        import utils as ut
        import parameters as par

        self.solnum = solnum
        self.nr     = nr
        self.lmax   = par.lmax
        self.m      = par.m
        self.symm   = par.symm
        self.N      = par.N
        self.Ek     = par.Ek
        self.ricb   = par.ricb
        self.rcmb   = 1
        self.n      = ut.n

        if nphi is None or ntheta is None:
            self.nphi   = int(3 * self.lmax/2) * 2
            self.ntheta = int(self.nphi/2)

        if self.ntheta%2 != 0:
            self.ntheta -= 1
            self.nphi = self.ntheta*2

        sol.__init__(self,self.solnum,self.lmax,self.m,self.symm,self.N,self.Ek,
                     self.ricb,self.rcmb,self.n,self.nr,self.ntheta,self.nphi)

        out = sol.get_sol(self,datDir=datDir)

        # Unpacking

        if par.thermal == 1:
            if par.chemical == 1:
                if par.magnetic == 1:
                    [self.r,self.theta,self.phi,self.ur,self.utheta,self.uphi],\
                        [self.br,self.btheta,self.bphi],self.temp,self.chem = out
                else:
                    [self.r,self.theta,self.phi,self.ur,self.utheta,self.uphi],self.temp,self.chem = out
            else:
                if par.magnetic == 1:
                    [self.r,self.theta,self.phi,self.ur,self.utheta,self.uphi],[self.br,self.btheta,self.bphi],self.temp = out
                else:
                    [self.r,self.theta,self.phi,self.ur,self.utheta,self.uphi],self.temp = out
        elif par.magnetic == 1:
            [self.r,self.theta,self.phi,self.ur,self.utheta,self.uphi],[self.br,self.btheta,self.bphi] = out
        else:
            [self.r,self.theta,self.phi,self.ur,self.utheta,self.uphi] = out[0]


    def get_data(self,field):

        field = field.lower()

        if field in ['ur','vr']:
            data = self.ur
            titl = r'$u_r$'

        if field in ['up','vp']:
            data = self.uphi
            titl = r'$u_\phi$'

        if field in ['ut','vt']:
            data = self.utheta
            titl = r'$u_\theta$'

        if field == 'br':
            data = self.br
            titl = r'$B_r$'

        if field == 'bp':
            data = self.bphi
            titl = r'$B_\phi$'

        if field == 'bt':
            data = self.btheta
            titl = r'$B_\theta$'

        if field in ['t','temp']:
            data = self.temp
            titl = r'Temperature'

        if field in ['c','chem']:
            data = self.chem
            titl = r'Composition'

        return data, titl

    def surf(self,field='ur',r=0.5,cm='seismic',levels=30,cmap='RdBu_r'):

        idxPlot = find_rad(self.r,r)

        data, titl = self.get_data(field)

        data = data[...,idxPlot]

        plt.figure(figsize=(12,6))
        radContour(self.theta,self.phi,data,levels=levels,cmap=cmap)

        titl = titl + r' at $r/r_o = %.2f$' %(self.r[idxPlot]/self.r.max())
        plt.title(titl,fontsize=30)

        plt.axis('off')

        plt.show()

    def slice(self, field='ur',phi=0,levels=30,cmap='RdBu_r'):

        phi *= np.pi/180.
        idxPlot = find_phi(self.phi,phi)

        data, titl = self.get_data(field)
        data = data[idxPlot,...]

        plt.figure(figsize=(5,10))

        merContour(self.r,self.theta,data,levels=levels,cmap=cmap)

        titl = titl + r' at $\phi=%.1f^\circ$' %(self.phi[idxPlot] * 180/np.pi)
        plt.title(titl,fontsize=20)

        plt.axis('off')

        plt.show()

    def equat(self,field='ur',levels=30,cmap='RdBu_r'):

        idxPlot = int(self.ntheta/2)

        data, titl = self.get_data(field)
        data = data[:,idxPlot,:]

        plt.figure(figsize=(6.2,5))

        eqContour(self.r,self.phi,data,levels=levels,cmap=cmap)

        titl = titl + ' at equator'
        plt.title(titl,fontsize=20)

        plt.axis('off')
        plt.show()

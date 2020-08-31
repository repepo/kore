import matplotlib.pyplot as plt
from .sht import *
from .plotlib import *
import sys
sys.path.append('/home/ankit/kore/bin')

import utils as ut
import parameters as par

class kmode(sol):
    def __init__(self,solnum=0,nr=100,nphi=None,ntheta=None):

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
        
        sol.__init__(self,self.solnum,self.lmax,self.m,self.symm,self.N,self.Ek,
                     self.ricb,self.rcmb,self.n,self.nr,self.ntheta,self.nphi)
        
        self.r,self.theta,self.phi,self.ur,self.utheta,self.uphi = sol.get_sol(self,datDir='/home/ankit/kore/bin/')

    def surf(self,field='ur',r=0.5,cm='seismic',levels=30,cmap='RdBu_r'):
            
        idxPlot = find_rad(self.r,r)

        plt.figure(figsize=(12,6))

        if field in ['ur','UR','uR','Ur']:
            data = self.ur[...,idxPlot]

        if field in ['up','UP','uP','Up']:
            data = self.uphi[...,idxPlot]

        if field in ['ut','UT','uT','Ut']:
            data = self.utheta[...,idxPlot]
        
        if field in ['br','BR','bR','Br']:
            data = self.ur[...,idxPlot]

        if field in ['bp','BP','bP','Bp']:
            data = self.uphi[...,idxPlot]

        if field in ['bt','BT','bT','Bt']:
            data = self.utheta[...,idxPlot]


        radContour(self.theta,self.phi,data,levels=levels,cmap=cmap)

        plt.axis('off')

        plt.show()

    def slice(self, field='ur',phi=0,levels=30,cmap='RdBu_r'):

        phi *= np.pi/180.
        idxPlot = find_phi(self.phi,phi)

        print(idxPlot)

        plt.figure(figsize=(5,10))

        if field in ['ur','UR','uR','Ur']:
            data = self.ur[idxPlot,...]

        if field in ['up','UP','uP','Up']:
            data = self.uphi[idxPlot,...]

        if field in ['ut','UT','uT','Ut']:
            data = self.utheta[idxPlot,...]
        
        if field in ['br','BR','buR','Br']:
            data = self.ur[idxPlot,...]

        if field in ['bp','BP','bP','Bp']:
            data = self.uphi[idxPlot,...]

        if field in ['bt','BT','bT','Bt']:
            data = self.utheta[idxPlot,...]


        merContour(self.r,self.theta,data,levels=levels,cmap=cmap)
        plt.axis('off')

        plt.show()
    
    def equat(self,field='ur',levels=30,cmap='RdBu_r'):

        idxPlot = int(self.ntheta/2)

        if field in ['ur','UR','uR','Ur']:
            data = self.ur[:,idxPlot,:]

        if field in ['up','UP','uP','Up']:
            data = self.uphi[:,idxPlot,:]

        if field in ['ut','UT','uT','Ut']:
            data = self.utheta[:,idxPlot,:]

        if field in ['br','BR','bR','Br']:
            data = self.ur[:,idxPlot,:]

        if field in ['bp','BP','bP','Bp']:
            data = self.uphi[:,idxPlot,:]

        if field in ['bt','BT','bT','Bt']:
            data = self.utheta[:,idxPlot,:]
         
        eqContour(self.r,self.phi,data,levels=levels,cmap=cmap)

        plt.axis('off')
        plt.show()

        
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def find_rad(r,rPlot):
    return np.argmin(abs(r-rPlot))

def find_phi(phi,phiPlot):
    return np.argmin(abs(phi - phiPlot))

def hammer2cart(ttheta, pphi, colat=False):
    """
    This function is used to define the Hammer projection used when
    plotting surface contours
    """

    if not colat: # for lat and phi \in [-pi, pi]
        xx = 2.*np.sqrt(2.) * np.cos(ttheta)*np.sin(pphi/2.)\
             /np.sqrt(1.+np.cos(ttheta)*np.cos(pphi/2.))
        yy = np.sqrt(2.) * np.sin(ttheta)\
             /np.sqrt(1.+np.cos(ttheta)*np.cos(pphi/2.))
    else:  # for colat and phi \in [0, 2pi]
        xx = -2.*np.sqrt(2.) * np.sin(ttheta)*np.cos(pphi/2.)\
             /np.sqrt(1.+np.sin(ttheta)*np.sin(pphi/2.))
        yy = np.sqrt(2.) * np.cos(ttheta)\
             /np.sqrt(1.+np.sin(ttheta)*np.sin(pphi/2.))
    return xx, yy

def radContour(theta,phi,dat,levels=30,cmap='RdBu_r',colbar=True):

    phi2D, theta2D = np.meshgrid(phi,theta,indexing='ij')
    xx,yy = hammer2cart(theta2D,phi2D,colat=True)

    datMax = (np.abs(dat)).max()
    divnorm = colors.TwoSlopeNorm(vmin=-datMax, vcenter=0, vmax=datMax)
    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap,norm=divnorm)

    for c in cont.collections:
        c.set_edgecolor("face")

    thB = np.linspace(np.pi/2, -np.pi/2, len(theta))
    xxout, yyout  = hammer2cart(thB, -np.pi-1e-3)
    xxin, yyin  = hammer2cart(thB, np.pi+1e-3)

    plt.plot(xxout,yyout,'k',lw=1)
    plt.plot(xxin,yyin,'k',lw=1)

    if colbar:
        cbar = plt.colorbar(cont)

def merContour(r,theta,dat,levels=30,cmap='RdBu_r',colbar=True):

    theta2D, r2D = np.meshgrid(theta,r,indexing='ij')
    xx = r2D * np.sin(theta2D)
    yy = r2D * np.cos(theta2D)

    datMax = (np.abs(dat)).max()
    divnorm = colors.TwoSlopeNorm(vmin=-datMax, vcenter=0, vmax=datMax)
    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap,norm=divnorm)

    plt.plot(r[0]*np.sin(theta),r[0]*np.cos(theta),'k',lw=1)
    plt.plot(r[-1]*np.sin(theta),r[-1]*np.cos(theta),'k',lw=1)
    plt.plot([0,0], [ r.min(),r.max() ], 'k', lw=1)
    plt.plot([0,0], [ -r.max(),-r.min() ], 'k', lw=1)

    for c in cont.collections:
        c.set_edgecolor("face")

    if colbar:
        cbar = plt.colorbar(cont)

def eqContour(r,phi,dat,levels=30,cmap='RdBu_r',colbar=True):

    phi2D, r2D = np.meshgrid(phi,r,indexing='ij')
    xx = r2D * np.cos(phi2D)
    yy = r2D * np.sin(phi2D)

    datMax = (np.abs(dat)).max()
    divnorm = colors.TwoSlopeNorm(vmin=-datMax, vcenter=0, vmax=datMax)
    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap,norm=divnorm)

    plt.plot(r[0]*np.cos(phi), r[0]*np.sin(phi),'k',lw=1)
    plt.plot(r[-1]*np.cos(phi), r[-1]*np.sin(phi),'k',lw=1)

    for c in cont.collections:
        c.set_edgecolor("face")

    if colbar:
        cbar = plt.colorbar(cont)

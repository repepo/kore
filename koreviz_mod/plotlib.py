import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def add_colorbar(im, aspect=40, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


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


def radContour(theta,phi,dat,levels=42,cmap='RdBu_r',limits=[0,0]):

    phi2D, theta2D = np.meshgrid(phi,theta,indexing='ij')
    xx,yy = hammer2cart(theta2D,phi2D,colat=True)

    if limits[0]==limits[1]:
        if (dat.min()<0) and (dat.max()>0) :
            datMax = (np.abs(dat)).max()
            datMin = -datMax
        else:
            datMax = dat.max()
            datMin = dat.min()
    else:
        datMin = min(limits)
        datMax = max(limits)
    datCenter = (datMin+datMax)/2

    divnorm = colors.TwoSlopeNorm(vmin=datMin, vcenter=datCenter, vmax=datMax)
    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap,norm=divnorm)

    for c in cont.collections:
        c.set_edgecolor("face")

    thB = np.linspace(np.pi/2, -np.pi/2, len(theta))
    xxout, yyout  = hammer2cart(thB, -np.pi-1e-3)
    xxin, yyin  = hammer2cart(thB, np.pi+1e-3) 
 
    plt.plot(xxout,yyout,'k',lw=0.6)
    plt.plot(xxin,yyin,'k',lw=0.6)
 
    return cont


def merContour(r,theta,dat,levels=42,cmap='RdBu_r',limits=[0,0]):

    theta2D, r2D = np.meshgrid(theta,r,indexing='ij')
    xx = r2D * np.sin(theta2D)
    yy = r2D * np.cos(theta2D)

    if limits[0]==limits[1]:
        if (dat.min()<0) and (dat.max()>0) :
            datMax = (np.abs(dat)).max()
            datMin = -datMax
        else:
            datMax = dat.max()
            datMin = dat.min()
    else:
        datMin = min(limits)
        datMax = max(limits)
    datCenter = (datMin+datMax)/2
    
    divnorm = colors.TwoSlopeNorm(vmin=datMin, vcenter=datCenter, vmax=datMax)
    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap,norm=divnorm)

    plt.plot(r[0]*np.sin(theta),r[0]*np.cos(theta),'k',lw=0.6)
    plt.plot(r[-1]*np.sin(theta),r[-1]*np.cos(theta),'k',lw=0.6)
    plt.plot([0,0], [ r.min(),r.max() ], 'k', lw=0.6)
    plt.plot([0,0], [ -r.max(),-r.min() ], 'k', lw=0.6)

    for c in cont.collections:
        c.set_edgecolor("face")

    return cont


def eqContour(r,phi,dat,levels=42,cmap='RdBu_r',limits=[0,0]):

    phi2D, r2D = np.meshgrid(phi,r,indexing='ij')
    xx = r2D * np.cos(phi2D)
    yy = r2D * np.sin(phi2D)

    if limits[0]==limits[1]:
        if (dat.min()<0) and (dat.max()>0) :
            datMax = (np.abs(dat)).max()
            datMin = -datMax
        else:
            datMax = dat.max()
            datMin = dat.min()
    else:
        datMin = min(limits)
        datMax = max(limits)
    datCenter = (datMin+datMax)/2
	
    divnorm = colors.TwoSlopeNorm(vmin=datMin, vcenter=datCenter, vmax=datMax)
    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap,norm=divnorm)

    plt.plot(r[0]*np.cos(phi), r[0]*np.sin(phi),'k',lw=0.6)
    plt.plot(r[-1]*np.cos(phi), r[-1]*np.sin(phi),'k',lw=0.6)

    for c in cont.collections:
        c.set_edgecolor("face")
    
    return cont

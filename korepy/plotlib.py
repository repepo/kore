import matplotlib.pyplot as plt
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

    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap)

    if colbar:
        cbar = plt.colorbar(cont)
    
def merContour(r,theta,dat,levels=30,cmap='RdBu_r',colbar=True):

    theta2D, r2D = np.meshgrid(theta,r,indexing='ij')
    xx = r2D * np.sin(theta2D)
    yy = r2D * np.cos(theta2D)

    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap)

    if colbar:
        cbar = plt.colorbar(cont)

def eqContour(r,phi,dat,levels=30,cmap='RdBu_r',colbar=True):

    phi2D, r2D = np.meshgrid(phi,r,indexing='ij')
    xx = r2D * np.cos(phi2D)
    yy = r2D * np.sin(phi2D)

    cont = plt.contourf(xx,yy,dat,levels,cmap=cmap)

    if colbar:
        cbar = plt.colorbar(cont)

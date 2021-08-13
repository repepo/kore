import numpy as np

# some useful targets for the solver

def som(alpha,E):
    '''
    spin-over freq and damping according to Zhang (2004)
    '''
    eps2 = 2*alpha - alpha**2
    reG = -2.62047 - 0.42634 * eps2
    imG = 0.25846 + 0.76633 * eps2
    sigma = 1./(2.-eps2) #inviscid freq
    G = reG + 1.j*imG
    return  1j*( 2*sigma - 1j*G * E**0.5 )


def wattr(n,Ek):
    '''
    Useful to set targets when reproducing Figs 3 and 4 of Rieutord & Valdettaro, JFM (2018)
    See also equation (3.2) in that paper
    '''
    w0 = 0.782413
    tau1 = 0.485
    phi1 = -np.pi/3
    tau2 = 1.82
    phi2 = -np.pi/4
    out0 = 1j*w0
    out1 = -2*tau1*( np.cos(phi1)+1j*np.sin(phi1) )*(Ek**(1/3))
    out2 = -(n+0.5)*np.sqrt(2)*tau2*( np.cos(phi2)+1j*np.sin(phi2) )*(Ek**(1/2))
    return out0+out1+out2

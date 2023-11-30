import sys
import numpy as np

'''
Reads results collected with the reap.sh script into python.
Using IPython do:

%run -i getresults.py somename

where 'somename' is the prefix used when creating and collecting
the results, i.e. with dodirs.sh and reap.sh
'''



if len(sys.argv) == 2:
    u = np.loadtxt(sys.argv[1]+'.flo') # flow data
    p = np.loadtxt(sys.argv[1]+'.par') # parameters
else:
    u = np.loadtxt('flow.dat')         # flow data
    p = np.loadtxt('params.dat')       # parameters
    
if len(u.shape)==1:
    u = u.reshape((-1,len(u)))
    p = p.reshape((-1,len(p)))
    
KP = u[:,1]                         # Poloidal kinetic energy
KT = u[:,2]                         # Toroidal kinetic energy
K  = u[:,0]
t2p = KT/KP
p2t = KP/KT

ricb = p[:,7]                       # inner core radius
wf   = p[:,11]                       # forcing frequency
Ek   = p[:,4]                       # Ekman number
ek   = np.log10(Ek).round(decimals=6)

Dkin = u[:,3]                    # Kinetic energy dissipation
Dint = u[:,4]                    # Internal energy dissipation


forcing = p[:,10]

bci = p[:,8]                        # inner core boundary condition
bco = p[:,9]                        # cmb boundary condition



N    = p[:,46]
lmax = p[:,47]


if sum(forcing) == 0:   # reads eigenvalue data
    
    if len(sys.argv) == 2:
        w = np.loadtxt(sys.argv[1]+'.eig')
    else:
        w = np.loadtxt('eigenvalues.dat')
    if len(w.shape)==1: 
        w = w.reshape((-1,len(w)))

    sigma = w[:,0]
    scd = -sigma/np.sqrt(Ek)
    

# cmb viscous torque, use 2*real part of this
trq = u[:,10] + 1j*u[:,11]

# icb viscous torque, use 2*real part of this
trq_icb = u[:,12] + 1j*u[:,13]
    

tsol = p[:,48]
ncpus = p[:,45]
    


hydro = p[:,0]
magnetic = p[:,1]
thermal = p[:,2]
compositional = p[:,3]

resid0 = u[:,8]
resid1 = u[:,9]

	
    
if sum(magnetic) == np.shape(p)[0]: # reads magnetic data

    if len(sys.argv) == 2:
        b = np.loadtxt(sys.argv[1]+'.mag')
    else:
        b = np.loadtxt('magnetic.dat')
        
    if len(b.shape)==1: 
        b = b.reshape((-1,len(b)))
        
    M    = b[:,0]         # Total magnetic field energy
    Le2  = p[:,26]                  # Lehnert number squared
    Em   = p[:,25]                  # Magnetic Ekman number

    Le   = np.sqrt(Le2)                # Lehnert number
    Pm   = Ek/Em                    # Magnetic Prandtl number
    Lam  = Le2/Em                   # Elsasser number

    pm = np.log10(Pm).round(decimals=4)
    ss = np.log10(Lam).round(decimals=4);

    mu_i2o = p[:,49]
    sigma_i2o = p[:,50]

    resid2 = b[:,3]    

        

    



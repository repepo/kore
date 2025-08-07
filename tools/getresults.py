import sys
import numpy as np

'''
Reads results collected with the reap.sh script into python.
Using IPython do:

%run -i getresults.py somename

where 'somename' is the prefix used when creating and collecting
the results, i.e. with dodirs.sh and reap.sh
'''

'''
params:  [Ek, m, symm, ricb, bci, bco, projection, forcing, \
            forcing_amplitude, forcing_frequency, magnetic, Em, Le2, N, lmax, toc1-tic, ncpus]
            
ken_dis: [KP, KT, internal_dis, rekin_dis, imkin_dis, repower, impower]
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
K  = KP + KT
t2p = KT/KP
p2t = KP/KT

ricb = p[:,3]                       # inner core radius
wf   = p[:,9]                       # forcing frequency
Ek   = p[:,0]                       # Ekman number
ek   = np.log10(Ek).round(decimals=6)

Dkin0 = u[:,3]                    # Kinetic energy dissipation
Dint0 = u[:,2]                    # Internal energy dissipation
rpow = u[:,5]                       # Input power for forced problems, real part
ipow = u[:,6]                       # Input power for forced problems, imaginary part


forcing = p[:,7]

bci = p[:,4]                        # inner core boundary condition
bco = p[:,5]                        # cmb boundary condition
amp = p[:,8]                        # forcing amplitude (cmb)

if np.shape(p)[1]>=15:
    N    = p[:,13]
    lmax = p[:,14]


if sum(forcing) == 0:   # reads eigenvalue data
    
    if len(sys.argv) == 2:
        w = np.loadtxt(sys.argv[1]+'.eig')
    else:
        w = np.loadtxt('eigenvalues.dat')
    if len(w.shape)==1: 
        w = w.reshape((-1,len(w)))

    sigma = w[:,0]
    pss = np.zeros( np.shape(p)[0] )
    pvf = np.zeros( np.shape(p)[0] )
    scd = -sigma/np.sqrt(Ek)
    
elif sum(forcing) == 7*np.shape(p)[0]:  # libration, boundary forcing
    sigma = np.zeros( np.shape(p)[0] )
    pvf = np.zeros( np.shape(p)[0] )
    pss = rpow

elif sum(forcing) == 8*np.shape(p)[0]:  # libration, volume forcing
    sigma = np.zeros( np.shape(p)[0] )
    pss = np.zeros( np.shape(p)[0] )
    pvf = rpow
    
elif sum(forcing) == 9*np.shape(p)[0]:  # m=2 boundary forcing, needs fixing
    sigma = np.zeros( np.shape(p)[0] )
    pss = rpow
    pvf = np.zeros( np.shape(p)[0] )
    
#if shape(u)[1]>=10:
    # viscous dissipation in the bulk, without boundary layers
    #vd1 = Dint - (u[:,7] + u[:,8])
    #vd2 = Dint - (u[:,7] + u[:,9])
    
if np.shape(u)[1]>=12:
    # cmb torque, use 2*real part of this
    trq = u[:,10] + 1j*u[:,11]
    
if np.shape(u)[1]>=14:
    # icb torque, use 2*real part of this
    trq_icb = u[:,12] + 1j*u[:,13]
    
if np.shape(p)[1]>=17:
    tsol = p[:,15]
    ncpus = p[:,16]
    
if np.shape(p)[1]>=21:
    tol = p[:,17]
    thermal = p[:,18]
    Prandtl = p[:,19]
    Brunt = p[:,20]
    
    # use this for the Ra_c, rXX_E-7 runs in the cluster
    #Ra_gap = p[:,18]
    #Prandtl = p[:,19]
    #thermal = p[:,20]  

'''   
magnetic = p[:,10]
if np.shape(p)[1]>=30:
    time_scale = p[:,29]
else:
	time_scale = 0
	
    
if sum(magnetic) == np.shape(p)[0]: # reads magnetic data

    if len(sys.argv) == 2:
        b = np.loadtxt(sys.argv[1]+'.mag')
    else:
        b = np.loadtxt('magnetic.dat')
        
    if len(b.shape)==1: 
        b = b.reshape((-1,len(b)))
        
    M0    = b[:,0] + b[:,1]          # Total magnetic field energy
    Le2  = p[:,12]                  # Lehnert number squared
    Em   = p[:,11]                  # Magnetic Ekman number
    Dohm0 = (b[:,2]+b[:,3])   # Ohmic dissipation
    Le   = np.sqrt(Le2)                # Lehnert number
    Pm   = Ek/Em                    # Magnetic Prandtl number
    Lam  = Le2/Em                   # Elsasser number

    pm = np.log10(Pm).round(decimals=4)
    ss = np.log10(Lam).round(decimals=4);
    
    #if shape(b)[1]>=7 :
    #    od1 = Dohm - (b[:,4] + b[:,5])
    #    od2 = Dohm - (b[:,4] + b[:,6])
    #    d1 = od1/vd1                # dissip ratio in the bulk, without boundary layers
    #    d2 = od2/vd2                # a bit deeper in the bulk
        
else:
    
    M0 = np.zeros( np.shape(p)[0] )
    Dohm0 = np.zeros( np.shape(p)[0] )

Le2  = p[:,12]                  # Lehnert number squared
Em   = p[:,11]                  # Magnetic Ekman number

if time_scale == 0:
    M = M0*( Le2 )

    Dint = Dint0*( Ek )
    Dkin = Dkin0*( Ek )
    Dohm = Dohm0*( Le2*Em )
elif time_scale == 1:
    M = M0

    Dint = Dint0*( Ek/Le )
    Dkin = Dkin0*( Ek/Le )
    Dohm = Dohm0*( Em/Le )
elif time_scale == 2:
    M = M0* ( Le2/(Ek**2) )

    Dint = Dint0
    Dkin = Dkin0
    Dohm = Dohm0* ( Le2*Em/(Ek**2) )

d = Dohm/Dint                   # Ohmic to viscous dissipation ratio



if sum(thermal) == np.shape(p)[0]: # reads thermal data
    
    if len(sys.argv) == 2:
        th = np.loadtxt(sys.argv[1]+'.thm')
    else:
        th = np.loadtxt('thermal.dat')
        
    #if len(th.shape)==1:    
    #    th = th.reshape((-1,len(th)))
    #    Dtemp = th[:,0]
    #elif size(th)==1:
    #    Dtemp = array([th])
    Dtemp = np.array([th])[0]

else:
    
    Dtemp = np.zeros( np.shape(p)[0] )
                
                
resid1 = abs( Dint + Dkin - pss ) / np.amax( [ abs(Dint), abs(Dkin), abs(pss) ], 0 )

resid2 = abs( 2*sigma*(K + M) - Dkin - Dtemp + Dohm - pvf ) / \
 np.amax( [ abs(2*sigma*(K+M)), abs(Dkin), abs(Dohm), abs(Dtemp), abs(pvf) ], 0 )

'''     

    



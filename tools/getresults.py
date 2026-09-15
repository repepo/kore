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
params[i,:] = np.array([
 {                      par.hydro,                      #0
                        par.magnetic,                   #1
                        par.thermal,                    #2
                        par.compositional,              #3

                        par.m,                          #4
                        par.symm,                       #5
                        par.ricb,                       #6
                        par.bci,                        #7
                        par.bco,                        #8

                        par.forcing,                    #9
                        par.forcing_frequency,          #10
                        par.forcing_amplitude_cmb,      #11
                        par.forcing_amplitude_icb,      #12
                        par.projection,                 #13

                        par.Gaspard,                    #14
                        par.Beyonce,                    #15
                        par.Hendrik,                    #16
                        par.ViscosD,                    #17
                        par.ThermaD,                    #18
                        par.MagnetD,                    #19

                        par.ncpus,                      #20
                        par.N,                          #21
                        par.lmax,                       #22

                        timing+toc-tic,                 #23

                        par.aux0,                       #24
                        par.aux1,                       #25
                        par.aux2,                       #26
                        par.aux3,                       #27
                        par.aux4,                       #28
                        par.aux5,                       #29

                        par.visc0,			            #30
                        par.hvisc,			            #31
                        par.rvisc			            #32
                        ])  # 33 total
}
            
postprocessing data: [ KE, KP, KT, Dkin, ldom, lwidth, lconv ]
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

ricb = p[:,6]
bci  = p[:,7]                        # inner core boundary condition
bco  = p[:,8]                        # cmb boundary condition
Ek   = p[:,17]                       # Reference Ekman number
N    = p[:,21]
lmax = p[:,22]
rp   = p[:,24]
A    = p[:,27]
nu   = p[:,30]

KE   = u[:,0]
KP   = u[:,1]
KT   = u[:,2]
Dkin = u[:,3]
ll   = u[:,4]
lw   = u[:,5]
lc   = u[:,6]

t2p  = KT/KP
ek   = np.log10(Ek).round(decimals=6)

forcing = p[:,9]
if sum(forcing) == 0:   # reads eigenvalue data
    
    if len(sys.argv) == 2:
        w = np.loadtxt(sys.argv[1]+'.eig')
    else:
        w = np.loadtxt('eigenvalues.dat')
    if len(w.shape)==1: 
        w = w.reshape((-1,len(w)))

    sigma = w[:,0]
    omg = w[:,1]

resid1 = np.ones_like(omg)
for i in range(len(omg)):
    resid1[i] = abs( 2*sigma[i]*KE[i] - Dkin[i] ) / max( (abs(2*sigma[i]*KE[i]), abs(Dkin[i])) )  
    #resid1[i] = abs( 2*sigma[i]*KE[i] - Dkin[i] ) / max( abs(2*sigma[i]*KE[i]), abs(Dkin[i]) )


    



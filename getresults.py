import sys
from numpy import loadtxt

'''
Reads results collected with the reap.sh script into python.

Using IPython do:

%run -i getresults.py somename

where 'somename' is the prefix used when creating and collecting
the results, i.e. with dodirs.sh and reap.sh
'''

#params:  [Ek, m, symm, ricb, bci, bco, projection, forcing, \
#			forcing_amplitude, forcing_frequency, magnetic, Em, Le2, N, lmax, toc1-tic, ncpus]
#			
#ken_dis: [KP, KT, internal_dis, rekin_dis, imkin_dis, repower, impower]


u = loadtxt(sys.argv[1]+'.flo')  # flow data
p = loadtxt(sys.argv[1]+'.par')  # parameters

	  
KP = u[:,0] # Poloidal kinetic energy
KT = u[:,1] # Toroidal kinetic energy
K  = KP + KT

ricb = p[:,3] # inner core radius
wf   = p[:,9] # forcing frequency
Ek   = p[:,0] # Ekman number

Dkin = u[:,3]*Ek # Kinetic energy dissipation
Dint = u[:,2]*Ek # Internal energy dissipation
rpow = u[:,5] # Input power from body forcing, real part
ipow = u[:,6] # Input power from body forcing, imaginary part
 
forcing = p[:,7] 

if sum(forcing)==0: # reads eigenvalue data
	w = loadtxt(sys.argv[1]+'.eig')
 
magnetic = p[:,10]
if sum(magnetic)>0: # reads magnetic data
	b = loadtxt(sys.argv[1]+'.mag')
	M = b[:,0] + b[:,1] # Total magnetic field energy
	Le2  = p[:,12] # Lehnert number squared
	Em   = p[:,11] # Magnetic Ekman number
	Dohm = (b[:,2]+b[:,3])*Le2*Em # Ohmic dissipation
	Le = sqrt(Le2) # Lehnert number



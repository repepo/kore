import sys
from numpy import loadtxt

'''
Script to assign variable names to the solver's output

params:  [Ek, m, symm, ricb, bci, bco, projection, forcing, \
			forcing_amplitude, forcing_frequency, magnetic, Em, Le2, N, lmax, toc1-tic, ncpus]
			
ken_dis: [KP, KT, internal_dis, rekin_dis, imkin_dis, repower, impower]
'''

u = loadtxt(sys.argv[1]+'.flo')
p = loadtxt(sys.argv[1]+'.par')

	  
KP = u[:,0]
KT = u[:,1]
K  = KP + KT

ricb = p[:,3]
wf   = p[:,9]
Ek   = p[:,0]

Dkin = u[:,3]*Ek
Dint = u[:,2]*Ek
rpow = u[:,5]
ipow = u[:,6]
 
forcing = p[:,7]

if sum(forcing)==0:
	w = loadtxt(sys.argv[1]+'.eig')
 
magnetic = p[:,10]
if sum(magnetic)>0:
	b = loadtxt(sys.argv[1]+'.mag')
	M = b[:,0] + b[:,1]
	Le2  = p[:,12]
	Em   = p[:,11]
	Dohm = (b[:,2]+b[:,3])*Le2*Em
	Le = sqrt(Le2)

import numpy as np
import sys
sys.path.insert(1,'bin/')
sys.path.insert(1,'koreviz/')


from koreviz import kmode as k
import parameters as par

u = k.kmode('u', 0, nr=2400, nphi=720)

#j = np.where(u.r>1-50*np.sqrt(par.Ek))[0]
j = np.where(u.r>0.99)[0]

urmax = np.amax(u.ur[j,:,:])
utmax = np.amax(u.utheta[j,:,:])
upmax = np.amax(u.uphi[j,:,:])

np.savetxt('umax.dat', np.c_[par.Ek, urmax, utmax, upmax] )

del u

if par.magnetic==1:
    
    b = k.kmode('b', 0, nr=2400, nphi=720)
    
    #j = np.where(b.r>1-50*np.sqrt(par.Ek))[0]
    j = np.where(b.r>0.99)[0]

    brmax = np.amax(b.ur[j,:,:])
    btmax = np.amax(b.utheta[j,:,:])
    bpmax = np.amax(b.uphi[j,:,:])

    brsurf = np.amax(b.ur[0,:,:])
    btsurf = np.amax(b.utheta[0,:,:])
    bpsurf = np.amax(b.uphi[0,:,:])
    
    np.savetxt('bmax.dat', np.c_[par.Ek, brmax, btmax, bpmax, brsurf, btsurf, bpsurf] )

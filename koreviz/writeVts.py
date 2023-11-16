#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

import numpy as np

try:
    try: # Version 2 changed naming convention of functions
        from evtk.hl import structuredToVTK
        gridToVTK = structuredToVTK
    except:
        import evtk
        gridToVTK = evtk.hl.gridToVTK
except:
    print("If you need 3D visualization:")
    print("writeVts requires the use of evtk library.")
    print("You can get it from https://github.com/paulo-herrera/PyEVTK")

def get_grid(r,theta,phi):

    r3D,th3D,phi3D = np.meshgrid(r,theta,phi,indexing='ij')

    s3D = r3D * np.sin(th3D)
    x3D = s3D * np.cos(phi3D)
    y3D = s3D * np.sin(phi3D)
    z3D = r3D * np.cos(th3D)

    return r3D,th3D,phi3D, x3D,y3D,z3D, s3D

def get_cart(vr,vt,vp,th3D,p3D):

    vs = vr * np.sin(th3D) + vt *np.cos(th3D)
    vz = vr * np.cos(th3D) - vt *np.sin(th3D)

    vx = vs * np.cos(p3D) - vp * np.sin(p3D)
    vy = vs * np.sin(p3D) + vp * np.cos(p3D)

    return vx,vy,vz

def tile_and_fix(data,m,nr,ntheta,nphi,step,potextra,bfield=False):

    if potextra:
        if not bfield:
            data = data[::-1,...]
            scal = np.zeros([nr,ntheta,nphi])
            scal_tile = (np.tile(data,m))[::step,::step,:]
            scal[:data.shape[0],:,:-1] = scal_tile
            scal[:data.shape[0],:,-1]  = scal_tile[...,0]
            scal = np.asfortranarray(scal)
        else:
            scal = np.zeros([nr,ntheta,nphi])
            scal_tile = (np.tile(data,m))
            scal[:data.shape[0],:,:-1] = scal_tile
            scal[:data.shape[0],:,-1]  = scal_tile[...,0]
            scal = np.asfortranarray(scal)
            del scal_tile
            return scal
    else:
        scal = np.zeros([nr,ntheta,nphi])
        scal_tile = (np.tile(data,m))[::step,::step,:]
        scal[:data.shape[0],:,:-1] = scal_tile
        scal[:data.shape[0],:,-1]  = scal_tile[...,0]
        scal = np.asfortranarray(scal)

    del scal_tile

    return scal

def writeVts(mode, scals=[],vecs=[],potextra=False,
             nrout=32,radratio=2.0,step=5):

    # Make everything case insensitive

    for k in range(len(scals)):
        scals[k] = scals[k].lower()

    for k in range(len(vecs)):
        vecs[k] = vecs[k].lower()

    # Figure out if magnetic field needs plotting

    plotb = ( any(elem in ['br','bphi','bp','bt','btheta'] for elem in scals) or
              any(elem in ["b"] for elem in vecs) )
    btile = True
    # Potential extrapolation? radratio = r_surface/r_cmb

    if plotb and potextra:
        rout = np.linspace(mode.r[0],radratio,nrout)
        brout,btout,bpout = mode.potextra(mode.br[0,...],mode.r[0],rout)
        r = np.concatenate((mode.r[::step][::-1],rout))
        nr = len(r)
        br     = mode.br[::step,::step,:]
        btheta = mode.btheta[::step,::step,:]
        bphi   = mode.bphi[::step,::step,:]
        br     = np.concatenate((br[::-1,...],    brout[:,::step,:]),axis=0)
        btheta = np.concatenate((btheta[::-1,...],btout[:,::step,:]),axis=0)
        bphi   = np.concatenate((bphi[::-1,...],  bpout[:,::step,:]),axis=0)
    elif plotb and not potextra:
        r = mode.r[::step]
        nr = len(r)
        br     = mode.br
        btheta = mode.btheta
        bphi   = mode.bphi
    else:
        r = mode.r[::step]
        nr = len(r)

    theta = mode.theta[::step]
    ntheta = len(theta)
    nphi   = mode.phi.shape[0]

    r3D,th3D,p3D, x3D,y3D,z3D, s3D = get_grid(r,theta,mode.phi)

    keys = []
    values = []

    keys.append("radius")
    keys.append("cyl_radius")

    values.append(r3D)
    values.append(s3D)



    if any(elem in ["u","v"] for elem in vecs):

        # Tiling and steps

        ur = tile_and_fix(mode.ur,mode.m,nr,ntheta,nphi,step,potextra)
        ut = tile_and_fix(mode.utheta,mode.m,nr,ntheta,nphi,step,potextra)
        up = tile_and_fix(mode.uphi,mode.m,nr,ntheta,nphi,step,potextra)

        ux,uy,uz = get_cart(ur,ut,up,th3D,p3D)

        ux = np.asfortranarray(ux)
        uy = np.asfortranarray(uy)
        uz = np.asfortranarray(uz)

        keys.append("vecV")
        values.append((ux,uy,uz))

    if any(elem in ["b"] for elem in vecs):

        # Tiling and steps

        br = tile_and_fix(br,mode.m,nr,ntheta,nphi,step,potextra,bfield=True)
        bt = tile_and_fix(btheta,mode.m,nr,ntheta,nphi,step,potextra,bfield=True)
        bp = tile_and_fix(bphi,mode.m,nr,ntheta,nphi,step,potextra,bfield=True)

        bx,by,bz = get_cart(br,bt,bp,th3D,p3D)

        bx = np.asfortranarray(bx)
        by = np.asfortranarray(by)
        bz = np.asfortranarray(bz)

        keys.append("vecB")
        values.append((bx,by,bz))
        btile = False

    if any(elem in ["ur", "vr"] for elem in scals):
        ur = tile_and_fix(mode.ur,mode.m,nr,ntheta,nphi,step,potextra)
        keys.append("Radial vel")
        values.append(ur)

    if any(elem in ["ut", "utheta", "vt", "vtheta"] for elem in scals):
        utheta = tile_and_fix(mode.utheta,mode.m,nr,ntheta,nphi,step,potextra)
        keys.append("U theta")
        values.append(utheta)

    if any(elem in ["up","uphi","vp","vphi"] for elem in scals):
        uphi = tile_and_fix(mode.uphi,mode.m,nr,ntheta,nphi,step,potextra)
        keys.append("Zonal flow")
        values.append(uphi)

    if any(elem in ["us","vs"] for elem in scals):
        us = np.zeros_like(mode.ur)
        for k,ktheta in enumerate(mode.theta):
            us[:,k,:] = ( mode.ur[:,k,:]*np.sin(ktheta)
                         +mode.utheta[:,k,:]*np.cos(ktheta) )
        us = tile_and_fix(us,mode.m,nr,ntheta,nphi,step,potextra)
        keys.append("Cyl rad vel")
        values.append(us)

    if any(elem in ["br"] for elem in scals):
        if btile:
            br = tile_and_fix(br,mode.m,nr,ntheta,nphi,step,potextra,bfield=True)

        keys.append("Radial mag. field")
        values.append(br)

    if any(elem in ["bt", "btheta"] for elem in scals):
        if btile:
            btheta = tile_and_fix(btheta,mode.m,nr,ntheta,nphi,step,potextra,bfield=True)

        keys.append("B_theta")
        values.append(btheta)

    if any(elem in ["bp","bphi"] for elem in scals):
        if btile:
            bphi = tile_and_fix(bphi,mode.m,nr,ntheta,nphi,step,potextra,bfield=True)

        keys.append("Zonal mag. field")
        values.append(bphi)

    if any(elem in ["t","temp","temperature"] for elem in scals):
        temperature = tile_and_fix(mode.temperature,mode.m,
                                    nr,ntheta,nphi,step,potextra)
        keys.append("Temperature")
        values.append(temperature)

    if any(elem in ["c","xi","comp","compositon","chem"] for elem in scals):
        composition= tile_and_fix(mode.composition,mode.m,
                                   nr,ntheta,nphi,step,potextra)
        composition = np.asfortranarray(composition)
        keys.append("Composition")
        values.append(composition)

    dataDict = dict(zip(keys,values))

    gridToVTK("out",x3D,y3D,z3D,pointData= dataDict)

    print("Output written to out.vts!")

    return 0

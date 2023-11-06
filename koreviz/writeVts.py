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

def get_grid(r,theta,phi,nr,ntheta,nphi):

    r3D  = np.zeros([nr,ntheta,nphi])
    th3D = np.zeros([nr,ntheta,nphi])
    p3D  = np.zeros([nr,ntheta,nphi])

    for i in range(nr):
        r3D[i,...] = r[i]
    for j in range(ntheta):
        th3D[:,j,:] = theta[j]
    for k in range(nphi):
        p3D[...,k] = phi[k]

    s3D = r3D * np.sin(th3D)
    x3D = s3D * np.cos(p3D)
    y3D = s3D * np.sin(p3D)
    z3D = r3D * np.cos(th3D)

    return r3D,th3D,p3D, x3D,y3D,z3D, s3D

def get_cart(vr,vt,vp,r3D,th3D,p3D):

    vs = vr * np.sin(th3D) + vt *np.cos(th3D)
    vz = vr * np.cos(th3D) - vt *np.sin(th3D)

    vx = vs * np.cos(p3D) - vp * np.sin(p3D)
    vy = vs * np.sin(p3D) + vp * np.cos(p3D)

    return vx,vy,vz

def tile_and_fix(data,m,nr,ntheta,nphi,step):

    scal = np.zeros([nr,ntheta,nphi])
    scal_tile = (np.tile(data,m))[::step,::step,:]
    scal[...,:-1] = scal_tile
    scal[...,-1]  = scal_tile[...,0]
    scal = np.asfortranarray(scal)

    del scal_tile

    return scal

def writeVts(mode, scals=[],vecs=[],step=5):

    r     = mode.r[::step]
    theta = mode.theta[::step]

    nr     = len(r)
    ntheta = len(theta)
    nphi   = mode.phi.shape[0]

    r3D,th3D,p3D, x3D,y3D,z3D, s3D = get_grid(r,theta,mode.phi,nr,ntheta,nphi)

    keys = []
    values = []

    keys.append("radius")
    keys.append("cyl_radius")

    values.append(r3D)
    values.append(s3D)

    # Make everything case insensitive

    for k in range(len(scals)):
        scals[k] = scals[k].lower()

    for k in range(len(vecs)):
        vecs[k] = vecs[k].lower()

    if any(elem in ["u","v"] for elem in vecs):

        # Tiling and steps

        ur = tile_and_fix(mode.ur,mode.m,nr,ntheta,nphi,step)
        ut = tile_and_fix(mode.utheta,mode.m,nr,ntheta,nphi,step)
        up = tile_and_fix(mode.uphi,mode.m,nr,ntheta,nphi,step)

        ux,uy,uz = get_cart(ur,ut,up,r3D,th3D,p3D)

        ux = np.asfortranarray(ux)
        uy = np.asfortranarray(uy)
        uz = np.asfortranarray(uz)

        keys.append("vecV")
        values.append((ux,uy,uz))

    if any(elem in ["b"] for elem in vecs):

        # Tiling and steps

        br = tile_and_fix(mode.br,mode.m,nr,ntheta,nphi,step)
        bt = tile_and_fix(mode.btheta,mode.m,nr,ntheta,nphi,step)
        bp = tile_and_fix(mode.bphi,mode.m,nr,ntheta,nphi,step)

        bx,by,bz = get_cart(br,bt,bp,r3D,th3D,p3D)

        bx = np.asfortranarray(bx)
        by = np.asfortranarray(by)
        bz = np.asfortranarray(bz)

        keys.append("vecB")
        values.append((bx,by,bz))

    if any(elem in ["ur", "vr"] for elem in scals):
        ur = tile_and_fix(mode.ur,mode.m,nr,ntheta,nphi,step)
        keys.append("Radial vel")
        values.append(ur)

    if any(elem in ["ut", "utheta", "vt", "vtheta"] for elem in scals):
        utheta = tile_and_fix(mode.utheta,mode.m,nr,ntheta,nphi,step)
        keys.append("U theta")
        values.append(utheta)

    if any(elem in ["up","uphi","vp","vphi"] for elem in scals):
        uphi = tile_and_fix(mode.uphi,mode.m,nr,ntheta,nphi,step)
        keys.append("Zonal flow")
        values.append(uphi)

    if any(elem in ["us","vs"] for elem in scals):
        us = np.zeros_like(mode.ur)
        for k,ktheta in enumerate(mode.theta):
            us[:,k,:] = ( mode.ur[:,k,:]*np.sin(ktheta)
                         +mode.utheta[:,k,:]*np.cos(ktheta) )
        us = tile_and_fix(us,mode.m,nr,ntheta,nphi,step)
        keys.append("Cyl rad vel")
        values.append(us)

    if any(elem in ["br"] for elem in scals):
        br = tile_and_fix(mode.br,mode.m,nr,ntheta,nphi,step)
        br = np.asfortranarray(mode.br)
        keys.append("Radial mag. field")
        values.append(br)

    if any(elem in ["bt", "btheta"] for elem in scals):
        btheta = tile_and_fix(mode.btheta,mode.m,nr,ntheta,nphi,step)
        btheta = np.asfortranarray(mode.btheta)
        keys.append("B_theta")
        values.append(btheta)

    if any(elem in ["bp","bphi"] for elem in scals):
        bphi = tile_and_fix(mode.bphi,mode.m,nr,ntheta,nphi,step)
        bphi = np.asfortranarray(mode.bphi)
        keys.append("Zonal mag. field")
        values.append(bphi)

    if any(elem in ["t","temp","temperature"] for elem in scals):
        temperature = tile_and_fix(mode.temperature,mode.m,
                                    nr,ntheta,nphi,step)
        keys.append("Temperature")
        values.append(temperature)

    if any(elem in ["c","xi","comp","compositon","chem"] for elem in scals):
        composition= tile_and_fix(mode.composition,mode.m,
                                   nr,ntheta,nphi,step)
        composition = np.asfortranarray(composition)
        keys.append("Composition")
        values.append(composition)

    dataDict = dict(zip(keys,values))

    gridToVTK("out",x3D,y3D,z3D,pointData= dataDict)

    print("Output written to out.vts!")

    return 0

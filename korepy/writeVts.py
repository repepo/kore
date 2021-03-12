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
    print("movie2vtk requires the use of evtk library!")
    print("You can get it from https://github.com/paulo-herrera/PyEVTK")

def get_grid(r,theta,phi,nr,ntheta,nphi):

    r3D  = np.zeros([nphi,ntheta,nr])
    th3D = np.zeros([nphi,ntheta,nr])
    p3D  = np.zeros([nphi,ntheta,nr])

    for i in range(nr):
        r3D[...,i] = r[i]
    for j in range(ntheta):
        th3D[:,j,:] = theta[j]
    for k in range(nphi):
        p3D[k,...] = phi[k]

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

def writeVts(mode, scals=[],vecs=[]):

    r3D,th3D,p3D, x3D,y3D,z3D, s3D = get_grid(mode.r,mode.theta,mode.phi,mode.nr,mode.ntheta,mode.nphi)

    keys = []
    values = []

    keys.append("radius")
    keys.append("cyl_radius")

    values.append(r3D)
    values.append(s3D)

    if any(elem in ["u", "U", "v", "V"] for elem in vecs):
        ux,uy,uz = get_cart(mode.ur, mode.utheta, mode.uphi,r3D,th3D,p3D)

        ux = np.asfortranarray(ux)
        uy = np.asfortranarray(uy)
        uz = np.asfortranarray(uz)

        keys.append("vecV")
        values.append((ux,uy,uz))

    if any(elem in ["b","B"] for elem in vecs):
        bx,by,bz = get_cart(mode.br, mode.btheta, mode.bphi,r3D,th3D,p3D)

        bx = np.asfortranarray(bx)
        by = np.asfortranarray(by)
        bz = np.asfortranarray(bz)

        keys.append("vecB")
        values.append((bx,by,bz))

    if any(elem in ["ur", "vr"] for elem in scals):
        ur = np.asfortranarray(mode.ur)
        keys.append("Radial vel")
        values.append(ur)

    if any(elem in ["ut", "utheta", "vt", "vtheta"] for elem in scals):
        utheta = np.asfortranarray(mode.utheta)
        keys.append("U_theta")
        values.append(utheta)

    if any(elem in ["up","uphi","vp","vphi"] for elem in scals):
        uphi = np.asfortranarray(mode.uphi)
        keys.append("Zonal flow")
        values.append(uphi)

    if any(elem in ["br", "Br"] for elem in scals):
        br = np.asfortranarray(mode.br)
        keys.append("Radial mag. field")
        values.append(br)

    if any(elem in ["bt", "btheta", "Bt", "Btheta"] for elem in scals):
        btheta = np.asfortranarray(mode.btheta)
        keys.append("B_theta")
        values.append(btheta)

    if any(elem in ["bp","bphi","Bp","Bphi"] for elem in scals):
        bphi = np.asfortranarray(mode.bphi)
        keys.append("Zonal mag. field")
        values.append(bphi)

    if any(elem in ["T","temp"] for elem in scals):
        temp = np.asfortranarray(mode.temp)
        keys.append("Temperature")
        values.append(temp)

    if any(elem in ["C","chem"] for elem in scals):
        chem = np.asfortranarray(mode.chem)
        keys.append("Composition")
        values.append(chem)

    dataDict = dict(zip(keys,values))

    gridToVTK("out",x3D,y3D,z3D,pointData= dataDict)

    print("Output written to out.vts!")

    return 0

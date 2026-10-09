#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

import numpy as np

try:
    from pyevtk.hl import gridToVTK
except ImportError:
    print("If you need 3D visualization:")
    print("writeVts requires the use of pyevtk library.")
    print("You can install it with pip: pip install pyevtk")

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

def tile_and_fix(data,m,nr,ntheta,nphi,step,ext=None):
    '''
    Puts a field of the mode (radius, theta, one azimuthal period) on the output grid: every step-th radius
    and colatitude, tiled over the m periods and closed in longitude (last = first). The output grid has the
    radii of the mode (in its order, i.e. decreasing) at the end; ext, if given, is put in the rows before it
    (e.g. the field outside the CMB, already at the output colatitudes, also in decreasing radius), otherwise
    those rows are zero.
    '''

    scal = np.zeros([nr,ntheta,nphi])
    rows = np.tile(data[::step,::step,:], max(m,1))
    scal[nr-rows.shape[0]:,:,:-1] = rows
    if ext is not None:
        scal[:ext.shape[0],:,:-1] = np.tile(ext, max(m,1))
    scal[...,-1] = scal[...,0]

    return np.asfortranarray(scal)

def writeVts(mode, scals=[],vecs=[],potextra=False,
             nrout=32,radratio=2.0,step=5):
    '''
    Writes the fields of a kmode to out.vts (pyevtk). With potextra=True and a magnetic field, the potential
    field outside the CMB is added on nrout radii up to radratio*rcmb (other fields are zero there).
    '''

    # Make everything case insensitive

    scals = [elem.lower() for elem in scals]
    vecs  = [elem.lower() for elem in vecs]

    # Figure out if magnetic field needs plotting

    plotb = ( any(elem in ['br','bphi','bp','bt','btheta'] for elem in scals) or
              any(elem in ["b"] for elem in vecs) )

    # the extrapolation only applies with a magnetic field
    potextra = potextra and plotb

    # Radii: those of the mode (decreasing, from rcmb), with step, and with potextra the exterior ones before
    # them, also decreasing, from radratio*rcmb down to just above rcmb (rcmb itself is the mode's first radius)

    r = mode.r[::step]
    ext = {}
    if potextra:
        rext = np.linspace(radratio*mode.rcmb, mode.rcmb, nrout)[:-1]
        brout, btout, bpout = mode.potextra(rext)
        ext = {'br': brout[:,::step,:], 'btheta': btout[:,::step,:], 'bphi': bpout[:,::step,:]}
        r = np.concatenate((rext, r))
    nr = len(r)

    theta = mode.theta[::step]
    ntheta = len(theta)
    nphi   = mode.phi.shape[0]

    grid = lambda data, name=None: tile_and_fix(data, mode.m, nr, ntheta, nphi, step, ext.get(name))

    r3D,th3D,p3D, x3D,y3D,z3D, s3D = get_grid(r,theta,mode.phi)

    keys = []
    values = []

    keys.append("radius")
    keys.append("cyl_radius")

    values.append(r3D)
    values.append(s3D)

    if plotb:
        br, btheta, bphi = grid(mode.br, 'br'), grid(mode.btheta, 'btheta'), grid(mode.bphi, 'bphi')

    if any(elem in ["u","v"] for elem in vecs):

        ux,uy,uz = get_cart(grid(mode.ur),grid(mode.utheta),grid(mode.uphi),th3D,p3D)

        keys.append("vecV")
        values.append((np.asfortranarray(ux),np.asfortranarray(uy),np.asfortranarray(uz)))

    if any(elem in ["b"] for elem in vecs):

        bx,by,bz = get_cart(br,btheta,bphi,th3D,p3D)

        keys.append("vecB")
        values.append((np.asfortranarray(bx),np.asfortranarray(by),np.asfortranarray(bz)))

    if any(elem in ["ur", "vr"] for elem in scals):
        keys.append("Radial vel")
        values.append(grid(mode.ur))

    if any(elem in ["ut", "utheta", "vt", "vtheta"] for elem in scals):
        keys.append("U theta")
        values.append(grid(mode.utheta))

    if any(elem in ["up","uphi","vp","vphi"] for elem in scals):
        keys.append("Zonal flow")
        values.append(grid(mode.uphi))

    if any(elem in ["us","vs"] for elem in scals):
        sint = np.sin(mode.theta)[None,:,None]
        cost = np.cos(mode.theta)[None,:,None]
        keys.append("Cyl rad vel")
        values.append(grid(mode.ur*sint + mode.utheta*cost))

    if any(elem in ["br"] for elem in scals):
        keys.append("Radial mag. field")
        values.append(br)

    if any(elem in ["bt", "btheta"] for elem in scals):
        keys.append("B_theta")
        values.append(btheta)

    if any(elem in ["bp","bphi"] for elem in scals):
        keys.append("Zonal mag. field")
        values.append(bphi)

    if any(elem in ["t","temp","temperature"] for elem in scals):
        keys.append("Temperature")
        values.append(grid(mode.temperature))

    if any(elem in ["c","xi","comp","compositon","chem"] for elem in scals):
        keys.append("Composition")
        values.append(grid(mode.composition))

    dataDict = dict(zip(keys,values))

    gridToVTK("out",x3D,y3D,z3D,pointData= dataDict)

    print("Output written to out.vts!")

    return 0

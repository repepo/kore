#!/usr/bin/env python3

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import scipy.io as sio
import scipy.sparse as ss

import utils as u


def main():
    

    Print = PETSc.Sys.Print
    rank = PETSc.COMM_WORLD.getRank()
    size = PETSc.COMM_WORLD.getSize()
    opts = PETSc.Options()

    input_matrix  = sys.argv[1]
    output_matrix = sys.argv[2]

    # read and prepare matrix A
    #A = sio.mmread('A_ns_ch.mtx')
    #A = sio.mmread('A1.mtx')
    #A = ss.csr_matrix(A)
    A = u.load_csr(input_matrix)
    nb_l,nb_c = A.shape



    MA = PETSc.Mat()
    MA.create(PETSc.COMM_WORLD)
    MA.setSizes([nb_l,nb_l])
    MA.setType('mpiaij')
    MA.setFromOptions()
    MA.setUp()

    Istart,Iend = MA.getOwnershipRange()
    indptrA = A[Istart:Iend,:].indptr
    indicesA = A[Istart:Iend,:].indices
    dataA = A[Istart:Iend,:].data
    del A

    MA.setPreallocationCSR(csr=(indptrA,indicesA))
    MA.setValuesCSR(indptrA,indicesA,dataA)
    MA.assemblyBegin()
    MA.assemblyEnd()
    del indptrA,indicesA,dataA
    print('Done reading and preparing input matrix')



    viewer = PETSc.Viewer().createBinary(output_matrix, 'w',PETSc.COMM_WORLD)
    viewer(MA)

    MA.destroy()
    viewer.destroy()


    PETSc.COMM_WORLD.Barrier()
    print('Done writing binary matrix')

if __name__ == '__main__':
    main()

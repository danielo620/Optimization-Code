import numpy as np
cimport numpy as np
cimport cython
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def NewStiff(int NDOF,double[:, :,::1] Global_Matrix,int NM,int[:, ::1]  MemberCOORDNum,int[::1] row,int[::1] col,double[::1] data):
    cdef int count = 0
    cdef int z
    cdef int x
    cdef int y
    for z in range(NM):
        for x in range(12):
            for y in range(12):
                if Global_Matrix[z, x, y] != 0 and MemberCOORDNum[z, x] <= NDOF and MemberCOORDNum[z, y] <= NDOF:
                    row[count] = MemberCOORDNum[z, x]
                    col[count] = MemberCOORDNum[z, y]
                    data[count] = Global_Matrix[z, x, y]
                    count = count + 1
    return row, col, data

import numpy as np
cimport numpy as np
cimport cython


# noinspection PyUnreachableCode
@cython.boundscheck(False)
@cython.wraparound(False)
def SupportCy(int[:, ::1] COORDNum,int[:, ::1] goat,int AA,int BB,int Count):
    cdef int x
    cdef int y
    for x in range(AA):
        for y in range(0, 6):
            if goat[x, y + 1] == 1:
                COORDNum[goat[x, 0], y] = Count
                Count = Count + 1
    Count = 0
    for x in range(BB):
        for y in range(0, 6):
            if COORDNum[x, y] == 0:
                COORDNum[x, y] = Count
                Count = Count + 1
    return COORDNum
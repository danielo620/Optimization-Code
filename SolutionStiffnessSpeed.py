import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from Matrix import localmatrix
import StiffnessNewCyth


def DisplacementCy2(NM, Modules, TransT, Transmatrix, NDOF, MemberCOORDNum, L, Pf,
                    MemberProp, Local_Matrix, COORDNum, x, AgrD, DNumber, Agr1, Agr2):

    # Local Stiffness Matrix
    Local_Matrix = localmatrix(Local_Matrix, MemberProp[:, 0], MemberProp[:, 1], MemberProp[:, 2],
                                MemberProp[:, 3], Modules[:, 0], Modules[:, 1], L)

    # Global Stiffness Matrix
    Global_Matrix = np.einsum('ijk,ikl ->ijl', TransT, Local_Matrix)
    Global_Matrix = np.einsum('ijk,ikl ->ijl', Global_Matrix, Transmatrix)

    # Stiffness Matrix
    Number = NM * 12 * 12  # length of Vector
    row = np.zeros(Number, dtype=np.intc)  # create array to store row locations
    col = np.zeros(Number, dtype=np.intc)  # create array to store column locations
    data = np.zeros(Number)  # create array to store data locations
    StiffnessNewCyth.NewStiff(NDOF - 1, Global_Matrix, NM, MemberCOORDNum, row, col, data)

    # Solve for joint Displacements
    ST = csr_matrix((data, (row, col)), shape=(NDOF, NDOF))  # Turn Row,Col, Data into a sparse matrix
    displacement = spsolve(ST, Pf)  # calculate displacement

    # Find node displacement for the L1 and L2
    Agr1[0] = displacement[COORDNum[DNumber[0], 1]]
    Agr2[0] = displacement[COORDNum[DNumber[1], 1]]
    AgrD[x] = displacement[COORDNum[DNumber[0], 1]] + displacement[COORDNum[DNumber[1], 1]]
    return displacement, Agr1, Agr2

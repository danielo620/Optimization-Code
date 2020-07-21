import numpy as np


# create transformation matrix for each member
def transformation(Matrix, Xe, Xb, Ye, Yb, Ze, Zb, L, x):
    MatrixR = np.zeros((3, 3))
    # if member is only Vertical in the Y-axis
    if np.abs(Ye - Yb) == L:
        RXy = (Ye - Yb) / L
        MatrixR[0, :] = [0, RXy, 0]
        MatrixR[1, :] = [-RXy, 0, 0]
        MatrixR[2, :] = [0, 0, 1]
        Matrix[x, :3, :3] = MatrixR
        Matrix[x, 3:6, 3:6] = MatrixR
        Matrix[x, 6:9, 6:9] = MatrixR
        Matrix[x, 9:, 9:] = MatrixR
        return Matrix
    # If member is not vertical only
    else:
        RXx = (Xe - Xb) / L
        RXy = (Ye - Yb) / L
        RXz = (Ze - Zb) / L
        Rsq = np.sqrt(RXx ** 2 + RXz ** 2)
        MatrixR[0, :] = [RXx, RXy, RXz]
        MatrixR[1, :] = [-RXx * RXy / Rsq, Rsq, -RXy * RXz / Rsq]
        MatrixR[2, :] = [-RXz / Rsq, 0, RXx / Rsq]
        Matrix[x, :3, :3] = MatrixR
        Matrix[x, 3:6, 3:6] = MatrixR
        Matrix[x, 6:9, 6:9] = MatrixR
        Matrix[x, 9:, 9:] = MatrixR
        return Matrix


# create the fix end vector
def fixendforces(L, w, Pf, Location):
    Pf[Location[:, 1]] = -w * L / 2 + Pf[Location[:, 1]]
    Pf[Location[:, 5]] = -w * (L ** 2) / 12 + Pf[Location[:, 5]]
    Pf[Location[:, 7]] = -w * L / 2 + Pf[Location[:, 7]]
    Pf[Location[:, 11]] = w * (L ** 2) / 12 + Pf[Location[:, 11]]
    return Pf
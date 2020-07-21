import numpy as np


def transformation(Matrix, Xe, Xb, Ye, Yb, Ze, Zb, L, NM):
    for x in range(NM):
        MatrixR = np.zeros((3, 3))
        if np.abs(Ye[x] - Yb[x]) == L[x]:
            RXy = (Ye[x] - Yb[x]) / L[x]
            MatrixR[0, :] = [0, RXy, 0]
            MatrixR[1, :] = [-RXy, 0, 0]
            MatrixR[2, :] = [0, 0, 1]
            Matrix[x, :3, :3] = MatrixR
            Matrix[x, 3:6, 3:6] = MatrixR
            Matrix[x, 6:9, 6:9] = MatrixR
            Matrix[x, 9:, 9:] = MatrixR
        else:
            RXx = (Xe[x] - Xb[x]) / L[x]
            RXy = (Ye[x] - Yb[x]) / L[x]
            RXz = (Ze[x] - Zb[x]) / L[x]
            Rsq = np.sqrt(RXx ** 2 + RXz ** 2)
            MatrixR[0, :] = [RXx, RXy, RXz]
            MatrixR[1, :] = [-RXx * RXy / Rsq, Rsq, -RXy * RXz / Rsq]
            MatrixR[2, :] = [-RXz / Rsq, 0, RXx / Rsq]
            Matrix[x, :3, :3] = MatrixR
            Matrix[x, 3:6, 3:6] = MatrixR
            Matrix[x, 6:9, 6:9] = MatrixR
            Matrix[x, 9:, 9:] = MatrixR
    return Matrix

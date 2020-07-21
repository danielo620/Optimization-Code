# Create the local stiffness matrix for each member


def localmatrix(matrix, A, Iz, Iy, J, E, G, L):
    EE = E / (L ** 3)
    k1 = A * (L ** 2) * EE
    k2 = 12 * Iz * EE
    k3 = 6 * L * Iz * EE
    k4 = 12 * Iy * EE
    k5 = 6 * L * Iy * EE
    k6 = G * J * (L ** 2) / E * EE
    k7 = 4 * (L ** 2) * Iy * EE
    k8 = 2 * (L ** 2) * Iy * EE
    k9 = 4 * (L ** 2) * Iz * EE
    k10 = 2 * (L ** 2) * Iz * EE
    k1 = k1[:, None]
    k2 = k2[:, None]
    k3 = k3[:, None]
    k4 = k4[:, None]
    k5 = k5[:, None]
    k6 = k6[:, None]
    k7 = k7[:, None]
    k8 = k8[:, None]
    k9 = k9[:, None]
    k10 = k10[:, None]
    matrix[:, [0, 6], [0, 6]] = k1
    matrix[:, [0, 6], [6, 0]] = -k1
    matrix[:, [1, 7], [1, 7]] = k2
    matrix[:, [1, 7], [7, 1]] = -k2
    matrix[:, [1, 1, 5, 11], [5, 11, 1, 1]] = k3
    matrix[:, [5, 7, 7, 11], [7, 5, 11, 7]] = -k3
    matrix[:, [2, 8], [2, 8]] = k4
    matrix[:, [2, 8], [8, 2]] = -k4
    matrix[:, [4, 8, 8, 10], [8, 4, 10, 8]] = k5
    matrix[:, [2, 2, 4, 10], [4, 10, 2, 2]] = -k5
    matrix[:, [3, 9], [3, 9]] = k6
    matrix[:, [3, 9], [9, 3]] = -k6
    matrix[:, [4, 10], [4, 10]] = k7
    matrix[:, [4, 10], [10, 4]] = k8
    matrix[:, [5, 11], [5, 11]] = k9
    matrix[:, [5, 11], [11, 5]] = k10
    return matrix

import numpy as np


# Cross parents to create new bridge
def Uniform(P1, P2, CGroup, z, chance, cr, best):
    k = np.size(P1)
    j = np.random.randint(2, size=k)  # Create a random vector with 0's and 1's ([Ex: [0, 0, 1, 1, 0, 1])
    if z == 0:
        CGroup[z * 2, :] = best  # keep best solution for next generation
    else:
        if chance[2 * z] <= cr:
            CGroup[z * 2, :] = P1 * j + (1 - j) * P2  # where j = 1 parent 1's gene where j = 0  parents 2's genes
        else:
            CGroup[z * 2, :] = P1

    if chance[2 * z + 1] <= cr:
        CGroup[z * 2 + 1, :] = P2 * j + (1 - j) * P1 # where j = 1 parent 2's gene where j = 0  parents 1's genes
    else:
        CGroup[z * 2 + 1, :] = P2
    return CGroup

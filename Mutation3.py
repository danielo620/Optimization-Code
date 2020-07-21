import numpy as np


def mutant(C1, C2, CGroup, mu, z, sigma):
    Loc1 = np.where(np.random.random(np.size(CGroup[z])) < mu)  # Choose random group to mutate from child one
    Loc2 = np.where(np.random.random(np.size(CGroup[z])) < mu)  # Choose random group to mutate from child two
    R1 = np.random.randn(np.size(Loc1))  # return randomly standard normal number for child one
    R2 = np.random.randn(np.size(Loc2))  # return randomly standard normal number for child two
    # update gene[Loc1] on number of available cross-sections(Gaussian Distribution)
    CGroup[z * 2, Loc1] = C1[Loc1] + sigma[Loc1] * R1
    # update gene[Loc2] on number of available cross-sections(Gaussian Distribution)
    CGroup[z * 2 + 1, Loc2] = C2[Loc2] + sigma[Loc2] * R2
    return CGroup

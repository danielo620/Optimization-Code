import numpy as np
# Choose the parents

def Wheel(P):
    r = np.random.random() * np.sum(P)  # create a random number
    C = np.cumsum(P)  # create an array that adds each element (Ex: [1, 3, 5 ,10 ,15] = [1, 4, 9, 19, 34])
    A = np.argmax(r <= C)  # find the first occurrence where random number is greater than C
    return A

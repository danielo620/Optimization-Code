# Create an array that contains the joint number corresponding to the end and start of member
# Create an array that correlates the coordinates to the joint number
import numpy as np


def Coordinate(COORD, AA, BB, CC, Mbe, StartCord, EndCord):
    J = 0
    for x in range(0, np.size(CC)):
        Start = np.where(CC[x] == AA)
        End = np.where(CC[x] == BB)
        Mbe[Start, 0] = J
        Mbe[End, 1] = J
        if Start[0].size > 0:
            COORD[x, :] = StartCord[np.amin(Start), :]
        else:
            COORD[x, :] = EndCord[np.amin(End), :]
        J += 1
    return COORD

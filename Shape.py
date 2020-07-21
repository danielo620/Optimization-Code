import numpy as np


# calculate the properties of each cross-section
def availableshapes(Available_Shape, Shape_Dimension, Section_Prop, Count):
    for x in range(Count):
        if Available_Shape[x] == 'Round Tubing':
            D = Shape_Dimension[x, 0]
            t = Shape_Dimension[x, 1]
            Section_Prop[x, 0] = np.pi * ((D / 2) ** 2 - (D / 2 - t) ** 2)  # Area
            Section_Prop[x, 1:3] = np.pi / 4 * ((D / 2) ** 4 - (D / 2 - t) ** 4)  # moment of Inertia
            Section_Prop[x, 3] = 2 * Section_Prop[x, 1]  # Torsion Constant
        elif Available_Shape[x] == 'Rectangular Tubing':
            b = Shape_Dimension[x, 2]
            h = Shape_Dimension[x, 0]
            t = Shape_Dimension[x, 1]
            bi = (b - 2 * t)
            hi = (h - 2 * t)
            Section_Prop[x, 0] = b * h - bi * hi  # Area
            Section_Prop[x, 1] = b * (h ** 3) / 12 - bi * (hi ** 3) / 12  # moment of Inertia Iz
            Section_Prop[x, 2] = h * (b ** 3) / 12 - hi * (bi ** 3) / 12  # moment of Inertia Iy
            # Torsion Constant
            Section_Prop[x, 3] = ((h * b ** 3) * (1/3 - .21 * b/h * (1 - ((b ** 4)/(12 * h ** 4)))) - (hi * bi ** 3) * (1/3 - .21 * bi/hi * (1 - ((bi ** 4)/(12 * hi ** 4))))) * .888
        elif Available_Shape[x] == 'Rigid Link':
            Section_Prop[x, 0] = 10
            Section_Prop[x, 1] = 10
            Section_Prop[x, 2] = 10
            Section_Prop[x, 3] = 10
    return Section_Prop

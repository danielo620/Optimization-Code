# Cost of weight and deflection of each bridge
def BridgeCost(SlopD, Weight, AgrD, NP, Cost):
    for x in range(NP):
        if Weight[x] <= 175:  # if weight is less than or greater than 175
            Cost[x] = SlopD * 3150000 * AgrD[x]
        elif Weight[x] > 300:  # if weight is less than or greater than 300
            Cost[x] = 16000 * (Weight[x] - 237.5) + SlopD * 3150000 * AgrD[x]
        else:  # if weight is between
            Cost[x] = 8000 * (Weight[x] - 175) + SlopD * 3150000 * AgrD[x]
    return Cost

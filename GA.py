import numpy as np
import Section
from joblib import Parallel, delayed
import SolutionStiffnessSpeed
import CostCalculator
import Mutation3
import UniformCrossover
import Roullette
import os
from timeit import default_timer as timer
import xlsxwriter
import pandas as pd

path = 'C:/Users/rocky/Desktop/Optimization/NPZ'  # path of location of data.npz folder
pathOPT = 'C:/Users/rocky/Desktop/Optimization/OPT'  # path to where to place information for further analyses
pathOP = 'C:/Users/rocky/Desktop/Optimization/Op/'  # path to where to place excel files with optimize cross-sections

# Load Case
LC = 3

# Genetic Algorithm parameter
NP = 50  # number of particles
NPhalf = 25
Itt = 100  # number of Iterations
PC = 1  # Ratio of children made per generation
mu = 0.035  # probability of mutating
cr = .6  # probability of crossing

# Cost function Slop
SlopD = 1.9
SlopW = 13
'''
















'''
# start Parallel Pool
with Parallel(n_jobs=12, prefer="threads") as Parallel:
    with os.scandir(path=path) as entries:
        for entry in entries:

            # Extract Data from file
            File = entry.name
            npzfile = np.load((os.path.join(path, File)))
            Shape_Dimension = npzfile['Shape_Dimension']
            AgrJN = npzfile['AgrJN']
            Transmatrix = npzfile['Transmatrix']
            TransT = npzfile['TransT']
            Section_Prop = npzfile['Section_Prop']
            P = npzfile['P']
            L = npzfile['L']
            MemberCOORDNum = npzfile['MemberCOORDNum']
            G = np.min(MemberCOORDNum)
            Shape_Set = npzfile['Shape_Set']
            Group = npzfile['Group']
            NM = npzfile['NM']
            NR = npzfile['NR']
            NDOF = npzfile['NDOF']
            COORDNum = npzfile['COORDNum']
            Modules = npzfile['Modules']
            Wt = npzfile['Wt']

            # Choose from desire Load Case
            if LC == 1:
                Pf = P[:, 0]
                DNumber = AgrJN[:, 0]
            elif LC == 2:
                Pf = P[:, 1]
                DNumber = AgrJN[:, 1]
            elif LC == 3:
                Pf = P[:, 2]
                DNumber = AgrJN[:, 2]
            elif LC == 4:
                Pf = P[:, 3]
                DNumber = AgrJN[:, 3]
            elif LC == 5:
                Pf = P[:, 4]
                DNumber = AgrJN[:, 4]
            else:
                Pf = P[:, 5]
                DNumber = AgrJN[:, 5]

            # Dynamic Exploration Parameters
            nvarmin = Shape_Set[:, 0]  # number of first available cross-section of each Group
            nvarmax = Shape_Set[:, 1]  # number of last available cross-section of each Group
            sigma = (Shape_Set[:, 1] - Shape_Set[:, 0] + 1) / 2  # starting Gaussian variance for each Group
            dynamicSig = (sigma - 1) / 100  # how the variance changes over time for each Group

            # Blanks for Optimization
            size = np.shape(Shape_Set[:, 0])[0]
            MemberProp = np.zeros((NP, NM, 4))
            GroupShape = np.zeros((NP, size), dtype=np.intc)
            Local_Matrix = np.zeros((NP, NM, 12, 12))
            AgrD = np.zeros(NP)
            AgrDC = np.zeros(NP)
            weight = np.zeros(NP)
            Cost = np.zeros(NP)
            CostC = np.zeros(NP)
            Agr1 = np.zeros(1)
            Agr2 = np.zeros(1)

            # Create Random finesses population
            Section.memgroupsec(NP, GroupShape, Shape_Set)
            # shape properties for each the random cross-sections
            MemberProp[:, :] = Section_Prop[GroupShape[:, Group], :]

            # start timer
            start = timer()

            # Run fitness function for starting population
            Parallel(
                delayed(SolutionStiffnessSpeed.DisplacementCy2)(NM, Modules, TransT, Transmatrix, NDOF, MemberCOORDNum,
                                                                L, Pf, MemberProp[x], Local_Matrix[x], COORDNum,
                                                                x, AgrD, DNumber, Agr1, Agr2)
                for x in range(NP))

            # evaluate starting population
            weight[:] = np.sum(Wt[GroupShape[:, Group[:]]] * L, axis=1) / 12 + SlopW  # weight function
            CostCalculator.BridgeCost(SlopD, weight, AgrD, NP, Cost)  # Cost Function
            A = np.argmin(Cost)  # index of fittest individual
            BestP = Cost[A]  # Cost of fittest individual
            W = weight[A]  # weight of fittest individual
            Deflection = AgrD[A]  # Deflection of fittest individual
            setT = GroupShape[A]  # cross section of of each group for fittest individual

            for y in range(Itt):
                J = np.arange(1, NP + 1)  # create an array from 1 to number of particles
                J = np.flip(J)  # flip array(number of particles to 1)
                J = J ** 5  # increase the difference between each value
                Jsum = np.abs(np.sum(J))
                PP = J / Jsum
                CGroup = np.zeros(GroupShape.shape, dtype=np.intc)
                chance = np.random.random(NP)  # random numbers between 0 and 1

                # Elitism (Keep the best individual of the population for the next generation)
                Elite = GroupShape[0, :]
                EliteCost = Cost[0]

                # Parent Choosing and Mutation
                for z in range(NPhalf):
                    # select parents
                    P1 = Roullette.Wheel(PP)
                    P2 = Roullette.Wheel(PP)
                    # Crossover (Create children)
                    UniformCrossover.Uniform(GroupShape[P1], GroupShape[P2], CGroup, z, chance, cr, GroupShape[0])
                    # Mutation
                    Mutation3.mutant(CGroup[2 * z], CGroup[2 * z + 1], CGroup, mu, z, sigma)

                # constrain offsprings
                CGroup[:] = np.where(CGroup > Shape_Set[:, 0], CGroup, Shape_Set[:, 0])
                CGroup[:] = np.where(CGroup < Shape_Set[:, 1], CGroup, Shape_Set[:, 1])

                # evaluate children fitness
                MemberProp[:, :] = Section_Prop[CGroup[:, Group], :]
                Parallel(
                    delayed(SolutionStiffnessSpeed.DisplacementCy2)(NM, Modules, TransT, Transmatrix, NDOF,
                                                                    MemberCOORDNum, L, Pf, MemberProp[x],
                                                                    Local_Matrix[x], COORDNum, x, AgrDC, DNumber, Agr1, Agr2)
                    for x in range(NP))

                # evaluate cost of each children
                weightC = np.zeros(NP)
                weightC[:] = np.sum(Wt[CGroup[:, Group[:]]] * (L / 12), axis=1) + SlopW

                # cost function
                CostCalculator.BridgeCost(SlopD, weightC, AgrDC, NP, CostC)
                A = np.argmin(CostC)   # index of fittest child
                BestC = CostC[A]   # Cost of fittest child

                # Update Population Best
                if BestC < BestP:
                    setT = CGroup[A]
                    BestP = BestC
                    W = weightC[A]
                    Deflection = AgrDC[A]

                # print current best
                print("Cost = ", BestP, "      AgrD = ", Deflection, "      Weight = ", W)

                # merge population
                Cost = np.hstack([Cost, CostC, EliteCost])  # Stack parents and children cost
                X = np.argsort(Cost)  # sort cost
                GroupShape = np.vstack([GroupShape, CGroup, Elite])  # stack Parent shape and Child shape
                GroupShape = GroupShape[X, :]  # rearrange Groupshape
                GroupShape = GroupShape[:NP, :]  # extract the best children for next generation
                Cost = Cost[X]  # rearrange cost
                Cost = Cost[:NP]  # extract cost of best children for next generation

                # dynamic mutation parameters
                mu = mu - .000005  # Reduce mutation rate
                sigma -= dynamicSig  # Reduce sigma

            # time taken to run each file
            end = timer()
            print(end - start)

            # parameters of the most fit child
            Result = Shape_Dimension[setT]
            Q = np.where(Result[:, 2] == 0)
            Result = Result.astype(np.object_)
            Result[Q, 2] = "NaN"

            # save results for further analysis
            np.savez((os.path.join(pathOPT, File[:-4] + 'OPT')), setT=setT, W=W, NDOF=NDOF, COORDNum=COORDNum, MemberCOORDNum=MemberCOORDNum, Section_Prop=Section_Prop, Group=Group, AgrJN=AgrJN, P=P, L=L, NM=NM, Modules=Modules, TransT=TransT, Transmatrix=Transmatrix)
            workbook = xlsxwriter.Workbook(pathOP + File[:-4] + '.xlsx')
            worksheet = workbook.add_worksheet()
            workbook.close()
            df = pd.DataFrame(Result)
            df.to_excel(pathOP + File[:-4] + '.xlsx', index=False)
            end = timer()
            print(end - start)



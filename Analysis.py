import numpy as np
import os
import SolutionStiffnessSpeed
import xlwings


path = 'C:/Users/rocky/Desktop/Optimization/OPT'
ExcelPath = 'C:/Users/rocky/Desktop/Optimization/Result/Result.xlsx'
''''



























'''
row = 6
col = 4
with os.scandir(path=path) as entries:
    for entry in entries:
        File = entry.name
        npzfile = np.load((os.path.join(path, File)))
        setT = npzfile['setT']
        W = npzfile['W']
        Section_Prop = npzfile['Section_Prop']
        Group = npzfile['Group']
        AgrJN = npzfile['AgrJN']
        P = npzfile['P']
        NM = npzfile['NM']
        Modules = npzfile['Modules']
        TransT = npzfile['TransT']
        Transmatrix = npzfile['Transmatrix']
        NDOF = npzfile['NDOF']
        MemberCOORDNum = npzfile['MemberCOORDNum']
        L = npzfile['L']
        COORDNum = npzfile['COORDNum']
        MemberProp = Section_Prop[setT[Group], :]
        Local_Matrix = np.zeros((6, NM, 12, 12))
        AgrDC = np.zeros(6)
        Agr1 = np.zeros(1)
        Agr2 = np.zeros(1)
        Agr = np.zeros((2, 6))
        for x in range(6):
            Pf = P[:, x]
            DNumber = AgrJN[:, x]
            SolutionStiffnessSpeed.DisplacementCy2(NM, Modules, TransT, Transmatrix, NDOF,
                                                   MemberCOORDNum, L, Pf, MemberProp,
                                                   Local_Matrix[x], COORDNum, x, AgrDC, DNumber, Agr1, Agr2)
            Agr[0, x] = Agr1
            Agr[1, x] = Agr2
        wb = xlwings.Book(ExcelPath)
        Sheet1 = wb.sheets[0]
        for y in range(6):
            Sheet1.range(row, 4 + y).value = Agr[0, y]
            Sheet1.range(row + 1, 4 + y).value = Agr[1, y]
        Sheet1.range(row - 3, 3).value = File[:-7]
        Sheet1.range(row + 3, 4).value = W
        row = row + 11
    wb.save()
    wb.close()


import numpy as np
import scipy.linalg as linear

def latexMatrix(M, matrixName, eq=True, complx=False, form='%0.0f'):
    s=''
    if eq:
        s ='\\[ \n' + matrixName + ' = \n'
    s += '\\begin{pmatrix} \n'
    [rows, cols] = M.shape
    if complx:
        form = form+form[0]+'+'+form[1:]
        for i in range(rows):
            for j in range(cols):
                s += (form % (M[i,j].real, A[i,j].imag)) + ' '
                
                if not j+1 == cols:
                    s += '& '           
            s += '\\\\ \n'
    else:
        for i in range(rows):
            for j in range(cols):
                s += (form % (M[i,j].real)) + ' '
                if not (j+1 == cols):
                    s += '& '
            s += '\\\\ \n'
    s += '\\end{pmatrix}'
    if eq:
        s += '\n \\]'
    return s

def latexVector(v, vecName, eq=True, complx=False, form='%0.0f'):
    s=''
    if eq:
        s ='\\[ \n \\vec{' + vecName + '} = \n'
    s += '\\begin{pmatrix} \n'
    if complx:
        form = form+form[0]+'+'+form[1:]
        for x in np.nditer(v):
            s += (form % (x.real,x.imag) + ' \\\\ \n')
    else:
        for x in np.nditer(v):
            s += (form % (x.real) + ' \\\\ \n')
    s += '\\end{pmatrix}'
    if eq:
        s += '\n \\]'
    return s
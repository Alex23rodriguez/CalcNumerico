import sys
sys.path.append('../')
from IPython.display import Latex
import latexStrings as ls
import numpy as np
import scipy.linalg as linear
import eigenvalues as ev


# Definimos la matriz A como sigue:
A = np.array([[1,-4,-6],[-12,-8,-6],[11,10,10]])

#Calculamos los valores exactos
[eigenvalues,eigenvectors]=linear.eig(A)
[eigenvalues,eigenvectors]=ev.pairSort(eigenvalues,eigenvectors)
print('Eigenvalues: \n' + str(eigenvalues))
print('Eigenvectors: \n' + str(eigenvectors))

#Calculamos Potencia Inversa
q = np.array([1,1,1])
[w,l,i] = ev.inversePower(A,q)
print('Potencia Inversa con q0  = (1,1,1)')
print("Numero de iteraciones: "+ str(i))
print("eigenvector = " + str(w))
print("eigenvalue = " + str(l))
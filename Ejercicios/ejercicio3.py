import sys
sys.path.append('../')
from IPython.display import Latex
import latexStrings as ls
import numpy as np
import scipy.linalg as linear
import eigenvalues as ev

#Ejercicio 3

#Tomando A y q_0 como los definimos anteriormente:
A = np.array([[1,1,2],[-1,9,3],[0,-1,3]])
q = np.array([1,1,1])

#Se calcularán 10 iteraciones del método de la potencia con shift \rho=3.6
[q10, l10 ,iterations]=ev.inversePowerShift(A,q,3.6,1e-15)
print("3.6 Shift IPM")
print("Iterations : " + str (iterations))
print("Eigenvalue = " + str(l10))
print("Eigenvector = " + str(q10))

#Eigenvalores exactos:
[L,V] = linear.eig(A)
print("A")
print("Eigenvalues = " + str(L))
print("Eigenvectors = " + str(V))

#Veamos todos los eigenvalores de la matriz con el shift \rho=3.6:
[L,V] = linear.eig(A-3.6*np.identity(len(A)))
print("A-3.6I")
print("Eigenvalues = " + str(L))
print("Eigenvectors = " + str(V))
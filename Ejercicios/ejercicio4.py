import sys
sys.path.append('../')
from IPython.display import Latex
import latexStrings as ls
import numpy as np
import scipy.linalg as linear
import eigenvalues as ev


# # Ejercicio 4

# Definimos la matriz $A$ como en el ejercicio 1:
A = np.array([[1,1,2],[-1,9,3],[0,-1,3]])


# Los eigenvalores y eigenvectores exactos a los que queremos llegar con nuestro método son los siguientes:
[eigenvalues,eigenvectors]=linear.eig(A)
[eigenvalues,eigenvectors]=ev.pairSort(eigenvalues,eigenvectors)
print('Eigenvalues: \n' + str(eigenvalues))
print('Eigenvectors: \n' + str(eigenvectors))

# Base Inicial
q1 = np.array([0,1,0])
q2 = np.array([0,0,1])
q3 = np.array([1,0,0])
Latex(ls.latexVector(q1,'q_0^1') + ls.latexVector(q2,'q_0^2')+ls.latexVector(q3,'q_0^3'))


#MPISR 1
[w1,l1,i1] = ev.inversePowerRayleigh(A,q1)
print('MPISR q1 = (0,1,0)')
print('w1 = ' + str(w1))
print('s1 = ' + str(l1))

#MPISR 2
[w2,l2,i2] = ev.inversePowerRayleigh(A,q2)
print('MPISR q2 = (0,0,1)')
print('w2 = ' + str(w2))
print('s2 = ' + str(l2))

#MPISR 3
[w3,l3,i3] = ev.inversePowerRayleigh(A,q3)
print('MPISR q3 = (1,0,0)')
print('w3 = ' + str(w3))
print('s3 = ' + str(l3))


#Aproximaciones q1
aprox=[]
for i in range(1,7):
    [_,sigmai,_] = ev.inversePowerRayleigh(A,q1,1e-14,i)
    aprox.append(sigmai)
print('Aproximaciones con q1: \n' + str(aprox))

#Aproximaciones q2
aprox=[]
for i in range(1,7):
    [_,sigmai,_] = ev.inversePowerRayleigh(A,q2,1e-15,i)
    aprox.append(sigmai)
print('Aproximaciones con q2: \n' + str(aprox))

#Aproximaciones q3
aprox=[]
for i in range(1,7):
    [_,sigmai,_] = ev.inversePowerRayleigh(A,q3,1e-14,i)
    aprox.append(sigmai)
print('Aproximaciones con q3: \n' + str(aprox))
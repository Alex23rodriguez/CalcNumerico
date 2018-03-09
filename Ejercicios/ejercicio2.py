import sys
sys.path.append('../')
from IPython.display import Latex
import latexStrings as ls
import numpy as np
import scipy.linalg as linear
import eigenvalues as ev


#Ejercicio 2.1

# Utilizando la matriz A y el vector q_0 del ejercicio anterior definidos como sigue:
A = np.array([[1,1,2],[-1,9,3],[0,-1,3]])
q = np.array([1,1,1])

# Se calcularán 10 iteraciones del método de la potencia con shift 0
[q10, l10 ,iterations]=ev.inversePowerShift(A,q,0,1e-6,10)
print("0 Shift IPM")
print("Iterations : " + str (iterations))
print("Eigenvalue = " + str(l10))
print("Eigenvector = " + str(q10))

#Eigenvalores exactos
[L,V] = linear.eig(A)
[L,V] = ev.pairSort(L,V)
print("A")
print("Eigenvalues = " + str(L))
print("Eigenvectors = " + str(V))

# Guardamos -v_3
v=V[:,2]
v=-v
print('v_3 = ' + str(v))

# Calculamos las razones de convergencia en cada paso de la iteracion
ratios=[]
prevq=q
for i in range(1,10):
    [currentq,_,_] = ev.inversePowerShift(A,q,0,1e-6,i)
    ratio = linear.norm(currentq-v,np.inf)/linear.norm(prevq-v,np.inf)
    ratios.append(ratio)
    prevq = currentq
print('Ratios: '+ str(ratios))
print('Ratio teorico:'+str(abs(L[2]/L[1])))

#Ejercicio 2.2
#Para la misma matriz A y vector q_0 defininidos anteriormente, se realizarán las mismas pruebas para un shift 3.3
[q10, l10 ,iterations]=ev.inversePowerShift(A,q,3.3,1e-6,10)
print("3.3 Shift IPM")
print("Iterations : " + str (iterations))
print("Eigenvalue = " + str(l10))
print("Eigenvector = " + str(q10))

#Calculamos las razones de convergencia en cada paso de la iteracion:
ratios=[]
v=V[:,1]
prevq=q
for i in range(1,11):
    [currentq,_,_] = ev.inversePowerShift(A,q,3.3,0,i)
    ratio = linear.norm(currentq-v,np.inf)/linear.norm(prevq-v,np.inf)
    ratios.append(ratio)
    prevq = currentq
Latex(ls.latexList(ratios,'\widetilde{r}', form='%f'))
print('Ratios 1 - 10: '+ str(ratios))

# Imprimimos razones de convergencia adicionales
for i in range(11,21):
    [currentq,_,_] = ev.inversePowerShift(A,q,3.3,0,i)
    ratio = linear.norm(currentq-v,np.inf)/linear.norm(prevq-v,np.inf)
    ratios.append(ratio)
    prevq = currentq
Latex(ls.latexList(ratios[9:20],'\widetilde{r}_{10-20}', form='%f'))
print('Ratios 10 - 20: '+ str(ratios[9:21]))
print('Ratio teorico:'+str(abs((L[1]-3.3)/(L[2]-3.3))))
import sys
sys.path.append('../')
from IPython.display import Latex
import latexStrings as ls
import numpy as np
import scipy.linalg as linear
import eigenvalues as ev


# # Ejercicio 1

# Tenemos la matriz A y el vector q_0 definidos como:
A = np.array([[1,1,2],[-1,9,3],[0,-1,3]])
q = np.array([1,1,1])

# Queremos calcular 10 iteraciones del metodo de la potencia. Esto nos da como resultado:
[q10, l10 ,iterations]=ev.powerMethod(A,q,1e-6,10)
print('Numero de Iteraciones: '+ str(iterations))
print('l_10 = '+str(l10))
print('q_10 = '+str(q10))

# Despues comparemos los resultados con los valores 'exactos' calculados por el paquete _scipy.linalg_:
[L,V] = linear.eig(A)
print('Eigenvalores: ' + str(L))
print('Eigenvectores: '+ str(V))


# Vemos que en efecto el metodo de la potencia calculo el eigenvector dominante de la matriz. Esto se debe a que se cumplen las condiciones del metodo, es decir:
# 1. A tiene un eigenvalor dominante (8.3545)
# 2. Nuestro vector inicial q_0 puede ser escrito como combinacion lineal de los eigenvectores de A con coeficientes todos distintos de 0

# Ahora, tomemos el eigenvector asociado al eigenvalos dominante, dado por:
v=V[:,0]
print('v = '+str(v))


# Y ahora queremos calcular las razones de convergencia en cada paso de la iteracion:
ratios=[]
prevq=q
for i in range(1,11):
    [currentq,_,_] = ev.powerMethod(A,q,1e-6,i)
    ratio = linear.norm(currentq-v,np.inf)/linear.norm(prevq-v,np.inf)
    ratios.append(ratio)
    prevq = currentq
print('Ratios: '+ str(ratios))
print('Ratio teorico:'+str(abs(L[2]/L[0])))
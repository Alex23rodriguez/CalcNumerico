import sys
sys.path.append('../')
from IPython.display import Latex
import latexStrings as ls
import numpy as np
import scipy.linalg as linear
from eigenvalues import *


# # Ejercicio 6
# Considerémos la matriz fiedler(25). Le aplicaremos el método de QR simple con un máximo de 2000 iteraciones, 
#y el de QR con shift dinámico con un máximo de 20 iteraciones. En ambos casos la tolerancia será de 1e-10. 
#Despues compararemos con los eigenvalores calculados por _scipy.linalg_
QR_eig, QR_iter = simpleQR(fiedler(25), maxIter=2000, tol=1e-10)
shQR_eig, shQR_iter = shiftQR(fiedler(25), maxIter=20, tol=1e-10)
actual = linear.eig(fiedler(25))[0].real
err_abs_QR = abs(QR_eig - actual)
err_abs_shQR = abs(shQR_eig - actual)
err_rel_QR = abs((QR_eig - actual) / actual)
err_rel_shQR = abs((shQR_eig - actual) / actual)

# Comparemos los errores absolutos y relativos:
# **Error absoluto:**
print('Error absoluto QR Simple: \n' + str(err_abs_QR))
print('Error absoluto QR Shift: \n' + str(err_abs_shQR))


# **Error relativo:**
print('Error relativo QR Simple: \n' + str(err_rel_QR))
print('Error relativo QR Shift: \n' + str(err_rel_shQR))


# Para ver de manera mas clara los errores, compararemos con los valores reales utilizando la norma 2 de los vectores de error:
print('Error absoluto:')
print('QR simple: ' + str(linear.norm(err_abs_QR)))
print('QR con shift dinámico: ' + str(linear.norm(err_abs_shQR)))


print('Error relativo')
print('QR simple: ' + str(linear.norm(err_rel_QR)))
print('QR con shift dinámico: ' + str(linear.norm(err_rel_shQR)))


# Iteraciones
print('Iteraciones QR simple: '+str(QR_iter))
print('Iteraciones QR shift dinamico: '+str(shQR_iter))

# Veamos que pasa si limitamos el método de QR simplemente a 20 iteraciones:
QR_eig, QR_iter = simpleQR(fiedler(25), maxIter=20, tol=1e-10)
err_abs_QR = abs(QR_eig - actual)
err_rel_QR = abs((QR_eig - actual) / actual)

print('Error absoluto QR Simple (20 its):' + str(linear.norm(err_abs_QR)))
print('Error relativo QR Simple (20 its):' + str(linear.norm(err_rel_QR)))

# Si no limitamos el número de iteraciones, observamos que:
QR_eig, QR_iter = simpleQR(fiedler(25), maxIter=10000,  tol=1e-10)
shQR_eig, shQR_iter = shiftQR(fiedler(25), tol=1e-10)
err_rel_QR = abs((QR_eig - actual) / actual)
err_rel_shQR = abs((shQR_eig - actual) / actual)

print('Error relativo')
print('QR simple sin limite: ' + str(linear.norm(err_rel_QR)))
print('QR con shift dinámico sin limite: ' + str(linear.norm(err_rel_shQR)))

#Numero de iteraciones:
print('Iteraciones QR simple:'+str(QR_iter))
print('Iteraciones QR shift dinamico:'+str(shQR_iter))
print('simple_iters/shift_iters = ' + str(QR_iter/shQR_iter))

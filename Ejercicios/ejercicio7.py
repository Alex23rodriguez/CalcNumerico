import sys
sys.path.append('../')
import numpy as np
import scipy.linalg as linear
from eigenvalues import *
from matplotlib import pyplot as plt
import imageio


# # Ejercicio 7

# Queremos probar el metodo SVD. Para esto, primero cargamos una imagen original:
img = imageio.imread("assets/small.jpg")
#plt.imshow(img)
#plt.show()

# Luego separamos los valores RGB en 3 matrices:

# In[3]:

img = img.astype('float')
red = np.array(img[:,:,0])
green = np.array(img[:,:,1])
blue = np.array(img[:,:,2])


# Hagamos una prueba sin reducir el rango de la matriz:

# In[4]:

tolerance = 0


# In[5]:

u1, s1, v1 = SVD(red,tolerance)
u2, s2, v2 = SVD(green,tolerance)
u3, s3, v3 = SVD(blue,tolerance)

print('\n-- Imágen original --')
print('Rango de las matrices RGB: ')
print(len(s1), len(s2), len(s3))

plt.imshow(makeNewImg([u1,s1,v1],[u2,s2,v2],[u3,s3,v3], 'assets\\tol_0.png'))
plt.show()

tolerance = 100

u1, s1, v1 = SVD(red,tolerance)
u2, s2, v2 = SVD(green,tolerance)
u3, s3, v3 = SVD(blue,tolerance)

print('\n-- Reconstrucción tras truncar valores singulares menores a 100 --')
print('Rango de las matrices RGB: ')
print(len(s1), len(s2), len(s3))

print('No hubo perdida de detalle, salvo el tono blanco del fondo')

# In[11]:

plt.imshow(makeNewImg([u1,s1,v1],[u2,s2,v2],[u3,s3,v3], 'assets\\tol_100.png'), )
plt.show()

tolerance = 1000

u1, s1, v1 = SVD(red,tolerance)
u2, s2, v2 = SVD(green,tolerance)
u3, s3, v3 = SVD(blue,tolerance)

print('\n-- Reconstrucción tras truncar valores singulares menores a 1000 --')
print('Rango de las matrices RGB: ')
print(len(s1), len(s2), len(s3))

print('A pesar de que existe una evidente perdida de detalle, aun es posible reconozer la imagen original')

plt.imshow(makeNewImg([u1,s1,v1],[u2,s2,v2],[u3,s3,v3], 'assets\\tol_1000.png'))
plt.show()

n = 3

u1, s1, v1 = SVD(red, n = n)
u2, s2, v2 = SVD(green, n = n)
u3, s3, v3 = SVD(blue, n = n)

print('\n-- Reconstrucción conservando 3 valores singulares --')
print('Rango de las matrices RGB: ')
print(len(s1), len(s2), len(s3))

print('La imagen original esta basicamente perdida, pero aun es posible reconocer la forma.')

plt.imshow(makeNewImg([u1,s1,v1],[u2,s2,v2],[u3,s3,v3], 'assets\\rank_3.png'))
plt.show()

n = 1

u1, s1, v1 = SVD(red, n = n)
u2, s2, v2 = SVD(green, n = n)
u3, s3, v3 = SVD(blue, n = n)

print('\n-- Reconstrucción conservando 1 sólo valor singular --')
print('Rango de las matrices RGB: ')
print(len(s1), len(s2), len(s3))

print('La imagen se pierde totalmente, y solo permanecen los colores dominantes en sus posiciones relativas.')

plt.imshow(makeNewImg([u1,s1,v1],[u2,s2,v2],[u3,s3,v3], 'assets\\rank_1.png'))
plt.show()



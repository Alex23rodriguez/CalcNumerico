import numpy as np
import scipy.linalg as linear
#Power Methods
def powerMethod(A, q, tol, maxIter='100000'):
	w = q / linear.norm(q)
    l = w.dot( A.dot(w) )
    i=0
    while(i<maxIter and linear.norm(A.dot(w) - l*w)>= tol*linear.norm(A.dot(w))):
        i += 1
        q = A.dot(w)
        l = w.dot( A.dot(w) )
        w = q / LA.norm(q)
    return w,l

def inversePower(A, q, tol, maxIter='100000'):
##Codigo potencia inversa

def inversePowerShift(A, q, r, tol, maxIter='100000'):
##Codigo potencia inversa con shift estatico

def inversePowerRayleigh(A, q, tol, maxIter='100000'):
##Codigo potencia inversa con shift de Rayleigh 
	
#QR Methods
def simpleQR(A, tol):
    while linear.norm(A.diagonal(-1))>tol:
        [Q,R]=linear.qr(A)
        A=R.dot(Q)
    return A
	
def shiftQRStep(A, tol):
    while A[-1,-2]>tol:
        shift=A[-1,-1]
        [Q,R]=linear.qr(A-shift*np.identity(len(A)))
        A=R.dot(Q)+shift*np.identity(len(A))
    return A
	
def shiftQR(A, tol):
    A=linear.hessenberg(A)
    eigenvalues=[]
    while(len(A)>1):
        A=shiftQRStep(A,tol)
        eigenvalues.append(A[-1,-1])
        A=A[:-1,:-1]
    eigenvalues.append(A[0,0])
    return eigenvalues	

#SVD Methods
def SVD(A):
#codigo svd
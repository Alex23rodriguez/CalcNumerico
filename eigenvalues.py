import numpy as np
import scipy.linalg as linear

def powerMethod(A, q, tol, maxIter=1e3):
    w = q/linear.norm(q)
    i = 0
    flag = False
    while(i < maxIter and not flag):
        i += 1
        q = A.dot(w)
        l = w.dot(q)
        w = q/linear.norm(q)
        flag = linear.norm(A.dot(w)-l*w) <= tol*linear.norm(A.dot(w))
    return w,l,i

def inversePower(A, q, tol, maxIter=1e3):
    w = q/linear.norm(q)
    i = 0
    flag = False
    while(i < maxIter and not flag):
        i += 1
        q = linear.solve(A,w)
        l = w.dot(q)
        w = q/linear.norm(q)
        flag = linear.norm(A.dot(w)-(1/l)*w) <= tol*linear.norm(A.dot(w))
    return w,1/l,i
        
def inversePowerShift(A, q, r, tol, maxIter=1e3):
    w = q/linear.norm(q)
    i = 0
    flag = False
    while(i < maxIter and not flag):
        i += 1
        q = linear.solve(A-r*np.identity(len(A)),w)
        l = w.dot(q)
        w = q/linear.norm(q)
        flag = linear.norm(A.dot(w)-(1/l+r)*w) <= tol*linear.norm(A.dot(w))
    return w,(1/l+r),i

def inversePowerRayleigh(A, q, tol, maxIter=1e3): 
    w = q/linear.norm(q)
    l = w.dot(A.dot(w))
    i = 0
    flag = False
    while(i < maxIter and not flag):
        i += 1
        #Agregar try-catch para matrix singular, terminar ejecucion
        q = linear.solve(A-l*np.identity(len(A)),w)
        w = q/linear.norm(q)
        l = w.dot(A.dot(w))
        flag = linear.norm(A.dot(w)-l*w) <= tol
    return w,l,i

#QR Methods
def simpleQR(A, tol, maxIter=1e3, symmetric=False):
    i = 0
    if not symmetric:
        while i < maxIter and linear.norm(A.diagonal(-1)) > tol:
            [Q,R] = linear.qr(A)
            A = R.dot(Q)
            i += 1
        return A,i
    else:
        V=np.identity(len(A))
        while i < maxIter and linear.norm(A.diagonal(-1)) > tol:
            [Q,R] = linear.qr(A)
            A = R.dot(Q)
            V = V.dot(Q)
            i += 1
        return A,V,i

def shiftQRStep(A, tol, symmetric=False):
    i = 0
    if not symmetric:
        while abs(A[-1,-2]) > tol:
            shift = A[-1,-1]
            [Q,R] = linear.qr(A-shift*np.identity(len(A)))
            A = R.dot(Q)+shift*np.identity(len(A))
            i += 1
        return A,i
    else:
        #Acumular eigenvectores
        while abs(A[-1,-2])>tol:
            delta = 0.5*(A[-2,-2]-A[-1,-1])
            shift = A[-1,-1] - np.sign(delta)*((A[-2,-1]**2)/(abs(delta)+np.sqrt(delta**2+A[-2,-1]**2)))
            [Q,R] = linear.qr(A-shift*np.identity(len(A)))
            A = R.dot(Q)+shift*np.identity(len(A))
            i += 1
        return A,i

def shiftQR(A, tol, maxIter=1e3, symmetric=False):
    i = 0
    A = linear.hessenberg(A)
    eigenvalues = []
    if not symmetric:
        while i<maxIter and len(A)>1:
            [A,j] = shiftQRStep(A,tol)
            eigenvalues.append(A[-1,-1])
            A = A[:-1,:-1]
            i += j
        eigenvalues.append(A[0,0])
        return np.array(eigenvalues),i
    else:
        while i<maxIter and len(A)>1:
            [A,j] = shiftQRStep(A,tol, True)
            eigenvalues.append(A[-1,-1])
            A = A[:-1,:-1]
            i += j
        eigenvalues.append(A[0,0])
        return np.array(eigenvalues),i

#SVD Methods
def SVD(A):
#codigo svd
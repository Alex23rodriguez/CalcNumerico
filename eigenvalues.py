import numpy as np
import scipy.linalg as linear

#Power Methods
def powerMethod(A, q, tol=1e-7, maxIter=1e3):
    """
    Applies the power method to a matrix A with an initial guess q to approximate the dominant eigenpair. 
	The method generates the sequence::
	    q, Aq, A^2q, ...
	Which converges to a dominant eigenvector of A under the following conditions::
		A is semisimple
		A has a dominant eigenvalue
		If v1, v2, ..., vm are the eigenvectors of A, ordered acording to the magnitude of their eigenvalues, 
		q=c1v1+c2v2+...+cmvm with c1 != 0
    
	Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose dominant eigenpair will be calculated.
    q : (M) double ndarray
        The initial guess of the eigenvector.
    tol : double, optional
        Maximum relative error. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
    
	Returns
    -------
    w : (M) double ndarray
        The normalized dominant eigenvector calculated.
    l : double
        The eigenvalue associated with ``w``.
    i : int
        The number of iterations to reach the approximation.
    """
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

def inversePower(A, q, tol=1e-7, maxIter=1e3):
    """
    Applies the inverse power method to a matrix A with an initial guess q to approximate the smallest eigenpair (i.e. the dominant eigenpair of A^-1). 
	The method generates the sequence::
	    q, (A^-1)q, (A^-1)^2q, ...
	Which converges to the 'smallest' eigenvector of A under the following conditions::
		A is semisimple
		A has a 'smallest' eigenvalue (i.e. |lk|<|li| for all i != k)
		If v1, v2, ..., vm are the eigenvectors of A, ordered acording to the magnitude of their eigenvalues, 
		q=c1v1+c2v2+...+cmvm with cm != 0
    
	Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose 'smallest' eigenpair will be calculated.
    q : (M) double ndarray
        The initial guess of the eigenvector.
    tol : double, optional
        Maximum relative error. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
    
	Returns
    -------
    w : (M) double ndarray
        The normalized 'smallest' eigenvector calculated.
    l : double
        The eigenvalue associated with ``w``.
    i : int
        The number of iterations to reach the approximation.
    """
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
        
def inversePowerShift(A, q, r, tol=1e-7, maxIter=1e3):
    """
    Applies the inverse power method with static shift to a matrix A with an initial guess q and a shift r to approximate an eigenpair. 
	The method generates the sequence::
	    q, ((A^-1)-rI)q, ((A^-1)-rI)^2q, ...
	Which converges to the 'smallest' eigenvector of (A-rI) under the following conditions::
		A is semisimple
		(A-rI) has a 'smallest' eigenvalue (i.e. |lk|<|li| for all i != k)
		If v1, v2, ..., vm are the eigenvectors of (A-rI), ordered acording to the magnitude of their eigenvalues, 
		q=c1v1+c2v2+...+cmvm with cm != 0
    
	Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose eigenpair will be calculated.
    q : (M) double ndarray
        The initial guess of the eigenvector.
	r : double
		The static shift applied to the matrix.
    tol : double, optional
        Maximum relative error. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
    
	Returns
    -------
    w : (M) double ndarray
        The normalized eigenvector calculated.
    l : double
        The eigenvalue associated with ``w``.
    i : int
        The number of iterations to reach the approximation.
    """
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

def inversePowerRayleigh(A, q, tol=1e-7, maxIter=1e3):
    """
    Applies the inverse power method with dynamic shift to a matrix A with an initial guess q to approximate an eigenpair. 
	The method generates the sequence::
	    q, ((A^-1)-r1I)q, ((A^-1)-r2I)((A^-1)-r1I)q, ...
	Where ri is the dynamic shift calculated each iteration using the Rayleigh Quotient.
	This converges to the an eigenvector of A under conditions similar to the invese power method with static shift, however the eigenpair it converges to depends on the initial guess.
    
	Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose eigenpair will be calculated.
    q : (M) double ndarray
        The initial guess of the eigenvector.
    tol : double, optional
        Maximum relative error. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
    
	Returns
    -------
    w : (M) double ndarray
        The normalized eigenvector calculated.
    l : double
        The eigenvalue associated with ``w``.
    i : int
        The number of iterations to reach the approximation.
    """
    w = q/linear.norm(q)
    l = w.dot(A.dot(w))
    i = 0
    flag = False
    while(i < maxIter and not flag):
        i += 1
        q = linear.solve(A-l*np.identity(len(A)),w)
        w = q/linear.norm(q)
        l = w.dot(A.dot(w))
        flag = linear.norm(A.dot(w)-l*w) <= tol
    return w,l,i

#QR Methods
def simpleQR(A, tol=1e-7, maxIter=1e3, symmetric=False):
    """
    Applies the simple QR algorithm to a matrix A, which generates the sequence::
	    A_(i)=QR
	    A_(i+1)=RQ
	This sequence converges to an upper triangle matrix with the eigenvalues of A in the diagonal.
	If A is symmetric, the normal eigenvectors associated to A can also be calculated by accumulating the Q matrices of each iteration.
	
	Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose eigenvalues will be calculated.
    tol : double, optional
        Maximum relative error. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
	symmetric : bool, optional
		Whether A is symmetric or not. Default is False.
    
	Returns
    -------
    A : (M,M) double ndarray
        The approximated real upper triangle matrix, containing the eigenvalues in the diagonal
    V : (M,M) double ndarray
        The normal eigenvectors. Only returned if ``symmetric=True``.
    i : int
        The number of iterations to reach the approximation.
    """
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

def shiftQRStep(A, tol=1e-7, maxIter=1e3, symmetric=False):
    """
    Applies one step of the QR algorithm with dynamic to a matrix A, which generates the sequence::
	    (A_(i)+rI)=QR
	    A_(i+1)=RQ-rI
	Where r is the dynamic shift calculated each iteration. The shift used is given by the Rayleigh Quotient, unless the matrix is symmetric, in which case
	the Wilkinson shift is used. One step of the algorithm reduces the last row of A to a row of zeroes except for the eigenvalue in the last entry.
	Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose eigenvalues will be calculated.
    tol : double, optional
        Maximum relative error. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
	symmetric : bool, optional
		Whether A is symmetric or not. Default is False.
    
	Returns
    -------
    A : (M,M) double ndarray
        The matrix A, with the last row of A all zeroes except the eigenvalue in the last entry.
    i : int
        The number of iterations to reach the approximation.
    """
    i = 0
    if not symmetric:
        while i<maxIter and abs(A[-1,-2]) > tol:
            shift = A[-1,-1]
            [Q,R] = linear.qr(A-shift*np.identity(len(A)))
            A = R.dot(Q)+shift*np.identity(len(A))
            i += 1
        return A,i
    else:
        while i<maxIter and abs(A[-1,-2])>tol:
            delta = 0.5*(A[-2,-2]-A[-1,-1])
            shift = A[-1,-1] - np.sign(delta)*((A[-2,-1]**2)/(abs(delta)+np.sqrt(delta**2+A[-2,-1]**2)))
            [Q,R] = linear.qr(A-shift*np.identity(len(A)))
            A = R.dot(Q)+shift*np.identity(len(A))
            i += 1
        return A,i

def shiftQR(A, tol=1e-7, maxIter=1e3, symmetric=False):
    """
    Applies the QR algorithm with dynamic shift to a matrix A, which consists using the ``shiftQRStep`` function on A, then taking the resulting eigenvalue
	and reducing the dimension of A to (M-1,M-1) until A becomes a single number.
	This algorithm converges to an upper triangle matrix with the eigenvalues of A in the diagonal.
	
	Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose eigenvalues will be calculated.
    tol : double, optional
        Maximum relative error. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
	symmetric : bool, optional
		Whether A is symmetric or not. Default is False.
    
	Returns
    -------
    A : (M) double ndarray
        The eigenvalues of A
    i : int
        The number of iterations to reach the approximation.
    """
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
    print('SVD')
import numpy as np
import scipy.linalg as linear
import imageio

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
    
    #Initialize w as the normailized version of q
    w = q/linear.norm(q)
    
    #Initialize i (a counter for the number of iterations) at 0 and flag (a boolean with the relative tolerance criteria) with False
    i = 0
    flag = False
    
    while i < maxIter and not flag:
        
        """
        A step of the power method:
            q = A*w
            l = <w,q>
            w = q/norm(q)
        """
        q = A.dot(w)
        l = w.dot(q)
        w = q/linear.norm(q)
        
        #Evaluate the relative tolerance criteria and increase the counter by one
        flag = linear.norm(A.dot(w)-l*w) <= tol*linear.norm(A.dot(w))
        i += 1
    
    return w, l, i

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
    
    #Initialize w as the normailized version of q
    w = q/linear.norm(q)
    
    #Initialize i (a counter for the number of iterations) at 0 and flag (a boolean with the relative tolerance criteria) with False
    i = 0
    flag = False
    
    while i < maxIter and not flag:
        
        """
        A step of the inverse power method:
            q = (A^-1)*w <-> A*q = w
            l = <w,q>
            w = q/norm(q)
        """
        q = linear.solve(A, w)
        l = w.dot(q)
        w = q/linear.norm(q)
        
        #Evaluate the relative tolerance criteria and increase the counter by one
        flag = linear.norm(A.dot(w)-(1/l)*w) <= tol*linear.norm(A.dot(w))
        i += 1
    
    #The returned eigenvalue is 1/l, since l is approximating the dominant eigenvalue of A^-1
    return w, 1/l, i
        
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
    
    #Initialize w as the normailized version of q
    w = q/linear.norm(q)
    
    #Initialize i (a counter for the number of iterations) at 0 and flag (a boolean with the relative tolerance criteria) with False
    i = 0
    flag = False
    
    while i < maxIter and not flag:
        
        """
        A step of the inverse power method with static shift:
            q = ((A-r*I)^-1)*w <-> (A-r*I)*q = w
            l = <w,q>
            w = q/norm(q)
        """
        q = linear.solve(A-r*np.identity(len(A)), w)
        l = w.dot(q)
        w = q/linear.norm(q)
        
        #Evaluate the relative tolerance criteria and increase the counter by one
        flag = linear.norm(A.dot(w)-(1/l+r)*w) <= tol*linear.norm(A.dot(w))
        i += 1
    
    #The returned eigenvalue is (1/l)+r, since l is approximating the dominant eigenvalue of (A-rI)^-1
    return w, (1/l+r), i

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
    
    #Initialize w as the normailized version of q and l as <w,A*w>
    w = q/linear.norm(q)
    l = w.dot(A.dot(w))
    
    #Initialize i (a counter for the number of iterations) at 0 and flag (a boolean with the relative tolerance criteria) with False
    i = 0
    flag = False
    
    while i < maxIter and not flag:
        
        """
        A step of the inverse power method with static shift:
            q = ((A-l*I)^-1)*w <-> (A-l*I)*q = w
            w = q/norm(q)
            l = <w,A*w>
        """
        q = linear.solve(A-l*np.identity(len(A)), w)
        w = q/linear.norm(q)
        l = w.dot(A.dot(w))
        
        #Evaluate the relative tolerance criteria and increase the counter by one
        flag = linear.norm(A.dot(w)-l*w) <= tol
        i += 1
    
    #The returned eigenvalue is simply l since the dynamic shift is taken as the approximation during each step
    return w, l, i

#QR Methods
def simpleQR(A, tol=1e-7, maxIter=1e3, calcQ=False):
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
    calcQ : bool, optional
        Whether to accumulate Q matrices or not. Default is False.
    
    Returns
    -------
    l : (M) double ndarray
        The eigenvalues of A
    V : (M,M) double ndarray
        The normal eigenvectors of A. Only returned if ``calcQ=True``.
    i : int
        The number of iterations to reach the approximation.
    """
    
    #Initialize i (a counter for the number of iterations) at 0 and flag (a boolean with the relative tolerance criteria) with False
    i = 0
    flag = False
    
    #Decide whether or not to accumulate Q based on the parameter
    if not calcQ:
        
        while i < maxIter and not flag :
            
            """
            A step of the simple QR algorithm:
                Calculate the QR decomposition of A (A = Q*R)
                A = R*Q
            """
            [Q, R] = linear.qr(A)
            A = R.dot(Q)
            
            #Evaluate the relative tolerance criteria and increase the counter by one
            flag = linear.norm(A.diagonal(-1)) < tol
            i += 1
        
        #Store the eigenvalues in l
        l = A.diagonal()
        return absSort(l), i
        
    else:
        
        #Initialize V as the identity in order to accumulate the Q matrices in it
        V=np.identity(len(A))
        
        while i < maxIter and not flag:
            
            """
            A step of the simple QR algorithm:
                Calculate the QR decomposition of A (A = Q*R)
                A = R*Q
                V = V*Q
            """
            [Q, R] = linear.qr(A)
            A = R.dot(Q)
            V = V.dot(Q)
            
            #Evaluate the relative tolerance criteria and increase the counter by one
            flag = linear.norm(A.diagonal(-1)) < tol
            i += 1
        
        #Order l and V according to the magnitude of the eigenvalues
        l = A.diagonal()
        [l, V] = pairSort(l, V)
        
        return l, V, i

def shiftQRStep(A, tol=1e-7, maxIter=1e3, calcQ=False):
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
    calcQ : bool, optional
        Whether to accumulate Q matrices or not. Default is False.
    
    Returns
    -------
    A : (M,M) double ndarray
        The matrix A, with the last row of A all zeroes except the eigenvalue in the last entry.
    V : (M, M) double ndarray
        Accumulated Q matrices. Only returned if ``calcQ=True``.
    i : int
        The number of iterations to reach the approximation.
    """
    
    #Initialize i (a counter for the number of iterations) at 0 and flag (a boolean with the relative tolerance criteria) with False
    i = 0
    flag = False
    
    #Decide whether or not to accumulate Q based on the parameter
    if not calcQ:
        
        while i < maxIter and not flag:
            
            """
            A step of the dynamic shift QR algorithm:
                Calculate the shift with using the Rayleigh quotient, which in this case is the last value in the diagonal of A (shift = A[-1,-1])
                Calculate the QR decomposition of A-shift*I (A-shift*I = Q*R)
                A = R*Q + shift*I
            """
            shift = A[-1,-1]
            [Q, R] = linear.qr(A-shift*np.identity(len(A)))
            A = R.dot(Q)+shift*np.identity(len(A))
            
            #Evaluate the relative tolerance criteria and increase the counter by one
            flag = abs(A[-1,-2]) < tol
            i += 1
            
        return A,i
        
    else:
        
        #Initialize V as the identity in order to accumulate the Q matrices in it
        V = np.identity(len(A))
        
        while i < maxIter and not flag:
            
            """
            A step of the dynamic shift step QR algorithm:
                Calculate the shift with using the Wilkinson Shift
                Calculate the QR decomposition of A-shift*I (A-shift*I = Q*R)
                A = R*Q + shift*I
                V = V*Q
            """
            delta = 0.5*(A[-2,-2]-A[-1,-1])
            shift = A[-1,-1] - np.sign(delta)*((A[-2,-1]**2)/(abs(delta)+np.sqrt(delta**2+A[-2,-1]**2)))
            [Q, R] = linear.qr(A-shift*np.identity(len(A)))
            A = R.dot(Q)+shift*np.identity(len(A))
            V = V.dot(Q)
            
            #Evaluate the relative tolerance criteria and increase the counter by one
            flag = abs(A[-1,-2]) < tol
            i += 1
        
        return A, V, i

def shiftQR(A, tol=1e-7, maxIter=1e3, calcQ=False):
    """
    Applies the QR algorithm with dynamic shift to a matrix A, which consists using the ``shiftQRStep`` function on A, then taking the resulting eigenvalue
    and reducing the dimension of A to (M-1,M-1) until A becomes a single number.
    This algorithm converges to an upper triangle matrix with the eigenvalues of A in the diagonal. If A is symmetric, the Q matrices of each iteration can be accumulated to approximate the normal eigenvectors
    
    Parameters
    ----------
    A : (M, M) double ndarray
        A real square matrix whose eigenvalues will be calculated.
    tol : double, optional
        Maximum relative error per step. Default is 1e-7.
    maxIter : int, optional
        Maximum number of iterations. Default is 1e3.
    calcQ : bool, optional
        Whether to accumulate Q matrices or not. Default is False.
    
    Returns
    -------
    l : (M) double ndarray
        The eigenvalues of A
    V : (M, M) double ndarray
        The normal eigenvectors of A. Only returned if ``calcQ=True``.
    i : int
        The number of iterations to reach the approximation.
    """
    
    #Initialize i (a counter for the number of iterations)
    i = 0
    
    #Reduce A to its hessenberg form and initialize an empty list for the eigenvalues
    [A, QH] = linear.hessenberg(A, True)
    l = []
    
    #Decide whether to accumulate Q based on the parameter
    if not calcQ:
        
        while i < maxIter and len(A) > 1:
            
            """
            A step of the shift QR algorithm:
                Do the dynamic shift step QR algorithm
                Save the last value of the diagonal of A in the eigenvalue list
                Reduce A from a m x m matrix to a (m-1) x (m-1) matrix
            """
            [A, j] = shiftQRStep(A, tol, maxIter-i)
            l.append(A[-1,-1])
            A = A[:-1,:-1]
            
            #Accumulate the number of steps in i
            i += j
        
        #Add the last value of A to the eigenvalue list and transform it into an array
        for i in range(len(A))[::-1]:
            l.append(A[i,i])
        l = np.array(l)
        
        #The eigenvalue list is ordered according to magnitude
        return absSort(l), i
        
    else:
        
        #Initialize V as QH (the matrix used to transform A into a hessenberg matrix) in order to accumulate the Q matrices in it and a counter k for the padding needed in each iteration
        V = QH
        k = 0
        
        while i < maxIter and len(A) > 1:
            
            """
            A step of the shift QR algorithm:
                Do the dynamic shift step QR algorithm
                Save the last value of the diagonal of A in the eigenvalue list
                Reduce A from a m x m matrix to a (m-1) x (m-1) matrix
                Pad the accumulated Q matrices with zeros in order to match the original dimensions
                V = V*Q
            """
            [A, Q, j] = shiftQRStep(A, tol, maxIter-i, True)
            l.append(A[-1,-1])
            A = A[:-1,:-1]
            Q = pad_diag(Q,k)
            V = V.dot(Q)
            
            #Increase the counter k and accumulate the number of steps in i
            k += 1
            i += j
        
        #Add the remaining values of A's diagonal to the eigenvalue list and transform it into an array
        for i in range(len(A))[::-1]:
            l.append(A[i,i])
        l.reverse()
        l = np.array(l)
        
        #Order eigenvalues and V according to the magnitude of the eigenvalues
        [l, V] = pairSort(l, V)
        
        return l, V, i

#SVD Methods
def SVD(A, tol=1e-7, n = None):
    """
    Given a matrix A, return its Singular Value Decomposition. That is, AV=SU
    
    Parameters
    ----------
    A : (M, N) double ndarray
        A real matrix whose SVD will be calculated.
    tol : double, optional
         Cutoff point that determines 'important' singular values. Default is 1e-7.
    n : integer, optional
        Keeps only the first n singular values. Overrides tolerance if not None.

    Returns
    -------
    U : (M, k) double ndarray
        A unitary matrix that contains the first k left singular vectors of A.
    S : (k, k) double ndarray
        A square real diagonal matrix whose elements are the k singular values of A greater that the tolerance. 
    V : (k, N) 
        A unitary matrix that contains the first k right singular vectors of A.
    """
    
    rows, cols = A.shape
    
    #Method procedes with lower rank matrix between AA* and A*A
    if rows < cols: 
        [L, U, _ ] = shiftQR(np.dot(A,A.T), calcQ=True)
    else:
        [L, V, _ ] = shiftQR(np.dot(A.T,A), calcQ = True)
    
    
    if n == None:
        #Cutoff eigenvalues smaller than tol^2
        tol2 = tol**2
        k = 0
        while k < len(L) and L[k] > tol2: 
            k+=1
    else:
        #Keep first n eigenvalues
        k = n
    
    #The singular values of A are the square root of the eigenvalues returned
    singularValues = [x**0.5 for x in L[:k]] 
    S = np.diag(singularValues[:k])
    
    #Calculate remaining singular vectors depending on path chosen at the beginning
    if rows < cols: 
        U = U[:,:k]
        V = np.array([1/singularValues[i]*A.T.dot(U[:,i]) for i in range(k)]).T
    else:
        V = V[:,:k]
        U = np.array([1/singularValues[i]*A.dot(V[:,i]) for i in range(k)]).T
    return U, S, V

#Auxiliary Functions
#Pads A with i rows and columns, adding 1 in the diagonal 
def pad_diag(A,i):
    
    if i>0:
        B = np.identity(len(A)+i)
        B[:-i,:-i] = A
    else:
        B = A
    
    return B

#Orders a vector according to the magnitud of its elements (greatest first), then permutes the matrix given accordingly
def pairSort(v, A):
    order = np.argsort(abs(v))[::-1]
    v = np.array([v[i] for i in order])
    A = np.array([A[:,i] for i in order]).T
    return v, A

#Orders a vector according to itsthe magnitude of its elements
def absSort(v):
    order = np.argsort(abs(v))[::-1]
    v = np.array([v[i] for i in order])
    return v

def makeNewImg(r, g, b, filepath='.\\aux.png'):
    '''
    Auxiliary function.
    Reconstructs image given the SVD decomposition of the RGB matrices
    Parameters
    ----------
    r : (u1, s1, v1) 
        A tuple or list containing SVD decomposition of the red channel
    g : (u2, s2, v2)
        A tuple or list containing SVD decomposition of the green channel
    b : (u3, s3, v3)
        A tuple or list containing SVD decomposition of the blue channel
    filepath : String
        A string indicating the path in which to save the image

    Returns
    -------
    newImg : (M,N) uint8 ndarray of the reconstructed image
    '''
    newRed = r[0].dot(r[1]).dot(r[2].T)
    newGreen = g[0].dot(g[1]).dot(g[2].T)
    newBlue = b[0].dot(b[1]).dot(b[2].T)
    newImg = []
    for i in range(len(newRed)):
        newImg.append([])
        for j in range(len(newRed[0])):
            newImg[i].append([newRed[i][j], newGreen[i][j], newBlue[i][j]])
            
    imageio.imwrite(filepath, np.array(newImg)) #picture is saved and reloaded to effectivly parse it back to uint8
    newImg=imageio.imread(filepath)
    return newImg
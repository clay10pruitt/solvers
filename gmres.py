import math
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from scipy.sparse import csc_array
from scipy.sparse import diags

ITER = 1000
TOL = 1e-10

debug = True

def debug_print(message):
    if debug:
        print(message)

def gmres(A, b, x, iterations=ITER, tolerance=TOL):

    debug_print("Running GMRES...")

    n = max(A.shape)
    m = iterations

    r = b - A@x
    r_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(b)
    residual = r_norm/b_norm

    debug_print("n:\t" + str(n) + "\titers:\t" + str(m) + "\tinitial residual: " + str(residual)+"\n")
    #residual = np.dot(np.linalg.norm(r), np.linalg.inv(b_norm)) # matrix right division

    # initialize 1D vectors
    sn = np.zeros(m)
    cs = np.zeros(m)
    e1 = np.zeros(m+1)
    e1[0] = 1
    e = [residual]
    H = np.zeros((1,1))
    Q = (r/r_norm)
    beta = r_norm*e1

    for k in range(3):
        debug_print("="*32)
        debug_print("\tITERATION:\t" + str(k))
        debug_print("="*32)

        debug_print("H:\n" + str(H))
        debug_print("Q:\n" + str(Q))

        # expand  H and Q for adding new entries
        expand = np.zeros((H.shape[0]+1, H.shape[0]))
        #debug_print("H EXPAND:\n" + str(expand))
        expand[:H.shape[0], :H.shape[1]] = H
        H = expand
        
        expand =  np.zeros((Q.shape[0], Q.shape[1]+1))
        #debug_print("Q EXPAND:\n" + str(expand))
        expand[:, :Q.shape[1]] = Q
        Q = expand

        # arnoldi
        H[:k+1, k], Q[:, k+1] = arnoldi(A, Q, k)

        debug_print("HESSENBERG:\n" + str(H))
        #debug_print("KRYLOV:\n" + str(Q))
        
        continue

        # eliminate last element in k^th row of H, then update rotation matrix
        H[:k+1, k], cs[k], sn[k] = apply_givens_rotation(H[:k+1, k], cs, sn, k)

        # update the residual vector
        beta[k+1] = -sn[k]@beta[k]
        beta[k] = cs[k]@beta[k]
        residual = abs(beta[k+1])/b_norm

        if residual < tolerance:
            break
    
    return
    # calculate result
    y, res, rank, s = np.linalg.lstsq(H[1:k, 1:k], beta[1:k]) # matrix left division
    x = x + Q[:, 1:k] @ y

    return x

"""
Run the Arnoldi iteration on a matrix A with Krylov subspace Q containing k vectors.
@param A: square ndarray
@param Q:
@param k: number of columns in the Krylov matrix
"""
def arnoldi(A, Q, k):
    debug_print("Calculating Arnoldi...")
    # TODO: Does the initial vector need to be normalized?
    q = A@Q[:,k] # initial Krylov vector
    h = np.zeros(k+1)
    for i in range(k): # modified Gram-Schmidt, keeping the Hessenberg
        h[i] = q.T@Q[:,i]
        q = q - h[i]*Q[:,i]
    h[k] = np.linalg.norm(q)
    q = q/h[k]
    #debug_print("h:\n" + str(h) + "\nq:\n" + str(q))
    return h, q

def apply_givens_rotation(h, cs, sn, k):

    debug_print("Applying Givens rotation...")

    # apply for ith column
    for i in range (k-1):
        temp = cs[i]*h[i] + sn[i]*h[i+1]
        h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1]
        h[i] = temp

    # update the next sin/cos values for rotation
    debug_print("h:\n" + str(h))
    [cs_k, sn_k] = givens_rotation(h[k], h[k+1])

    # eliminate H[i+1, i]
    h[k] = cs_k * h[k] + sn_k*h[k+1]
    h[k+1] = 0

def givens_rotation(v1, v2):
    t = math.sqrt(np.dot(v1, v1) + np.dot(v2, v2))
    cs = v1 * (1/t)
    sn = v2 * (1/t)

n = 10
diag = [(i%2)+1 for i in range(0,n)]
A = diags(diag, format="csc")
b = np.atleast_2d([rand.randint(1,10) for i in range(0,n)]).T
x0 = np.atleast_2d([1 for i in range(0, n)]).T
#b = csc_array(np.atleast_2d([rand.randint(1,10) for i in range(0,n)]).T, dtype=float)
#x0 = csc_array(np.atleast_2d([1 for i in range(0, n)]).T, dtype=float)

print ("=" * 4 + " A " + "=" * 8)
print(A.toarray())
print ("=" * 4 + " b " + "=" * 8)
print (b)
#print (b.toarray())
print ("=" * 4 + " x " + "=" * 8)
print(x0)
#print(x0.toarray())
print ("\n"*2)

gmres(A, b, x0)
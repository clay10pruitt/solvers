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

    for k in range(m):

        debug_print("="*32)
        debug_print("\tITERATION:\t" + str(k))
        debug_print("="*32)

        # expand  H
        # TODO This is bad
        expand = np.zeros((H.shape[0]+1, H.shape[0]+1))
        expand[:H.shape[0], :H.shape[0]] = H
        H = expand

        # arnoldi
        H[:k+1, k], Q[:, k+1] = arnoldi(A, Q, k)

        debug_print("HESSENBERG: ")
        debug_print(H)

        # eliminate last element in k^th row of H, then update rotation matrix
        H[0:k+1, k], cs, sn = apply_givens_rotation(H[1:k+1, k], cs, sn, k)

        # update the residual vector
        beta[k+1] = -sn[k]@beta[k]
        beta[k] = cs[k]@beta[k]
        residual = abs(beta[k+1])/b_norm

        if residual < tolerance:
            break
    
    # calculate result
    y, res, rank, s = np.lingalg.lstsq(H[1:k, 1:k], beta[1:k]) # matrix left division
    x = x + Q[:, 1:k] @ y

    return x

def arnoldi(A, Q, k):
    debug_print("Calculating Arnoldi...")
    H = np.empty((k+1, k+1))
    q = A@Q[:,k]
    for i in range(k):
        h = q.T@Q[:,i]
        q = q - h@Q[:,i]
        H = np.append(H, h, axis=1)
    H = np.append(H, np.linalg.norm(q))
    q = q/np.linalg.norm(q)
    return H, q

def apply_givens_rotation(h, cs, sn, k):
    # apply for ith column
    for i in range (0, k):
        temp = cs[i]*h[i] + sn[i]*h[i+1]
        h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1]
        h[i] = temp

    # update the next sin/cos values for rotation
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
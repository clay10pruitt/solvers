import numpy as np
import matplotlib.pyplot as plt

ITER = 10000
TOL = 1e-10

"""
Executes the Steepest Descent method for determining the solution of a system of linear equations.
@param matrix: A
@param vector: b
@param x0: initial guses
@param iterations: max number of iterations to run (default ITER)
@param tolerance: minimum difference between iterative solutions until method halts execution
@return: NumPy array of each iterative guess and NumPy array of differences between each iteration
"""
def steepdesc(matrix, vector, x0, iterations=ITER, tolerance=TOL):
    x = [x0] # guesses
    r = [vector - matrix@x[0]] # residuals
    r_mag = [np.linalg.norm(r[-1])]
    p = [matrix@r[0]]
    
    k = 0
    while k < iterations and r_mag[-1] > tolerance:
        alpha = (r[k].T@r[k])/(r[k].T@p[k])
        x.append(x[k] + alpha*r[k])
        r.append(r[k] - alpha*p[k])
        r_mag.append(np.linalg.norm(r[-1]))
        p.append(matrix@r[k+1])
        k += 1

    return x, r_mag

A1 = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
b1 = np.array([6., 25., -11., 15.])
x1 = np.array([1, 2, 3, 4])

A2 = np.array([[10., -1., 2000., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3000., -1., 8.]])
b2 = np.array([6., 25., -11., 15.])
x2 = np.array([1, 2, 3, 4])

A3 = np.array([[1., 0.],
               [0., 2.]])
b3 = np.array([3., 7.])
x3 = np.array([2., 1.])

A4 = np.array([[1., 0.],
               [0., 200.]])
b4 = np.array([3., 7.])
x4 = np.array([2., 1.])

entries =[(A1, b1, x1, "Convergence of Steepest Descent Method (4x4, small  eigenvalue difference)"),
          (A2, b2, x2, "Convergence of Steepest Descent Method (4x4, large eigenvalue difference)"),
          (A3, b3, x3, "Convergence of Steepest Descent Method (2x2, small  eigenvalue difference)"),
          (A4, b4, x4, "Convergence of Steepest Descent Method (2x2, large eigenvalue difference)")]

for entry in entries:
    x, r_mag = steepdesc(entry[0], entry[1], entry[2])

    # plot points
    #plt.scatter(range(1, len(r_mag)+1), r_mag, c='red', marker='.')
    plt.plot(range(1, len(r_mag)+1), r_mag, c='red')
    if len(r_mag) < 30:
        plt.xticks(np.arange(2, len(r_mag)+1, 2))
    elif len(r_mag) < 100:
        plt.xticks(np.arange(5, len(r_mag)+1, 5))
    else:
        plt.xticks(np.arange(500, len(r_mag)+1, 500))
    plt.xlabel("Iteration")
    plt.ylabel("Magnitude of Residual Vector (r=b-Ax_k)")
    plt.title(entry[3])
    plt.show()

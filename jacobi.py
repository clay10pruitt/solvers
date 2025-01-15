import numpy as np
import matplotlib.pyplot as plt

ITER = 1000
TOL = 1e-10

"""
Executes the Jacobi method for determining the solution of a system of linear equations.
@param matrix: A
@param vector: b
@param iterations: max number of iterations to run (default ITER)
@param tolerance: minimum difference between iterative solutions until method halts execution
@return: NumPy array of each iterative guess and NumPy array of differences between each iteration
"""
def jacobi(matrix, vector, iterations=ITER, tolerance=TOL):
    x = np.zeros(vector.size)
    x_next = np.zeros(vector.size)
    x_all = [x, x_next]
    x_diff = []
    k = 0
    # only iterate if x^k and x^(k+1) are reasonably close
    while True: 
        # calculate our next guess
        for i in range(matrix.shape[0]):
            sigma = 0
            for j in range(matrix.shape[0]):
                if j == i:
                    continue
                sigma += np.dot(matrix[i, j], x_all[k][j])
            x_all[k+1][i] = (vector[i] - sigma)/matrix[i, i]
        x_diff.append(np.linalg.norm(x_all[k+1]-x_all[k]))
        if np.allclose(x_all[k], x_all[k+1], atol=tolerance, rtol=0.) or k >= iterations or (x_diff[-1] > 1000): 
            break
        x_all.append(np.zeros(vector.size))
        k += 1

    return x_all, x_diff

A1 = np.array([[2., 1.],
               [1., 2.]])

A2 = np.array([[200., 1.],
               [1., 200.]])

b1 = np.array([17., 251.])

entries = [(A1, b1, "Convergence of Jacobi Method (2x2, weak diagonal dominance)"), (A2, b1, "Convergence of Jacobi Method (2x2, strong diagonal dominance)")]

for entry in entries:
    x, diff = jacobi(entry[0], entry[1])

    # plot points
    plt.scatter(range(1, len(diff)+1), diff, c='red', marker='.')
    plt.plot(range(1, len(diff)+1), diff, c='red')
    plt.xticks(np.arange(1, len(diff)+1, 2))
    plt.xlabel("Iteration")
    plt.ylabel("Magnitude in Difference From Previous Iteration")
    plt.title(entry[2])
    plt.show()

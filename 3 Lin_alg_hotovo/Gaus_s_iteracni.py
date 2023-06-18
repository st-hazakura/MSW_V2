import numpy as np

def gauss_se_iteracni(A, b, initial_guess, max_iterations, tolerance):
    x = initial_guess.copy()
    n = len(b)
    iterations = 0
    residual_norm = np.inf
    
    while iterations < max_iterations and residual_norm > tolerance:
        x_prev = x.copy()
        
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]
        
        residual_norm = np.linalg.norm(x - x_prev)
        iterations += 1
        
        print("Iteration:", iterations)
        print("Solution:", x)
        print()
    
    if residual_norm <= tolerance:
        print("Converged to the solution")
    else:
        print("Maximum number of iterations reached without convergence")
    
    return x, iterations

A = np.array([[4, -1, 0, 2],
              [-1, 10, 2, -1],
              [0, 2, 7, -4],
              [2, -1, -1, 5]])

b = np.array([7, -10, 4, 7])
x0 = np.array([0, 0, 0, 0])
tolerance = 1e-6
max_iterations = 100

x, iterations = gauss_se_iteracni(A, b, x0, max_iterations, tolerance)

print("Solution:", x)
print("Number of iterations:", iterations)

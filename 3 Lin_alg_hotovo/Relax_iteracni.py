import numpy as np

def relax_iteracni(A, b, omega, initial_guess, max_iterations, tolerance):
    x = initial_guess.copy()
    n = len(b)
    iterations = 0
    
    for k in range(max_iterations):
        r = b - np.dot(A, x)
        
        if np.linalg.norm(r, np.inf) < tolerance:
            return x, iterations
        
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)
        
        iterations += 1
        
        print("Iteration:", iterations)
        print("Solution:", x)
        print()
    
    print("Maximální počet iterací dosažených bez konvergence")
    return x, iterations

A = np.array([[4.0, -1.0, 0.0, 2.0],
              [-1.0, 10.0, 2.0, -1.0],
              [0.0, 2.0, 7.0, -4.0],
              [2.0, -1.0, -1.0, 5.0]])

b = np.array([7.0, -10.0, 4.0, 7.0])

initial_guess = np.array([0.0, 0.0, 0.0, 0.0])
omega = 1.5
max_iterations = 1000
tolerance = 1e-6

x, iterations = relax_iteracni(A, b, omega, initial_guess, max_iterations, tolerance)

print("Solution:", x)
print("Number of iterations:", iterations)

import numpy as np

def jakub_iteracni(A, b, x0=None, tol=1e-6, max_iter=1000):
    if x0 is None:
        x0 = np.ones(len(A))
    x = x0
    n = len(A)
    iter_count = 0
    
    for _ in range(max_iter):
        r = b - np.dot(A, x)

        if np.linalg.norm(r, np.inf) < tol:
            return x, iter_count

        new_x = np.zeros(n)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            new_x[i] = (b[i] - s) / A[i, i]
        
        x = new_x
        iter_count += 1

        print("Iteration:", iter_count)
        print("Solution:", x)
        print()

    print("Maximální počet iterací dosažených bez konvergence")
    return x, iter_count


A = np.array([[4, -1, 0, 2],
              [-1, 10, 2, -1],
              [0, 2, 7, -4],
              [2, -1, -1, 5]])

b = np.array([7, -10, 4, 7])
x0 = np.array([0, 0, 0, 0])
tol = 1e-6
max_iter = 1000

x, iterations = jakub_iteracni(A, b, x0, tol, max_iter)
print("Solution:", x)
print("Number of iterations:", iterations)

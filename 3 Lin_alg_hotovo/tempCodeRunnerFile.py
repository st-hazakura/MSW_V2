import numpy as np
import time
import matplotlib.pyplot as plt

#
def Gauss_el_prima(A, b):
    m, n = A.shape  # Získejte rozměry matice koeficientů

    # Proveďeme Gaussovu eliminaci
    iteration_count = 0  # Счетчик итераций
    for k in range(m-1):  # Cyklus přes otočné sloupce
        for i in range(k+1, m):  # Cyklus mezi řádky pod otočným řádkem
            # Vypočítejte násobek pro řádek i
            mult = A[i, k] / A[k, k]
            A[i, k:n] = A[i, k:n] - mult * A[k, k:n]  # Odstranit proměnnou v řádku i
            b[i] = b[i] - mult * b[k]  # Aktualizujeme pravou stranu
            iteration_count += 1  # Увеличиваем счетчик итераций

        # Výpis výsledků po každé iteraci
        x = np.zeros(m)  # Definujeme parametry vektoru řešení
        x[m-1] = b[m-1] / A[m-1, m-1]  # Vypočíteme hodnotu Poslední proměnné
        for i in range(m-2, -1, -1):  # Cyklus přes řádky v opačném pořadí
            x[i] = (b[i] - np.dot(A[i, i+1:m], x[i+1:m])) / A[i, i]  # Vypočítejte hodnotu proměnné v řádku i

        print("Iteration:", iteration_count)
        print("Solution:", x)
        print()

    return x, iteration_count


#
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


#
def jakub_iteracni(A, b, x0=None, tol=1e-6, max_iter=100):
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


#
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


# Создание списка размеров квадратных матриц
matrix_sizes = [2, 3, 4, 5, 6, 7]

# Создание списков для хранения времени выполнения и размеров матриц
jakub_iteracni_times = []
gauss_elimination_times = []
relax_iteracni_times = []
gauss_seidel_times = []

# Вычисление времени выполнения и добавление в списки
for size in matrix_sizes:
    A = np.random.rand(size, size)
    b = np.random.rand(size)
    x0 = np.zeros(size)
    


    # Gauss_el_prima
    start_time = time.time()
    Gauss_el_prima(A, b)
    end_time = time.time()
    gauss_elimination_times.append(end_time - start_time)

    # gauss_se_iteracni
    start_time = time.time()
    gauss_se_iteracni(A, b, x0, max_iterations, tolerance)
    end_time = time.time()
    gauss_seidel_times.append(end_time - start_time)
    
    # jakub_iteracni
    start_time = time.time()
    jakub_iteracni(A, b, x0)
    end_time = time.time()
    jakub_iteracni_times.append(end_time - start_time)

    # relax_iteracni
    start_time = time.time()
    relax_iteracni(A, b, omega, initial_guess, max_iterations, tolerance)
    end_time = time.time()
    relax_iteracni_times.append(end_time - start_time)    
    

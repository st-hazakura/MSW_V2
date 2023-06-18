import numpy as np
import time
import matplotlib.pyplot as plt

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

    return iteration_count


def gauss_se_iteracni(A, b, initial_guess=None, max_iterations=100, tolerance=1e-6):
    x = initial_guess.copy() if initial_guess is not None else np.zeros(len(b))
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
    
    return iterations


def jakub_iteracni(A, b, x0=None, tol=1e-6, max_iter=100):
    if x0 is None:
        x0 = np.ones(len(A))
    x = x0.copy()
    n = len(A)
    iter_count = 0
    
    for _ in range(max_iter):
        r = b - np.dot(A, x)

        if np.linalg.norm(r, np.inf) < tol:
            return iter_count

        new_x = np.zeros(n)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            new_x[i] = (b[i] - s) / A[i, i]
        
        x = new_x
        iter_count += 1

    return iter_count


def relax_iteracni(A, b, omega=1.5, initial_guess=None, max_iterations=100, tolerance=1e-6):
    x = initial_guess.copy() if initial_guess is not None else np.zeros(len(b))
    n = len(b)
    iterations = 0
    
    for k in range(max_iterations):
        r = b - np.dot(A, x)
        
        if np.linalg.norm(r, np.inf) < tolerance:
            return iterations
        
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)
        
        iterations += 1
    
    return iterations


# Создание списка размеров квадратных матриц
matrix_sizes = [50, 100, 200, 300, 400, 500]

# Создание списков для хранения времени выполнения и размеров матриц
jakub_iteracni_times = []
gauss_elimination_times = []
relax_iteracni_times = []
gauss_seidel_times = []

# Вычисление времени выполнения и добавление в списки
for size in matrix_sizes:
    A = np.random.rand(size, size)
    b = np.random.rand(size)
    
    x0 = np.zeros(size)  # Создание нового вектора x0 для текущей размерности

    # Gauss_el_prima
    start_time = time.time()
    Gauss_el_prima(A, b)
    end_time = time.time()
    gauss_elimination_times.append(end_time - start_time)

    # gauss_se_iteracni
    start_time = time.time()
    gauss_se_iteracni(A, b, x0, max_iterations=100, tolerance=1e-6)
    end_time = time.time()
    gauss_seidel_times.append(end_time - start_time)
    
    # jakub_iteracni
    start_time = time.time()
    jakub_iteracni(A, b, x0)
    end_time = time.time()
    jakub_iteracni_times.append(end_time - start_time)

    # relax_iteracni
    start_time = time.time()
    relax_iteracni(A, b, omega=1.5, initial_guess=x0, max_iterations=100, tolerance=1e-6)
    end_time = time.time()
    relax_iteracni_times.append(end_time - start_time)


plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, gauss_elimination_times, label='Gauss Elimination')
plt.plot(matrix_sizes, gauss_seidel_times, label='Gauss-Seidel')
plt.plot(matrix_sizes, jakub_iteracni_times, label='Jakub Iterative')
plt.plot(matrix_sizes, relax_iteracni_times, label='Relax Iterative')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Matrix Size for Linear Equation Solvers')
plt.legend()
plt.grid(True)

#plt.savefig('linear_equation_solver_times2.png', format='png', dpi=300)
plt.show()

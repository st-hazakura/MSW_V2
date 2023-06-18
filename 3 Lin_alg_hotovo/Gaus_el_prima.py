import numpy as np

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


A = np.array([[4, -1, 0, 2],
              [-1, 10, 2, -1],
              [0, 2, 7, -1],
              [2, -1, -1, 5]])

b = np.array([7, -10, 4, 7])

x, iterations = Gauss_el_prima(A, b)
print("Solution:", x)
print("Number of iterations:", iterations)

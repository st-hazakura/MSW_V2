import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Definice funkce
def func(x):
    return np.sin(x)  # Nahraďte tuto funkci jakoukoli jinou funkcí, kterou chcete zkoumat

# Analytické řešení derivace
x_sym = sp.symbols('x')
f_sym = sp.sin(x_sym)
f_prime_sym = sp.diff(f_sym, x_sym)
f_prime = sp.lambdify(x_sym, f_prime_sym)

# Definice intervalu a kroku
start = 0.0
end = 2.0 * np.pi

# Adaptivní krok
def adaptive_difference(x, y, method):
    n = len(x)
    dy_dx = np.zeros(n)

    for i in range(1, n-1):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        h = min(h1, h2)

        if method == 'central':
            dy_dx[i] = (y[i+1] - y[i-1]) / (2*h)
        elif method == 'backward':
            dy_dx[i] = (y[i] - y[i-1]) / h
        elif method == 'forward':
            dy_dx[i] = (y[i+1] - y[i]) / h

    return dy_dx

# Generování hodnot s pevným krokem
step_fixed = 0.1
x_fixed = np.arange(start, end, step_fixed)
y_fixed = func(x_fixed)

# Numerická derivace s adaptivním krokem
x_adaptive = np.linspace(start, end, 1000)  # Více bodů pro hladší křivku
y_adaptive = func(x_adaptive)

dy_dx_central_adaptive = adaptive_difference(x_adaptive, y_adaptive, 'central')
dy_dx_backward_adaptive = adaptive_difference(x_adaptive, y_adaptive, 'backward')
dy_dx_forward_adaptive = adaptive_difference(x_adaptive, y_adaptive, 'forward')

# Výpočet numerických derivací s pevným krokem
dy_dx_central_fixed = adaptive_difference(x_fixed, y_fixed, 'central')
dy_dx_backward_fixed = adaptive_difference(x_fixed, y_fixed, 'backward')
dy_dx_forward_fixed = adaptive_difference(x_fixed, y_fixed, 'forward')

# Výpočet chyb
error_central_adaptive = np.abs(dy_dx_central_adaptive - f_prime(x_adaptive))
error_backward_adaptive = np.abs(dy_dx_backward_adaptive - f_prime(x_adaptive))
error_forward_adaptive = np.abs(dy_dx_forward_adaptive - f_prime(x_adaptive))

error_central_fixed = np.abs(dy_dx_central_fixed - f_prime(x_fixed))
error_backward_fixed = np.abs(dy_dx_backward_fixed - f_prime(x_fixed))
error_forward_fixed = np.abs(dy_dx_forward_fixed - f_prime(x_fixed))


# Vykreslení výsledků
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x_adaptive, f_prime(x_adaptive), label='Analytické řešení')
plt.plot(x_adaptive, dy_dx_central_adaptive, label='Centrální rozdíl (adaptivní)')
plt.plot(x_adaptive, dy_dx_backward_adaptive, label='Zadní rozdíl (adaptivní)')
plt.plot(x_adaptive, dy_dx_forward_adaptive, label='Přední rozdíl (adaptivní)')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.title('Porovnání numerických a analytických derivací (adaptivní krok)')
plt.legend()
plt.grid(True)


plt.subplot(2, 1, 2)
plt.plot(x_fixed, f_prime(x_fixed), label='Analytické řešení')
plt.plot(x_fixed, dy_dx_central_fixed, '--', label='Centrální rozdíl (pevný krok)')
plt.plot(x_fixed, dy_dx_backward_fixed, '--', label='Zadní rozdíl (pevný krok)')
plt.plot(x_fixed, dy_dx_forward_fixed, '--', label='Přední rozdíl (pevný krok)')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.title('Porovnání numerických a analytických derivací (pevný krok)')
plt.legend()
plt.grid(True)
#plt.savefig('numerical_derivatives_adaptive.png')


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x_adaptive, error_central_adaptive, label='Chyba - Centrální rozdíl (adaptivní)')
plt.plot(x_adaptive, error_backward_adaptive, label='Chyba - Zadní rozdíl (adaptivní)')
plt.plot(x_adaptive, error_forward_adaptive, label='Chyba - Přední rozdíl (adaptivní)')
plt.xlabel('x')
plt.ylabel('Absolutní chyba')
plt.title('Chyba numerických derivací (adaptivní krok)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_fixed, error_central_fixed, '--', label='Chyba - Centrální rozdíl (pevný krok)')
plt.plot(x_fixed, error_backward_fixed, '--', label='Chyba - Zadní rozdíl (pevný krok)')
plt.plot(x_fixed, error_forward_fixed, '--', label='Chyba - Přední rozdíl (pevný krok)')
plt.xlabel('x')
plt.ylabel('Absolutní chyba')
plt.title('Chyba numerických derivací (pevný krok)')
plt.legend()
plt.grid(True)

plt.tight_layout()
#plt.savefig('chyba.png')
plt.show()

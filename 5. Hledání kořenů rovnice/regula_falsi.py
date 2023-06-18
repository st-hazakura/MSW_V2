import numpy as np
from sympy import symbols, lambdify, exp, sin

def regula_falsi(func, a, b, epsilon=1e-6, max_iterations=100):
    x = symbols('x')
    f = lambdify(x, func)

    if f(a) * f(b) >= 0:
        raise ValueError("Hodnoty funkce 'a' a 'b' musí mít opačné znaménko.")

    for i in range(max_iterations):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        
        if abs(f(c)) < epsilon:
            return c
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    raise ValueError("Metoda nepřešla požadovaným počtem iterací.")

# Příklad použití

# Polynomická funkce: f(x) = x^3 - 2x^2 - 5x + 6
x = symbols('x')
polynomial_func = x**3 - 2*x**2 - 5*x + 6
root_polynomial = regula_falsi(polynomial_func, -5, 5)
print("Kořen (Polynomická funkce):", root_polynomial)

# Exponenciální/logaritmická funkce: f(x) = e^x - 3
exponential_func = exp(x) - 3
root_exponential = regula_falsi(exponential_func, 0, 2)
print("Kořen (Exponenciální funkce):", root_exponential)

# Harmonická funkce s parametrem: f(x) = sin(x) - a
a = 0.5  # Konstanta 'a'
harmonic_func = sin(x) - a
root_harmonic = regula_falsi(harmonic_func, 0, 2)
print("Kořen (Harmonická funkce):", root_harmonic)

import timeit
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, exp, sin

def regula_falsi(func, a, b, epsilon=1e-6, max_iterations=1000):
    x = symbols('x')
    f = lambdify(x, func)

    if f(a) * f(b) >= 0:
        raise ValueError("Hodnoty funkce 'a' a 'b' musí mít opačné znaménko.")

    start_time = timeit.default_timer()

    for i in range(max_iterations):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))

        if abs(f(c)) < epsilon:
            end_time = timeit.default_timer()
            execution_time = end_time - start_time
            return c, execution_time

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    raise ValueError("Metoda nepřešla požadovaným počtem iterací.")


def bisection_method(func, a, b, epsilon):
    x = symbols('x')
    f = lambdify(x, func)

    if f(a) * f(b) >= 0:
        raise ValueError("Hodnoty funkce 'a' a 'b' musí mít opačné znaménko.")

    start_time = timeit.default_timer()

    while abs(b - a) > epsilon:
        c = (a + b) / 2
        if f(c) == 0:
            end_time = timeit.default_timer()
            execution_time = end_time - start_time
            return c, execution_time
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    return (a + b) / 2, execution_time



# Polynomická funkce: f(x) = x^3 - 2x - 5
x = symbols('x')
polynomial_func = x**3 - 2*x - 5
a_polynomial = -10.0
b_polynomial = 10.0
epsilon_polynomial = 1e-6

x_sol = np.roots([1, 0, -2, -5])

root_polynomial_rf, time_polynomial_rf = regula_falsi(polynomial_func, a_polynomial, b_polynomial, epsilon_polynomial, max_iterations=1000)
root_polynomial_bs, time_polynomial_bs = bisection_method(polynomial_func, a_polynomial, b_polynomial, epsilon_polynomial)

polynomial_error_rf = abs(root_polynomial_rf - x_sol).min()
polynomial_error_bs = abs(root_polynomial_bs - x_sol).min()

print("Polynomická funkce:")
print("Metoda regula falsi: Kořen =", root_polynomial_rf, "Časová náročnost =", time_polynomial_rf, "s")
print("Metoda bisekce: Kořen =", root_polynomial_bs, "Časová náročnost =", time_polynomial_bs, "s")
print("Chyba regula falsi:", polynomial_error_rf)
print("Chyba bisekce:", polynomial_error_bs)
comparison = "Bisection" if time_polynomial_rf > time_polynomial_bs else "Regula Falsi"
print(f"Metoda {comparison} rychlejší\n\n")




# Exponenciální/logaritmická funkce: f(x) = e^x - 3
exponential_func = exp(x) - 3
a_exponential = -5.0
b_exponential = 5.0
epsilon_exponential = 1e-6

x_sol_exponential = np.log(3)

root_exponential_rf, time_exponential_rf = regula_falsi(exponential_func, a_exponential, b_exponential, epsilon_exponential, max_iterations=1000)
root_exponential_bs, time_exponential_bs = bisection_method(exponential_func, a_exponential, b_exponential, epsilon_exponential)

exponential_error_rf = abs(root_exponential_rf - x_sol_exponential)
exponential_error_bs = abs(root_exponential_bs - x_sol_exponential)

print("Exponenciální/logaritmická funkce:")
print("Metoda regula falsi: Kořen =", root_exponential_rf, "Časová náročnost =", time_exponential_rf, "s")
print("Metoda bisekce: Kořen =", root_exponential_bs, "Časová náročnost =", time_exponential_bs, "s")
print("Chyba regula falsi:", exponential_error_rf)
print("Chyba bisekce:", exponential_error_bs)

comparison = "Bisection" if time_exponential_rf > time_exponential_bs else "Regula Falsi"
print(f"Metoda {comparison} rychlejší\n\n")




# Harmonická funkce s parametrem: f(x) = sin(x) + k
k = 0.5
harmonic_func = sin(x) + k
a_harmonic = -5.0
b_harmonic = 5.0
epsilon_harmonic = 1e-6

x_sol_harmonic = np.arcsin(-k)

root_harmonic_rf, time_harmonic_rf = regula_falsi(harmonic_func, a_harmonic, b_harmonic, epsilon_harmonic, max_iterations=1000)
root_harmonic_bs, time_harmonic_bs = bisection_method(harmonic_func, a_harmonic, b_harmonic, epsilon_harmonic)

harmonic_error_rf = abs(root_harmonic_rf - x_sol_harmonic)
harmonic_error_bs = abs(root_harmonic_bs - x_sol_harmonic)

print("Harmonická funkce:")
print("Metoda regula falsi: Kořen =", root_harmonic_rf, "Časová náročnost =", time_harmonic_rf, "s")
print("Metoda bisekce: Kořen =", root_harmonic_bs, "Časová náročnost =", time_harmonic_bs, "s")
print("Chyba regula falsi:", harmonic_error_rf)
print("Chyba bisekce:", harmonic_error_bs)

comparison = "Bisection" if time_harmonic_rf > time_harmonic_bs else "Regula Falsi"
print(f"Metoda {comparison} rychlejší\n\n")




# Vykreslení grafů
x_vals = np.linspace(-10, 10 , 1000)
polynomial_func_lambda = lambdify(x, polynomial_func)
exponential_func_lambda = lambdify(x, exponential_func)
harmonic_func_lambda = lambdify(x, harmonic_func)

plt.figure(figsize=(15, 6))
plt.plot(x_vals, polynomial_func_lambda(x_vals), label='Polynomial Function')
plt.plot(x_vals, exponential_func_lambda(x_vals), label='Exponential/Logarithmic Function')
plt.plot(x_vals, harmonic_func_lambda(x_vals), label='Harmonic Function')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graf funkcí')

# Značky nalezených kořenů
plt.scatter(root_polynomial_rf, polynomial_func_lambda(root_polynomial_rf), color='red', label='Root (Regula Falsi - Polynomial)')
plt.scatter(root_polynomial_bs, polynomial_func_lambda(root_polynomial_bs), color='green', label='Root (Bisection - Polynomial)')
plt.scatter(root_exponential_rf, exponential_func_lambda(root_exponential_rf), color='blue', label='Root (Regula Falsi - Exponential)')
plt.scatter(root_exponential_bs, exponential_func_lambda(root_exponential_bs), color='purple', label='Root (Bisection - Exponential)')
plt.scatter(root_harmonic_rf, harmonic_func_lambda(root_harmonic_rf), color='orange', label='Root (Regula Falsi - Harmonic)')
plt.scatter(root_harmonic_bs, harmonic_func_lambda(root_harmonic_bs), color='cyan', label='Root (Bisection - Harmonic)')

plt.legend()
plt.grid(True)
#plt.savefig('Hledání kořenů rovnice1.png', format='png', dpi=300)
plt.show()



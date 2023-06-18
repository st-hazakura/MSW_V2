from sympy import symbols, lambdify, exp, sin

def bisection_method(func, a, b, epsilon):
    x = symbols('x')
    f = lambdify(x, func)
    

    
    while abs(b - a) > epsilon:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2


# Polynomická funkce: f(x) = x^3 - 2x^2 - 5x + 6
x = symbols('x')
polynomial_func = x**3 - 2*x**2 - 5*x + 6
a_polynomial = 1.0
b_polynomial = 3.0
epsilon_polynomial = 1e-6

root_polynomial = bisection_method(polynomial_func, a_polynomial, b_polynomial, epsilon_polynomial)
if root_polynomial is not None:
    print("Kořen (Polynomická funkce):", root_polynomial)
else:
    print("Nepodařilo se najít kořen (Polynomická funkce).")

# Exponenciální/logaritmická funkce: f(x) = e^x - 3
exponential_func = exp(x) - 3
a_exponential = 0.0
b_exponential = 2.0
epsilon_exponential = 1e-6

root_exponential = bisection_method(exponential_func, a_exponential, b_exponential, epsilon_exponential)
if root_exponential is not None:
    print("Kořen (Exponenciální funkce):", root_exponential)
else:
    print("Nepodařilo se najít kořen (Exponenciální funkce).")

# Harmonická funkce s parametrem: f(x) = sin(x) - a
a = 0.5  # Konstanta 'a'
harmonic_func = sin(x) - a
a_harmonic = 0.0
b_harmonic = 2.0
epsilon_harmonic = 1e-6

root_harmonic = bisection_method(harmonic_func, a_harmonic, b_harmonic, epsilon_harmonic)
if root_harmonic is not None:
    print("Kořen (Harmonická funkce):", root_harmonic)
else:
    print("Nepodařilo se najít kořen (Harmonická funkce).")

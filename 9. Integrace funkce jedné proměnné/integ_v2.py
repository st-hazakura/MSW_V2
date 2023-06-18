import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Definice funkcí
x = sp.symbols('x')
polynomicka_funkce = x**3 - 2*x - 5
exp_log_funkce = sp.exp(x**2) - 3
harmonicka_funkce = sp.sin(x) + 1

# Analytické řešení
integral_analytic_polynomicka = sp.integrate(polynomicka_funkce, (x, 0, 1))
integral_analytic_exp_log = sp.integrate(exp_log_funkce, (x, 0, 1))
integral_analytic_harmonicka = sp.integrate(harmonicka_funkce, (x, 0, 1))

# Lichoběžníková metoda
def lichobeznikova_metoda(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    I = h/2 * (y[0] + 2*np.sum(y[1:n]) + y[n])
    return I

f_polynomicka = sp.lambdify(x, polynomicka_funkce)
integral_lichobeznik_polynomicka = lichobeznikova_metoda(f_polynomicka, 0, 1, 1000)

f_exp_log = sp.lambdify(x, exp_log_funkce)
integral_lichobeznik_exp_log = lichobeznikova_metoda(f_exp_log, 0, 1, 1000)

f_harmonicka = sp.lambdify(x, harmonicka_funkce)
integral_lichobeznik_harmonicka = lichobeznikova_metoda(f_harmonicka, 0, 1, 1000)

# Rombergova metoda
def rombergova_metoda(f, a, b, N):
    R = np.zeros((N, N))
    h = b - a
    R[0, 0] = h/2 * (f(a) + f(b))
    
    for n in range(1, N):
        h = h/2
        sum = 0
        for i in range(1, 2**n, 2):
            sum += f(a + i*h)
        R[n, 0] = 1/2 * R[n-1, 0] + h*sum
        
        for m in range(1, n+1):
            R[n, m] = (4**m * R[n, m-1] - R[n-1, m-1]) / (4**m - 1)
    
    return R[N-1, N-1]

integral_romberg_polynomicka = rombergova_metoda(f_polynomicka, 0, 1, 3)
integral_romberg_exp_log = rombergova_metoda(f_exp_log, 0, 1, 2)
integral_romberg_harmonicka = rombergova_metoda(f_harmonicka, 0, 1, 1)

# Simpsonova metoda
def simpsonova_metoda(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    S = y[0] + y[n]
    S += 4*np.sum(y[1:n:2])
    S += 2*np.sum(y[2:n-1:2])
    integral = h/3 * S
    return integral

integral_simpson_polynomicka = simpsonova_metoda(f_polynomicka, 0, 1, 100)
integral_simpson_exp_log = simpsonova_metoda(f_exp_log, 0, 1, 100)
integral_simpson_harmonicka = simpsonova_metoda(f_harmonicka, 0, 1, 100)

# Výpis výsledků
print("Analytické řešení:")
print("Polynomická funkce:", integral_analytic_polynomicka)
print("Exponenciální/logaritmická funkce:", integral_analytic_exp_log)
print("Harmonická funkce:", integral_analytic_harmonicka)
print()
print("Lichoběžníková metoda:")
print("Polynomická funkce:", integral_lichobeznik_polynomicka)
print("Exponenciální/logaritmická funkce:", integral_lichobeznik_exp_log)
print("Harmonická funkce:", integral_lichobeznik_harmonicka)
print()
print("Rombergova metoda:")
print("Polynomická funkce:", integral_romberg_polynomicka)
print("Exponenciální/logaritmická funkce:", integral_romberg_exp_log)
print("Harmonická funkce:", integral_romberg_harmonicka)
print()
print("Simpsonova metoda:")
print("Polynomická funkce:", integral_simpson_polynomicka)
print("Exponenciální/logaritmická funkce:", integral_simpson_exp_log)
print("Harmonická funkce:", integral_simpson_harmonicka)

# Vykreslení grafů
x_vals = np.linspace(0, 1, 100)
y_polynomicka = f_polynomicka(x_vals)
y_exp_log = f_exp_log(x_vals)
y_harmonicka = f_harmonicka(x_vals)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x_vals, y_polynomicka)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Polynomická funkce')

plt.subplot(1, 3, 2)
plt.plot(x_vals, y_exp_log)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Exponenciální/logaritmická funkce')

plt.subplot(1, 3, 3)
plt.plot(x_vals, y_harmonicka)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Harmonická funkce')

plt.tight_layout()
#plt.savefig('fce.png')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit(x, a, b, c): return a - b*np.exp(-c*x)


def log_fit(x, a, b, c): return a - b / x**c

# def chi_square(N, diff, err):


# dimensionality data:
data = np.array([[0.002, 0.59],
                [0.004, 0.62],
                [0.006, 0.66],
                [0.008, 0.69],
                [0.01, 0.70],
                [0.02, 0.77],
                [0.04, 0.82],
                [0.06, 0.87],
                [0.08, 0.87],
                [0.1, 0.88],
                [0.2, 0.89],
                [0.3, 0.89],
                [0.4, 0.89],
                [0.5, 0.89],
                [0.6, 0.90],
                 [0.7, 0.90],
                 [0.8, 0.90],
                 [0.9, 0.90],
                 [1, 0.90]])

data_new = np.array([0.002, 0.500, 0.003],
                    [0.004, 0.500, 0.003],
                    [0.006, 0.505, 0.003],
                    [0.008, 0.509, 0.004],
                    [0.01, 0.514, 0.004],
                    [0.05, 0.616, 0.005],
                    [1, 0.905, 0.013])

fig, ax = plt.subplots(figsize=(7, 5), dpi=400)
ax.errorbar(data[:, 0], data[:, 1], 0.01, fmt='kx', label="Data")
ax.set_xlabel(r"Mass Ratio $\gamma$")
# ax.errorbar(np.log(data[:, 0]), data[:, 1], 0.01, fmt='kx')
# ax.set_xlabel(r"Logarithm of Mass Ratio ln($\gamma$)")
ax.set_ylabel(r"Exponent of GL Fit")

popt, pcov = curve_fit(fit, data[:, 0], data[:, 1], sigma=[
    0.01 for i in range(len(data[:, 1]))], maxfev=2000)
params = popt
error = np.sqrt(np.diag(pcov))

params = np.array([0.89055188,  0.31336491, 45.64309261])
error = np.array([3.65816903e-03, 9.28428753e-03, 3.76354433e+00])
print(params, error)

x = np.linspace(0, data[-1, 0], 100)
ax.plot(x, fit(x, *params), 'r', label="Fit")
#ax.plot(x, log_fit(x, *params))
plt.legend(fontsize=8, loc="lower right")

diff = data[:, 1] - fit(data[:, 0], *params)
chi_square = 0
for i in range(len(data[:, 0])):
    chi_square += (diff[i] / 0.01)**2

print(chi_square / len(data[:, 0]))

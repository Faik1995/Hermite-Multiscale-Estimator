import numpy as np
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from scipy.special import hermite

mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"

np.random.seed(2025)

#%% Data

epsilon = 0.1

sigma = 1
V = lambda x: x**4/4 - x**2/2
dV = lambda x: x**3 - x
p = lambda y: np.cos(y)
dp = lambda y: - np.sin(y)
L = 2*np.pi

times = [50, 500, 5000]
coeff = [4, 16, 64]

N = coeff[-1]

X0 = 0
T = times[-1]
dt = epsilon**3
I = round(T/dt)

dW = np.random.normal(0, np.sqrt(dt), (I,))
X = np.empty(I+1)
X[0] = X0

R = 4
points = np.linspace(-R, R, 1001)

#%% Solution SDE

for i in tqdm(range(I)):
    X[i+1] = X[i] - dV(X[i])*dt - (1/epsilon)*dp(X[i]/epsilon)*dt + np.sqrt(2*sigma)*dW[i]

#%% Exact invariant measures

rho_ = lambda x: np.exp(-(1/sigma)*V(x))
Z = quad(rho_, -np.inf, 0)[0] + quad(rho_, 0, +np.inf)[0]
rho = lambda x: rho_(x)/Z

rho_e_ = lambda x: np.exp(-(1/sigma)*(V(x) + p(x/epsilon)))
Z_e = quad(rho_e_, -np.inf, 0)[0] + quad(rho_e_, 0, +np.inf)[0]
rho_e = lambda x: rho_e_(x)/Z_e

out = rho(points)
out_e = rho_e(points)

# create data directory if missing
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

np.savetxt("data/test_NT_homogenized.txt", out)
np.savetxt("data/test_NT_multiscale_e" + str(epsilon) + ".txt", out_e)

#%% Estimation invariant measure

psi = lambda n, x: hermite(n)(x)*np.exp(-x**2/2)/np.sqrt(np.sqrt(np.pi)*2**n*math.factorial(n))

for i in range(len(times)):
    data = X[:round(times[i]/dt)]
    
    alpha = np.empty(N)
    for n in range(N):
        alpha[n] =  np.mean(psi(n, data))
    
    for j in range(len(coeff)):
        print(str(1 + 3*i + j) + '/' + str(9))
        rho_hat = lambda x: np.sum(np.squeeze(np.array([alpha[n]*psi(n,x) for n in range(coeff[j])])), 0)
        out_hat = rho_hat(points)
        
        np.savetxt("data/test_NT_N" + str(coeff[j]) + "_e" + str(epsilon) + "_T" + str(times[i]) + ".txt", out_hat)

        plt.figure()
        plt.grid(alpha=0.3)
        plt.plot(points, out_e, label=rf'$\mu_\varepsilon$', color="#e7bf6d", linestyle = '-', linewidth = 2)
        plt.plot(points, out, label=rf'$\mu$', color='#851e09', linestyle = '--', linewidth = 2)
        plt.plot(points, out_hat, label=r'$\widehat \mu^\varepsilon_{N, T}$', color='#3754cc', linewidth = 2)
        plt.title(rf'$T = {times[i]}, N = {coeff[j]}$', fontsize=24)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)  
        plt.savefig("figures/varyNeT_N" + str(coeff[j]) + "_e" + str(epsilon) + "_T" + str(times[i]) + ".pdf", bbox_inches="tight")
        plt.show()
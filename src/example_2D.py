import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import dblquad
from scipy.special import hermite

mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"

np.random.seed(2025)

#%% Data

epsilon = 0.1
sigma = 1.5

p1 = lambda y: np.sin(y)
dp1 = lambda y: np.cos(y)
L1 = 2*np.pi
p2 = lambda y: np.sin(y)**2
dp2 = lambda y: 2*np.sin(y)*np.cos(y)
L2 = 2*np.pi
p = lambda y: p1(y[0]) + p2(y[1])
gradp = lambda y: np.array([dp1(y[0]), dp2(y[1])])

V = lambda x: (x[0]**4 + x[1]**4)/4 - (x[0]**2 + x[1]**2)/2
gradV = lambda x: np.array([x[0]**3 - x[0], x[1]**3 - x[1]])

T = 2000
dt = epsilon**3
I = round(T/dt)
X_initial = np.array([0,0])

#%% Solution SDE

dW = np.random.normal(0, np.sqrt(dt), (2,I))
X = np.empty((2,I+1))
X[:,0] = X_initial

for i in tqdm(range(I)):
    X[:,i+1] = X[:,i] - gradV(X[:,i])*dt - (1/epsilon)*gradp(X[:,i]/epsilon)*dt + np.sqrt(2*sigma)*dW[:,i]

#%% Exact invariant measures

rho_ = lambda x0, x1: np.exp(-(1/sigma)*V(np.array([x0, x1])))
Z = dblquad(rho_, -np.inf, 0, 0, +np.inf)[0] + dblquad(rho_, 0, +np.inf, 0, +np.inf)[0] + dblquad(rho_, -np.inf, 0, -np.inf, 0)[0] + dblquad(rho_, 0, +np.inf, -np.inf, 0)[0]
rho = lambda x: rho_(x[0], x[1])/Z

rho_e_ = lambda x0, x1: np.exp(-(1/sigma)*(V(np.array([x0, x1])) + p(np.array([x0, x1])/epsilon)))
Z_e = dblquad(rho_e_, -np.inf, 0, 0, +np.inf)[0] + dblquad(rho_e_, 0, +np.inf, 0, +np.inf)[0] + dblquad(rho_e_, -np.inf, 0, -np.inf, 0)[0] + dblquad(rho_e_, 0, +np.inf, -np.inf, 0)[0]
rho_e = lambda x: rho_e_(x[0], x[1])/Z_e

#%% Estimation invariant measure

N_max = 25

psi = lambda m, n, x: (hermite(m)(x[0])*np.exp(-x[0]**2/2)/np.sqrt(np.sqrt(np.pi)*2**m*math.factorial(m)))*(hermite(n)(x[1])*np.exp(-x[1]**2/2)/np.sqrt(np.sqrt(np.pi)*2**n*math.factorial(n)))

alpha = np.empty((N_max, N_max))
for m in tqdm(range(N_max)):
    for n in range(N_max):
        alpha[m,n] =  np.mean(psi(m, n, X))

#%% Results

N = 16
rho_hat = lambda x: np.sum(np.squeeze(np.array([alpha[m,n]*psi(m,n,x) for m in range(N) for n in range(N)])), 0)

R = 3
x0 = np.linspace(-R, R, 101)
x1 = np.linspace(-R, R, 101)
X0, X1 = np.meshgrid(x0, x1)
points = np.stack([X0.ravel(), X1.ravel()], axis=1)

out = np.array([rho(a) for a in tqdm(points)])
out = out.reshape(X0.shape)
out_e = np.array([rho_e(a) for a in tqdm(points)])
out_e = out_e.reshape(X0.shape)
out_hat = np.array([rho_hat(a) for a in tqdm(points)])
out_hat = out_hat.reshape(X0.shape)

np.savetxt("data/example_2D_multiscale_e" + str(epsilon) + "_T" + str(T) + ".txt", out_e)
np.savetxt("data/example_2D_homogenized_T" + str(T) + ".txt", out)
np.savetxt("data/example_2D_estimated_N" + str(N) + "_e" + str(epsilon) + "_T" + str(T) + ".txt", out_hat)

plt.figure()
c0 = plt.contourf(X0, X1, out_e, levels=50, cmap='viridis')
cb0 = plt.colorbar(c0, ticks = np.linspace(0, 0.2, 11), format="%.2f")
cb0.ax.tick_params(labelsize=14)
plt.title(rf'$\mu_\varepsilon$', fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figures/example2D_multiscale_e0.1.pdf", bbox_inches="tight")
plt.show()

vmin = min(out.min(), out_hat.min())
vmax = max(out.max(), out_hat.max())

levels = np.linspace(vmin, vmax, 51)      # 51 filled intervals
ticks = np.linspace(0, 0.1, 11)     # same tick positions on both bars

plt.figure()
c1 = plt.contourf(X0, X1, out, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
cb1 = plt.colorbar(c1, ticks=ticks, format="%.2f")
cb1.ax.tick_params(labelsize=14)
plt.title(rf'$\mu$', fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figures/example2D_homogenized.pdf", bbox_inches="tight")
plt.show()

plt.figure()
c2 = plt.contourf(X0, X1, out_hat, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
cb2 = plt.colorbar(c2, ticks=ticks, format="%.2f")
cb2.ax.tick_params(labelsize=14)
plt.title(r'$\widehat \mu^\varepsilon_{N, T}$', fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figures/example2D_N16_e0.1_T2000.pdf", bbox_inches="tight")
plt.show()
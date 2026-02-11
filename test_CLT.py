import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from scipy.special import hermite
from scipy.stats import chi2, norm

np.random.seed(2026)

#%% Poisson problem

def solve_Poisson(R, h, c, rho, psi):
    
    mean_psi = quad(lambda x: psi(x)*rho(x), -np.inf, 0)[0] + quad(lambda x: psi(x)*rho(x), 0, +np.inf)[0]
    psi_bar = lambda x: psi(x) - mean_psi
    
    xx = np.linspace(-R, +R, int(np.round(4*R/h+1)))
    rho_xx = rho(xx)
    psi_bar_xx = psi_bar(xx)
    
    a0 = (    np.concatenate(([0, 0], rho_xx[2:-2:2]))
          + 4*np.concatenate(([0], rho_xx[1:-1:2]))
          + 2*np.concatenate(([0], rho_xx[2:-2:2], [0]))
          + 4*np.concatenate((rho_xx[1:-1:2], [0]))
          +   np.concatenate((rho_xx[2:-2:2], [0, 0])))
    a1 = (    np.concatenate((rho_xx[2:-2:2], [0]))
          + 4*rho_xx[1:-1:2]
          +   np.concatenate(([0], rho_xx[2:-2:2])))
    A = (c/(6*h))*(np.diag(a0, 0) - np.diag(a1, +1) - np.diag(a1, -1))
    
    b0 = (  np.concatenate(([0], psi_bar_xx[1:-1:2]*rho_xx[1:-1:2]))
         + psi_bar_xx[::2]*rho_xx[::2]
         + np.concatenate((psi_bar_xx[1:-1:2]*rho_xx[1:-1:2], [0])))
    b = (h/3)*np.array(b0)
    
    m0 = (  np.concatenate(([0], rho_xx[1:-1:2]))
         + rho_xx[::2]
         + np.concatenate((rho_xx[1:-1:2], [0])))
    m = (h/3)*np.array(m0)
    
    n = A.shape[0]
    matrix = np.zeros((n + 1, n + 1))
    matrix[:n,:n] = A
    matrix[:n,n] = m
    matrix[n,:n] = m
    rhs = np.zeros(n + 1)
    rhs[:n] = b
    
    sol = np.linalg.solve(matrix, rhs)
    phi = sol[:-1]
    x = xx[::2]
    
    return x, phi, A

#%% Data

epsilon = 0.1

sigma = 1
V = lambda x: x**4/4 - x**2/2
dV = lambda x: x**3 - x
p = lambda y: np.cos(y)
dp = lambda y: - np.sin(y)
L = 2*np.pi

N = 16

X0 = 0
T = 1000
dt = epsilon**3
I = round(T/dt)

dW = np.random.normal(0, np.sqrt(dt), (I,))
X = np.empty(I+1)
X[0] = X0

R = 5
h = 0.01
points = np.linspace(-R, +R, int(np.round(2*R/h+1)))

M = 50
S = 1000

strata = 100

#%% Homogenization

Pi_plus = quad(lambda x: np.exp(+p(x)/(sigma**2)), 0, L)[0]
Pi_minus = quad(lambda x: np.exp(-p(x)/(sigma**2)), 0, L)[0]
K = (L**2)/(Pi_plus*Pi_minus)
Sigma = K*(sigma**2)

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

#%% Estimation invariant measure and exact coefficients

psi = lambda n, x: hermite(n)(x)*np.exp(-x**2/2)/np.sqrt(np.sqrt(np.pi)*2**n*math.factorial(n))

alpha = np.empty(N)
for n in tqdm(range(N)):
    alpha[n] =  np.mean(psi(n, X))
    
rho_hat = lambda x: np.sum(np.squeeze(np.array([alpha[n]*psi(n,x) for n in range(N)])), 0)
out_hat = rho_hat(points)

alpha_ex = np.empty(M)
for m in tqdm(range(M)):
    alpha_ex[m] = quad(lambda x: psi(m,x)*rho(x), -np.inf, 0)[0] + quad(lambda x: psi(m,x)*rho(x), 0, +np.inf)[0]

#%% Limit

phis = np.empty((points.shape[0], M))
for m in tqdm(range(M)):
    x, phis[:,m], A = solve_Poisson(R, h, Sigma, rho, lambda x: psi(m, x))
    
tau = np.zeros((M, M))
for m1 in range(M):
    for m2 in range(m1, M):
        tau[m1,m2] = 2*Sigma*np.dot(phis[:,m1], np.dot(A, phis[:,m2]))
tau = (tau + tau.T)/2
        
csi = np.random.multivariate_normal(np.zeros(M), tau, S)

samples = np.empty((points.shape[0], S))
for s in tqdm(range(S)):
    G = lambda x: np.sum(np.squeeze(np.array([csi[s,m]*psi(m,x) for m in range(M)])), 0)
    samples[:,s] = G(points)
standard_deviation = np.std(samples, axis=1)

#%% Plot

plt.figure()
for s in range(S):
    plt.plot(points, rho(points) + samples[:,s]/np.sqrt(T), color='lightblue', linewidth=1)
plt.plot(points, rho(points), label='Homogenized')
plt.plot(points, rho_e(points), label='Multiscale')
plt.plot(points, rho_hat(points), label='Estimated')
plt.legend()
plt.show()

plt.figure()
for s in range(strata):
    plt.fill_between(points, rho(points) + 3*standard_deviation*(s/strata)/np.sqrt(T), rho(points) + 3*standard_deviation*((s+1)/strata)/np.sqrt(T), color='lightblue', alpha=(1-s/strata))
    plt.fill_between(points, rho(points) - 3*standard_deviation*((s+1)/strata)/np.sqrt(T), rho(points) - 3*standard_deviation*(s/strata)/np.sqrt(T), color='lightblue', alpha=(1-s/strata))
plt.plot(points, rho(points), label='Homogenized')
plt.plot(points, rho_e(points), label='Multiscale')
plt.plot(points, rho_hat(points), label='Estimated')
plt.legend()
plt.show()

plt.figure()
for s in range(S):
    plt.plot(points, samples[:,s], color='lightblue', linewidth=1)
plt.plot(points, np.sqrt(T)*(rho_hat(points) - rho(points)), label='CLT')
plt.legend()
plt.show()

plt.figure()
for s in range(strata):
    plt.fill_between(points, +3*standard_deviation*(s/strata), +3*standard_deviation*((s+1)/strata), color='lightblue', alpha=(1-s/strata))
    plt.fill_between(points, -3*standard_deviation*((s+1)/strata), -3*standard_deviation*(s/strata), color='lightblue', alpha=(1-s/strata))
plt.plot(points, np.sqrt(T)*(rho_hat(points) - rho(points)), label='CLT')
plt.show()

#%% Test

alpha_completed = np.concatenate((alpha, np.zeros(M-N)))
coeff = np.sqrt(T)*(alpha_completed - alpha_ex)
L = np.linalg.cholesky(tau)
z = np.linalg.solve(L, coeff)
Q = np.sum(z**2)
p_value = 1 - chi2.cdf(Q, df=M)

grid = np.linspace(np.min(z), np.max(z), 1000)
plt.figure()
plt.hist(z, density=True)
plt.plot(grid, norm.pdf(grid, 0, 1))
plt.show()
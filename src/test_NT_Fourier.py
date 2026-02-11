import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import hermite

np.random.seed(2025)

#%% Data

epsilon = 0.075

sigma = 1
V = lambda x: x**4/4 - x**2/2
dV = lambda x: x**3 - x
p = lambda y: np.cos(y)
dp = lambda y: - np.sin(y)
L = 2*np.pi

coeff = [30, 60, 90]
N = coeff[-1]

X0 = 0
T = 1000
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

#%% Fourier transform

psi = lambda n, x: hermite(n)(x)*np.exp(-x**2/2)/np.sqrt(np.sqrt(np.pi)*2**n*math.factorial(n))
    
alpha = np.empty(N)
for n in tqdm(range(N)):
    alpha[n] =  np.mean(psi(n, X))

for j in range(len(coeff)):
    rho_hat_Fourier = lambda csi: np.sqrt(2*np.pi)*np.sum(np.squeeze(np.array([alpha[n]*((-1j)**n)*psi(n,2*np.pi*csi) for n in range(coeff[j])])), 0)
    out_hat_Fourier = np.abs(rho_hat_Fourier(points))
    
    np.savetxt("data/test_NT_Fourier_N" + str(coeff[j]) + "_e" + str(epsilon) + "_T" + str(T) + ".txt", out_hat_Fourier)
    
    plt.figure()
    plt.plot(points, out_hat_Fourier, label='Fourier')
    plt.title('e = ' + str(epsilon) + ', N = ' + str(coeff[j]))
    plt.show()
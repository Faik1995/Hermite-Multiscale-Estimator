import numpy as np
from numba import njit, prange
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import eval_hermite, factorial
from scipy.integrate import quad
from scipy.stats import norm

# for font consistency with latex classes
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"

# FEniCSx specific packages
from mpi4py import MPI              # for parallelism
import dolfinx.fem.petsc            # assembly functions that turn ufl symbolic forms into actual matrices and vectors, creating meshes, dolfinx.fem function spaces, variational forms, etc.
from petsc4py import PETSc          # linear algebra and solver library underneath FEniCSx
import ufl                          # symbolic language used to write variational forms


# defines the general Poisson equation setting and calculates the FEM solution of our singular Poisson equation on the whole line with natural boundary conditions
class Poisson1D:
    def __init__(self, potential, periodic_oscilation, period, sigma, epsilon, domain_cutoff, mesh_number):

        # homogenized diffusion coefficient
        Pi_plus = quad(lambda x: np.exp(+periodic_oscilation(x)/(sigma**2)), 0, period)[0]
        Pi_minus = quad(lambda x: np.exp(-periodic_oscilation(x)/(sigma**2)), 0, period)[0]
        K = (period**2)/(Pi_plus*Pi_minus)
        self.homog_diffusion = np.sqrt(2*sigma*K)

        # FEM parameters
        R = domain_cutoff
        N = mesh_number

        # mesh, finite element space and domain definitions
        self.domain = dolfinx.mesh.create_interval(MPI.COMM_WORLD, N, [-R, R])             # define mesh of interval [-R, R] with N nodes
        self.V_h = dolfinx.fem.functionspace(self.domain, ('Lagrange', 1))                                                      # define finite element space of piecewise linear functions
        x = ufl.SpatialCoordinate(self.domain)                                                                               # spatial coordinate
        self.dx = ufl.dx(domain=self.domain)                                                                                      # Lebesgue integration measure
        self.mesh = self.domain.geometry.x[:,0]                                                 # mesh for different purposes: integration, plotting etc.

        # Interpolate the large-scale potential into FEM space so we can use it in the FEniCXs setting of FEM and UFL language
        # Vfun is basically the approximation of V in the finite element space that we defined before
        V_fem = dolfinx.fem.Function(self.V_h)
        V_fem.interpolate(lambda x: potential(x[0]))
        V_potential = V_fem

        # Same for the invariant density, first unnormalized, then we normalize
        mu_fem = dolfinx.fem.Function(self.V_h)
        mu_fem.interpolate(lambda x: np.exp(-2 * K * potential(x[0]) / self.homog_diffusion**2))
        Z = quad(lambda x: np.exp(-2 * K * potential(x) / self.homog_diffusion**2), -np.inf, +np.inf)[0]         # normalization constant

        mu_fem.x.array[:] = mu_fem.x.array[:] / Z                                       # actually normalize the values
        mu_fem.x.scatter_forward()                                                      # only for parallelism, synchronize the solution data across MPI processes
        self.mu = mu_fem            

        # Same for the multiscale invariant density, first unnormalized, then we normalize in the FEniCXs way
        mu_eps_fem = dolfinx.fem.Function(self.V_h)
        mu_eps_fem.interpolate(lambda x: np.exp(-potential(x[0]) / sigma - periodic_oscilation(x[0] / epsilon) / sigma ))
        Z_eps = quad(lambda x: np.exp(-potential(x) / sigma - periodic_oscilation(x / epsilon) / sigma ), -np.inf, +np.inf)[0]

        mu_eps_fem.x.array[:] = mu_eps_fem.x.array[:] / Z_eps
        mu_eps_fem.x.scatter_forward()
        self.mu_eps = mu_eps_fem  

        # trial and test functions of variational formulation
        u = ufl.TrialFunction(self.V_h)                                                                # placeholder for unknown solution in V
        self.v = ufl.TestFunction(self.V_h)                                                            # placeholder for an arbitrary test function in V     

        # Bilinear form of variational formulation
        a = self.homog_diffusion**2/2*ufl.inner(self.mu*ufl.grad(u), ufl.grad(self.v))*self.dx                            # ufl expression, i.e., symbolic finite element expression
        a_form = dolfinx.fem.form(a)                                                                    # again, this wraps the symbolic UFL forms into DOLFINx forms that can be assembled

        # boundary condition
        bcs = []                                                                                # weak form uses only the natural boundary conditions, that is, zero-flux condition mu*Phi' = 0, or, in other words, nothing important leaves the boundary and the weak formulation already encodes the boundary condition

        # stiffness matrix
        A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)                                  # assemble the stiffness matrix
        A.assemble()  

        # nullspace definition
        nullspace = PETSc.NullSpace().create(constant=True, comm=self.domain.comm)                   # since our Poisson problem is singular (constants are in the kernel of the generator and the stiffness matrix), we must specify this nullspace of constants
        A.setNullSpace(nullspace)

        # Solver configuration and definition
        petsc_options = {                                                                       # configuring a direct solver (MUMPS)
            "ksp_error_if_not_converged": True,                                                 # give error if not converging
            "ksp_type": "preonly",                                                              # precondition only once since we use direct solver through factorization
            "pc_type": "lu",                                                                    # use LU factorization 
            "pc_factor_mat_solver_type": "mumps",                                               # MUMPS is a parallel sparse direct solver that does the LU factorization
            "ksp_monitor": None,                                                                # solver progress information is turned off
        }

        ksp = PETSc.KSP().create(self.domain.comm)                                                   # creating the Krylov subspace solver
        ksp.setOptionsPrefix("singular_direct")                                                 # giving own namespace to the solver, so the options only apply to our specified solver
        opts = PETSc.Options()                                                                  # PETSc options database
        opts.prefixPush(ksp.getOptionsPrefix())
        for key, value in petsc_options.items():                                                # loops over the dictionary and writes each option into the PETSc options database under the active prefix
            opts[key] = value                                                                   # e.g. singular_direct_ksp_error_if_not_converged = True
        ksp.setFromOptions()                                                                    # apply options to our solver
        for key, value in petsc_options.items():                                                # cleanup of solver options
            del opts[key]
        opts.prefixPop()
        ksp.setOperators(A)                                                                     # telling the solver which matrix to solve with
        self.ksp = ksp

    # interpolate a given Python callable function f into FEM space so we can use it in the FEniCXs setting of FEM and UFL language
    def interpolate(self, f):
        f_fem = dolfinx.fem.Function(self.V_h)
        f_fem.interpolate(lambda x: f(x[0]))
        return f_fem
    
    # calculate the integral of a finite element function f_fem with repect to invariant density mu in the FEniCSx way
    def mu_mean(self, f_fem):
        mean_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(f_fem * self.mu * self.dx))       # turn UFL object into DOLFINx object that can be assembled, that is, compute the numerical value of the integral
        mean = self.domain.comm.allreduce(mean_local, op=MPI.SUM)                                   # due to parallelism, each MPI process owns only part of the mesh, so the assembly initially gives only the local contribution, hence, we add the local contributions from all processes here
        return mean
    
    # solve the Poisson equation for given Python callable function psi
    def solve(self, psi):

        # interpolate psi into FEM space with previous method
        psi_fem = self.interpolate(psi)

        # subtract the mean to get a centered RHS of the Poisson equation
        psi_mean = self.mu_mean(psi_fem)

        bar_psi_fem = dolfinx.fem.Function(self.V_h)
        bar_psi_fem.x.array[:] = psi_fem.x.array - psi_mean
        bar_psi_fem.x.scatter_forward()

        # linear form in variational formulation
        l = bar_psi_fem * self.v * self.mu * self.dx                 
        l_form = dolfinx.fem.form(l)                                 # again, this wraps the symbolic UFL forms into DOLFINx forms that can be assembled                                                                             

        # RHS
        b = dolfinx.fem.petsc.assemble_vector(l_form)                                           # assemble the load vector
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)         # parallel stuff again

        # compute the FEM solution
        Phi_h = dolfinx.fem.Function(self.V_h)                                                           # define dolfinx.fem solution
        self.ksp.solve(b, Phi_h.x.petsc_vec)                                                                 # solve the linear system
        Phi_h.x.scatter_forward()                                                                 # only for parallelism, synchronize the solution data across MPI processes after the solve

        Phi_h.x.array[:] = Phi_h.x.array[:] - self.mu_mean(Phi_h)                 # subtract mean from solution to fix a unique solution, this is the necessary normalization condition for the Poisson equation
        Phi_h.x.scatter_forward()

        return Phi_h             

    # compute the Dirichlet form associated with the generator L in FEniCXs style
    def dirichlet_form(self, f_fem, g_fem):
        integral_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
            (self.homog_diffusion**2 / 2) * ufl.inner(ufl.grad(f_fem), ufl.grad(g_fem)) * self.mu * self.dx)
            )
        integral = self.domain.comm.allreduce(integral_local, op=MPI.SUM)

        return integral                                     

# Hermite basis functions, n and x can both be arrays/lists or just scalars here
# the code looks weird, but it is just because of broadcasting rules in python and because I want to apply 
# each Hermite function to a whole matrix of data points
def psi(n, x):
    n_scalar = np.isscalar(n)
    x_scalar = np.isscalar(x)

    n = np.atleast_1d(n)
    x = np.asarray(x)

    # make n have one extra axis for each axis of x
    # x scalar  -> n2 shape (M,)
    # x 1D      -> n2 shape (M,1)
    # x 2D      -> n2 shape (M,1,1)
    n2 = n.reshape((len(n),) + (1,) * x.ndim)

    normalization = 1.0 / np.sqrt(
        np.sqrt(np.pi) * (2.0 ** n2) * factorial(n2)
    )

    result = normalization * eval_hermite(n2, x) * np.exp(-x**2 / 2)

    # return shape depending on scalar/array input
    if n_scalar and x_scalar:
        return result[0]
    elif n_scalar:
        return result[0]
    elif x_scalar:
        return result[:, ...]
    else:
        return result


# truncated Gaussian limit element
class Gaussian_Limit1D:
    def __init__(self, poisson, fourier_coefficient_number, sample_size):
        self.poisson = poisson                      # instance of the class Poisson1D
        self.N = fourier_coefficient_number         # Fourier coefficient cutoff for the Gaussian limit element
        self.S = sample_size                        # number of samples for the Gaussian limit element

        # NxN-dimensional asymptotic covariance matrix of the truncated Gaussiam limit element
        Phi_solutions = [self.poisson.solve(lambda x, n=n: psi(n,x)) for n in range(self.N)]         # list of Poisson equation solutions with RHS given by weighted Hermite basis function 
        tau = np.zeros((self.N, self.N))
        for n in range(self.N):
            for m in range(n, self.N):
                val = 2 * self.poisson.dirichlet_form(Phi_solutions[n], Phi_solutions[m])
                tau[n,m] = val
                tau[m,n] = val

        self.tau = tau
    
    # producing samples of the Gaussian limit element, x can be an array, everything is vectorized
    def samples(self, x):
        csi = np.random.multivariate_normal(np.zeros(self.N), self.tau, self.S)                          # Fourier coefficients of the Gaussian limit element

        return csi @ psi(np.arange(self.N), x)                                                      # (G(x))_s = sum_0^M csi(s, m) psi(m, x), so it is just matrix multiplication where each row corresponds to one sample
                              
    def density_plot(self, T):
        plt.figure(figsize=(8, 6))
        plt.grid(alpha=0.3)

        G_samples = self.samples(self.poisson.mesh)

        for s in range(self.S):
            label = r'$\mu + \frac{1}{\sqrt{T}} \Pi_N (\mathbb{G})$' if s == 0 else '_nolegend_'
            plt.plot(self.poisson.mesh, self.poisson.mu.x.array + G_samples[s]/np.sqrt(T), label=label, color="#9BD2D4", linewidth=1)

        plt.plot(self.poisson.mesh, self.poisson.mu.x.array, label=r'$\mu$',  color='#851e09', linestyle = '--', linewidth=2)
        plt.plot(self.poisson.mesh, self.poisson.mu_eps.x.array, label=r'$\mu_\varepsilon$', color="#e7bf6d", linestyle = '-', linewidth=2)

        plt.title(rf'$T = {T}, N = {self.N}$', fontsize=24)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)  

        plt.legend()
        plt.show()

    def sample_plot(self):
        plt.figure(figsize=(10, 6))
        plt.grid(alpha=0.3)

        G_samples = self.samples(self.poisson.mesh)
        colors = plt.cm.jet(np.linspace(0, 1, self.S))          # cycle through colors because pretty

        for s in range(self.S):
            label = r'$\Pi_N (\mathbb{G})$' if s == 0 else '_nolegend_'
            plt.plot(self.poisson.mesh, G_samples[s], label=label, color=colors[s], alpha=0.4, linewidth=1)

        plt.title(rf'$S = {self.S}$' + ' samples of ' + r'$\Pi_N (\mathbb{G})$' + ' with ' + rf'$N = {self.N}$', fontsize=24)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        #plt.savefig("./figures/gaussian_process_samples_N_50.pdf", bbox_inches="tight")
        plt.show()
    
# initial_conditions must be an array, e.g., the same array entry repeated for 10 times would simulate 10 paths from the same SDE 
@njit(parallel=True)
def simulate(initial_conditions, T, sigma, eps, dV, dp):
    dt = eps**3             # higher value than this would create some annoying discretization errors in the simulation
    N = int(np.ceil(T / dt))
    S = len(initial_conditions)

    X = np.empty((S, N+1))
    X[:,0] = initial_conditions

    inv_eps = 1.0 / eps
    sqrt_2sigma_dt = np.sqrt(2.0 * sigma * dt)
        
    for s in prange(S):
        for n in range(N):
            X[s, n+1] = X[s, n] - dV(X[s, n])*dt - inv_eps*dp(X[s, n] * inv_eps)*dt + sqrt_2sigma_dt*np.random.normal()

    return X

class Langevin1D:
    # dV and dp must be Numba-jitted functions if we want speedy sample generation, e.g.,
    # @njit
    # def dV(x):
    #   return x**3 - x
    def __init__(self, sigma, epsilon, potential_deriv, periodic_oscilation_deriv):
        self.sigma = sigma
        self.eps = epsilon
        self.dV = potential_deriv
        self.dp = periodic_oscilation_deriv

    def trajectory(self, initial_conditions, time_horizon):
        initial_conditions = np.atleast_1d(np.asarray(initial_conditions, dtype=np.float64))
        return simulate(initial_conditions, time_horizon, self.sigma, self.eps, self.dV, self.dp)    

# Hermite estimator
def mu_hat(data, N, x):
    #S = len(data)
    #alpha = np.empty((S, N))
    #for s in tqdm(range(S)):
    #    alpha[s,:] = np.mean(psi(np.arange(N), data[s]), axis=1)
    alpha = np.mean(psi(np.arange(N), data), axis=2).T             # can be used but only for smaller arrays, e.g., smaller sample sizes S or coarser discretization of the SDE, as one runs into memory problems fast
    return alpha @ psi(np.arange(N), x)


############################
## small robustness tests ##
############################

# Poisson solution corresponding to multiscale overdamped Langevin equation model
Poisson = Poisson1D(
    potential = lambda x: x**6/6 - x**5/5 + x,
    periodic_oscilation = lambda x: np.cos(x),
    sigma = 1,
    period = 2*np.pi,
    epsilon = 0.1,
    domain_cutoff = 3,
    mesh_number = 1000
)   

# Gaussian limit element corresponding to multiscale overdamped Langevin equation model
Gaussian = Gaussian_Limit1D(
    poisson = Poisson,
    fourier_coefficient_number = 16,
    sample_size = 500
)     
    
# Multiscale overdamped Langevin equation model with Numba-jitted drift
@njit
def dV(x):
    return x**5 - x**4 + 1

@njit
def dp(x):
    return -np.sin(x)

Langevin = Langevin1D(
    sigma = 1,
    epsilon = 0.1,
    potential_deriv = dV,
    periodic_oscilation_deriv = dp
)

#################################################
## Fourier coefficient-based tests for the CLT ##
#################################################

#fourier_modes = np.arange(0, 16, 2)
#fourier_modes = np.arange(10, 16, 2)
#fourier_modes = np.arange(11, 16, 2)
fourier_modes = np.arange(0, 2)

S = Gaussian.S          # sample size
T = 1000                # observation time

batches = 20
batch_size = S // batches

x0 = 1.0
initial_conditions = np.full(S, x0)

# estimates of the Fourier coefficients of mu with respect to hermite function
alpha_hat = np.empty((len(fourier_modes), S))

# serve the data in appropriate batches to avoid memory problems and kernel crashing
for b in tqdm(range(batches)):
    start = b * batch_size
    stop = (b + 1) * batch_size
    batch_init_cond = initial_conditions[start:stop]

    data_batch = Langevin.trajectory(batch_init_cond, time_horizon=T)
    alpha_hat[:, start:stop] = np.mean(psi(fourier_modes, data_batch), axis=2)

# true Fourier coefficients of mu
def alpha(modes):
    psi_fem = [Poisson.interpolate(lambda x, n=n: psi(n, x)) for n in modes]
    psi_mean = [Poisson.mu_mean(psi_n_fem) for psi_n_fem in psi_fem]
    return np.array(psi_mean).reshape(len(modes), 1)

emp_coeffs = np.sqrt(T) * (alpha_hat - alpha(fourier_modes))

fig, axs = plt.subplots(figsize = (8, 24), 
                        nrows=len(fourier_modes), 
                        layout="constrained", 
                        gridspec_kw={"hspace": 0.1})

for row, mode in enumerate(fourier_modes):
        x = emp_coeffs[row]
        ax = axs[row]

        tau_nn = np.diag(Gaussian.tau)[mode]

        # Center plotting window around 0 using the theoretical std
        x_left = min(x.min(), -4*np.sqrt(tau_nn))
        x_right = max(x.max(), 4*np.sqrt(tau_nn))
        xs = np.linspace(x_left, x_right, 400)

        ax.hist(x, bins=40, density=True, color = "#b4c2fa")
        ax.plot(xs, norm.pdf(xs, loc=0.0, scale=np.sqrt(tau_nn)), 
                color = "#6A1F50", linestyle = '--',
                label=rf"$\langle \mathbb{{G}}, \psi_{{{mode}}} \rangle_{{L^2(\mathbb{{R}})}} \sim \mathcal{{N}}_1(0,\tau_{{{mode}, {mode}}})$")
        
        ax.grid(alpha=0.3)
        ax.set_title(rf"$\langle \sqrt{{T}} (\widehat \mu^\varepsilon_{{N, T}} - \mu) , \psi_{{{mode}}} \rangle_{{L^2(\mathbb{{R}})}}$")
        ax.legend()
        
#fig.suptitle(r"Fourier coefficient estimates for $\varepsilon = 0.1$")
fig.suptitle(r"Fourier coefficient estimates for $\varepsilon = 0.05$")
#fig.savefig("./figures/fourier_coeff_even_clt_eps_01_N_16.pdf", bbox_inches="tight")
#fig.savefig("./figures/fourier_coeff_odd_last_3_clt_eps_01_N_16.pdf", bbox_inches="tight")
#fig.savefig("./figures/fourier_coeff_even_last_3_clt_eps_005_N_16.pdf", bbox_inches="tight")
#fig.savefig("./figures/fourier_coeff_odd_last_3_clt_eps_005_N_16.pdf", bbox_inches="tight")

##################################
## further distributional plots ##
##################################

N = 16                  # number of Fourier coefficients in estimator
S = 100                 # sample size, don't need that much here anyway
m = len(Poisson.mesh)   # discretization number of mesh
T = 1000                # observation time

batches = 10
batch_size = S // batches

x0 = 1.0
initial_conditions = np.full(S, x0)

# estimates of the Fourier coefficients of mu with respect to hermite function
mu_hat_estimates = np.empty((S, m))

# serve the data in appropriate batches to avoid memory problems and kernel crashing
for b in tqdm(range(batches)):
    start = b * batch_size
    stop = (b + 1) * batch_size
    batch_init_cond = initial_conditions[start:stop]

    data_batch = Langevin.trajectory(batch_init_cond, time_horizon=T)
    mu_hat_estimates[start:stop, :] = mu_hat(data_batch, N, Poisson.mesh)

plt.figure(figsize=(8, 6))
plt.grid(alpha=0.3)

for s in range(S):
    label = r'$\widehat \mu^\varepsilon_{N, T}$' if s == 0 else '_nolegend_'
    plt.plot(Poisson.mesh, mu_hat_estimates[s], label=label, color="#b4c2fa", linewidth=1)

plt.plot(Poisson.mesh, Poisson.mu.x.array, label=r'$\mu$',  color='#851e09', linestyle = '--', linewidth=2)
plt.plot(Poisson.mesh, Poisson.mu_eps.x.array, label=r'$\mu_\varepsilon$', color="#e7bf6d", linestyle = '-', linewidth=2)

plt.suptitle(rf'$T = {T}, \, N = {N}, \, \varepsilon = {Langevin.eps}$', fontsize=24)
plt.legend(fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)  

plt.title(r"$V(x) = x^4/4 - x^2/2$", fontsize = 18)

plt.legend()
#plt.savefig("./figures/estimator_density_distributions_2.pdf", bbox_inches="tight")
plt.show()

G_samples = Gaussian.samples(Poisson.mesh)

plt.figure(figsize=(8, 6))
plt.grid(alpha=0.3)

for s in range(S):
    label = r'$\Pi_N( \sqrt{T} (\widehat \mu^\varepsilon_{N, T} - \mu))$' if s == 0 else '_nolegend_'
    plt.plot(Poisson.mesh, np.sqrt(T) * (mu_hat_estimates[s] - Poisson.mu.x.array), label=label, color="#b4c2fa", linewidth=1)

    label = r'$\Pi_N (\mathbb{G})$' if s == 0 else '_nolegend_'
    plt.plot(Poisson.mesh, G_samples[s], label=label, color="#6A1F50", linewidth=1)

plt.suptitle(rf'$T = {T}, \, N = {N}, \, \varepsilon = {Langevin.eps}$', fontsize=24)
plt.legend(fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.title(r"$V(x) = x^4/4 - x^2/2$", fontsize = 18)

plt.legend()
#plt.savefig("./figures/estimator_gaussian_process_proj_distributions.pdf", bbox_inches="tight")
plt.show()
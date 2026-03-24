import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial
from scipy.integrate import quad

# FEniCSx specific packages
from mpi4py import MPI              # for parallelism
import dolfinx.fem.petsc            # assembly functions that turn ufl symbolic forms into actual matrices and vectors, creating meshes, dolfinx.fem function spaces, variational forms, etc.
from petsc4py import PETSc          # linear algebra and solver library underneath FEniCSx
import ufl                          # symbolic language used to write variational forms


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
        mu_fem.x.scatter_forward()                                                      # only for parallelism, synchronize the solution data across MPI processes after the solve
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

# Multiscale overdamped Langevin equation with linear drift and cosine oscillation
Poisson_linear_drift = Poisson1D(
    potential = lambda x: x**4/4 - x**2/2,
    periodic_oscilation = lambda x: np.cos(x),
    sigma = 1,
    period = 2*np.pi,
    epsilon = 0.1,
    domain_cutoff = 5,
    mesh_number = 1000
)                                        

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

mesh = Poisson_linear_drift.domain.geometry.x[:,0]                                                              # mesh for different purposes, integration, plotting etc.
M = 16                                                                                 # Fourier coefficient cutoff for the Gaussian limit element
S = 500                                                                                                # number of samples for the Gaussian limit element
Phi_solutions = [Poisson_linear_drift.solve(lambda x, n=n: psi(n,x)) for n in range(M)]                              # list of Poisson equation solutions with RHS given by weighted Hermite basis function 

# asymptotic covariance matrix
tau = np.zeros((M, M))
for n in range(M):
    for m in range(n, M):
        tau[n,m] = 2 * Poisson_linear_drift.dirichlet_form(Phi_solutions[n], Phi_solutions[m])
tau = (tau + tau.T)/2

# Fourier coefficients of the Gaussian limit element
csi = np.random.multivariate_normal(np.zeros(M), tau, S)

# Samples of the Gaussian limit element;
# (G(x))_s = sum_0^M csi(s, m) psi(m, x), so it is just matrix multiplication where each row corresponds to one sample;
# x can be an array, everything is vectorized
def G(x):
    m = np.arange(M)
    return csi @ psi(m, x)

mu_points = Poisson_linear_drift.mu.x.array
mu_eps_points = Poisson_linear_drift.mu_eps.x.array
T = 1000

plt.figure()
plt.grid()
for s in range(S):
    plt.plot(mesh, mu_points + G(mesh)[s,:]/np.sqrt(T), color="#9BD2D4", linewidth=1)
plt.plot(mesh, mu_points, color='#851e09', linewidth=1.5)
plt.plot(mesh, mu_eps_points, color="#D5A927", linewidth=1.5)
plt.show()

plt.figure()
plt.grid()
for s in range(S):
    plt.plot(mesh, G(mesh)[s,:]/np.sqrt(T), color='lightblue', linewidth=1)
plt.show()

# Numba-jitted functions so we can simulate trajectories faster
@njit
def dV(x):
    return x**3 - x

@njit
def dp(x):
    return -np.sin(x)
    
# initial_conditions must be an array, e.g., the same array entry repeated for 10 times would simulate 10 paths from the same SDE 
@njit(parallel=True)
def simulate(initial_conditions, T, sigma, eps):
    dt = eps**2
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
    def __init__(self, sigma, epsilon):
        self.sigma = sigma
        self.eps = epsilon

    def trajectory(self, initial_conditions, time_horizon):
        initial_conditions = np.atleast_1d(np.asarray(initial_conditions, dtype=np.float64))
        return simulate(initial_conditions, time_horizon, self.sigma, self.eps)
    
# Multiscale overdamped Langevin equation with linear drift and cosine oscillation
Langevin_linear_drift = Langevin1D(
    sigma = 1,
    epsilon = 0.1
) 

data = Langevin_linear_drift.trajectory(np.full((S,), 1.0), T)

alpha = np.empty((S, M))
for s in range(S):
    alpha[s,:] = np.mean(psi(np.arange(M), data[s]), axis=1)

#alpha = np.mean(psi(np.arange(M), data), axis=2).T             # can be used but only for smaller sample sizes S as one runs into memory problems fast

def mu_hat(x):
    return alpha @ psi(np.arange(M), x)

plt.figure()
plt.grid()
for s in range(S):
    plt.plot(mesh, mu_hat(mesh)[s,:], color="#9BD2D4", linewidth=1.5)
plt.plot(mesh, mu_points, color='#851e09', linewidth=1.5)
plt.plot(mesh, mu_eps_points, color="#D5A927", linewidth=1.5)
plt.show()
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.integrate import quad

# FEniCSx specific packages
from mpi4py import MPI              # for parallelism
from dolfinx import mesh, dolfinx.fem       # creating meshes, dolfinx.fem function spaces, variational forms, etc.
import dolfinx.fem.petsc            # assembly functions that turn ufl symbolic forms into actual matrices and vectors
from petsc4py import PETSc          # linear algebra and solver library underneath FEniCSx
import ufl                          # symbolic language used to write variational forms

def solve_Poisson(domain_cutoff, mesh_number, homog_diffusion, potential, hermite):

    return

N = 1000
R = 5

domain = mesh.create_interval(MPI.COMM_WORLD, N, [-R, R])                               # define mesh of interval [-R, R]

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))                                          # define finite element space of piecewise linear functions

x = ufl.SpatialCoordinate(domain)                                                       # spatial coordinate 
dx = ufl.dx(domain=domain)                                                              # Lebesgue integration measure

Sigma = 1
L = 2*np.pi
p = lambda x: np.cos(x)
Pi_plus = quad(lambda x: np.exp(+p(x)/(Sigma**2)), 0, L)[0]
Pi_minus = quad(lambda x: np.exp(-p(x)/(Sigma**2)), 0, L)[0]
K = (L**2)/(Pi_plus*Pi_minus)
bar_Sigma = K*Sigma

V_potential = 0.5*x[0]**2
mu_unnormalized = ufl.exp(-2*V_potential/bar_Sigma**2)                                  # symbolic finite element expression

Z_local = dolfinx.fem.assemble_scalar(fem.form(mu_unnormalized*dx))                             # turn UFL object into DOLFINx object that can be assembled, that is, compute the numerical value of the normalization constant
Z = domain.comm.allreduce(Z_local, op=MPI.SUM)                                          # due to parallelism, each MPI process owns only part of the mesh, so the assembly initially gives only the local contribution,
mu = mu_unnormalized/Z                                                                  # hence, we add the local contributions from all processes here

psi = x[0]
psi_mean_local = dolfinx.fem.assemble_scalar(fem.form(psi*mu*dx))                               # same as above
psi_mean = domain.comm.allreduce(psi_mean_local, op=MPI.SUM)
bar_psi = psi - psi_mean

u = ufl.TrialFunction(V)                                                                # placeholder for unknown solution in V
v = ufl.TestFunction(V)                                                                 # placeholder for an arbitrary test function in V

a = bar_Sigma**2/2*ufl.inner(mu*ufl.grad(u), ufl.grad(v))*dx                            # bilinear form in variational formulation
l = bar_psi*v*mu*dx                                                                     # linear form in variational formulation

a_form = dolfinx.fem.form(a)                                                                    # again, this wraps the symbolic UFL forms into DOLFINx forms that can be assembled
l_form = dolfinx.fem.form(l)

bcs = []                                                                                # weak form uses only the natural boundary conditions, that is, zero-flux condition mu*Phi' = 0, or, in other words, nothing important leaves the boundary and the weak formulation already encodes the boundary condition

A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)                                  # assemble the stiffness matrix
A.assemble()                                                                              

b = dolfinx.fem.petsc.assemble_vector(l_form)                                           # assemble the load vector
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

nullspace = PETSc.NullSpace().create(constant=True, comm=domain.comm)                   # since our Poisson problem is singular (constants are in the kernel of the generator and the stiffness matrix), we must specify this nullspace of constants
assert nullspace.test(A)                                                                # test that it is indeed nullspace of A
A.setNullSpace(nullspace)

petsc_options = {                                                                       # configuring a direct solver (MUMPS)
    "ksp_error_if_not_converged": True,                                                 # give error if not converging
    "ksp_type": "preonly",                                                              # precondition only once since we use direct solver through factorization
    "pc_type": "lu",                                                                    # use LU factorization 
    "pc_factor_mat_solver_type": "mumps",                                               # MUMPS is a parallel sparse direct solver that does the LU factorization
    "ksp_monitor": None,                                                                # solver progress information is turned off
}

ksp = PETSc.KSP().create(domain.comm)                                                   # creating the Krylov subspace solver
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

u_h = dolfinx.fem.Function(V)                                                           # define dolfinx.fem solution
ksp.solve(b, u_h.x.petsc_vec)                                                           # solve the linear system
u_h.x.scatter_forward()                                                                 # only for parallelism, synchronize the solution data across MPI processes after the solve

u_mean_local = dolfinx.fem.assemble_scalar(fem.form(u_h*mu*dx))                                 # subtract mean from solution to fix a unique solution, this is the necessary boundary condition for the Poisson equation
u_mean = domain.comm.allreduce(u_mean_local, op=MPI.SUM)

u_h.x.array[:] -= u_mean
u_h.x.scatter_forward()


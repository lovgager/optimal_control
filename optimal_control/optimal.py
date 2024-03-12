import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#%% primal problem

@njit
def solve_primal(rhs, init, u): # rhs(r, t), init(r), u(t) arrays
    phi = np.zeros((Nt+1, Nr+1))
    phi[0] = init
    for n in range(Nt):
        p = dr/r
        phi[n+1][1:-1] = mu*gamma*((1-p[1:-1]/2)*phi[n][:-2] + (1+p[1:-1]/2)*phi[n][2:]) + \
                (1-2*mu*gamma-dt*b)*phi[n][1:-1] + dt*rhs[1:-1, n]
        phi[n+1][0] = phi[n+1][1]
        phi[n+1][-1] = phi[n+1][-2] + dr/mu*u[n+1]
        
    return phi


#%% conjugate problem

@njit
def solve_conj(rhs, final, bound):
    psi = np.zeros((Nt+1, Nr+1))
    psi[0] = final
    rhs = rhs[:, ::-1]
    u = bound[::-1]
    
    for n in range(Nt):
        p = dr/r
        psi[n+1][1:-1] = mu*gamma*((1-p[:-2]/2)*psi[n][:-2] + (1+p[2:]/2)*psi[n][2:]) + \
                (1-2*mu*gamma-dt*b)*psi[n][1:-1] + dt*rhs[1:-1, n]
        psi[n+1][0] = psi[n+1][1] + dr/mu*u[n+1]
        psi[n+1][-1] = psi[n+1][-2]
        
    return psi[::-1, :]


#%% parameters

mu = 1
b = 1
R0 = 2
R1 = 3
T = 1

Nr = 70
Nt = 10000
r = np.linspace(R0, R1, Nr + 1)
t = np.linspace(0, T, Nt + 1)
dr = (R1 - R0)/Nr
dt = T/Nt
gamma = dt/dr**2


f = lambda r, t: np.exp(t)*((1 + b + mu)*np.cos(r) + mu/r*np.sin(r))
u0 = lambda t: -mu*np.exp(t)*np.sin(R1)
phi_init = lambda r: np.cos(r)
phi_obs = lambda t: np.exp(t)*np.cos(R0)

r_grid, t_grid = np.meshgrid(r, t, indexing='ij')
f = f(r_grid, t_grid)
u = u0(t)
phi_init = phi_init(r)
phi_obs = phi_obs(t)

tau = 1
N = 31
    

#%% iterations

plt.figure(figsize=(10, 8))
for alpha in np.arange(0, 0.5, 0.1):

    u = u0(t)
    J = np.zeros(N)
    for k in range(N):
        phi = solve_primal(f, phi_init, u)
        psi = solve_conj(np.zeros(r_grid.shape), np.zeros(r.shape), phi[:,0] - phi_obs)
        u -= tau*(alpha*u + psi[:,-1])
        
        J[k] = dr/2*np.linalg.norm(phi[:,0] - phi_obs)**2 + alpha*dr/2*np.linalg.norm(u)**2
    plt.plot(J, '.-', label=f'alpha = {np.round(alpha,1)}')
        
plt.xlabel('iterations')
plt.ylabel('J(u)')
plt.title(f'tau = {tau}')
plt.legend()
plt.grid(True)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)

L = 5_000 # no. grid points
m = 10_000 # no. time steps

l_src = 1_000
delta = 0.02
tau = 0.9*delta # case 1) fullfill Courant condition => stable
# tau = 1.05*delta # case 2) violate Courant cond. => instable

Ez = np.zeros(L+1) # Electric field components
Hy = np.zeros(L) # Magnetic field components

N = 50 # grid points per space step
n = 1.46 # index of refraction
eps = np.ones(L+1) # permittivity of simulation space
eps[L//2:L//2+2*N] = n**2 # case 1) thin glass plate in middle
# eps[L//2:] = n**2 # case 2) very thick glass plate

sigma = np.ones(L+1) # boundary conditions
sigma[6*N:-6*N] = 0 # everywhere 0 except close to bounds

A = (1-sigma[1:]*tau/2) / (1 + sigma[1:]*tau/2)
B = tau / (1 + sigma[1:]*tau/2)
C = (1-sigma*tau/(2*eps)) / (1+sigma*tau/(2*eps))
D = (tau/eps) / (1+sigma*tau/(2*eps))

def source(t):
    t0 = 30
    spread = 10
    return np.sin(2*np.pi*t)*np.exp(-((t-t0)/spread)**2)

for i in range(m):

    Ez[1:-1] = D[1:-1]*(Hy[1:]-Hy[:-1])/delta + C[1:-1]*Ez[1:-1] # Update E field + bound. cond.
    Ez[l_src] -= D[l_src]*source(i*tau) # Update field due to source
    Hy = B*(Ez[1:]-Ez[:-1])/delta + A*Hy # Update H field incl. bound. cond.

    if i in [2000,5000]:
    # if i in [20,22,24,26]:
        plt.figure(figsize=(6,4))
        plt.ylim([-0.015,0.015])
        plt.plot(Ez)

        plt.fill_between(range(L+1), sigma, alpha=0.6, color='tab:blue')
        plt.fill_between(range(L+1), -sigma, alpha=0.6, color='tab:blue')
        plt.fill_between(range(L+1), eps-1, alpha=0.6, color='tab:orange')
        plt.fill_between(range(L+1), -(eps-1), alpha=0.6, color='tab:orange')

        plt.xlabel('l')
        plt.ylabel('E')
        plt.minorticks_on()
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        plt.show()

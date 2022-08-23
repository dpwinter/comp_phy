import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)

# parameters
gamma = 1
f_0 = 4
f_1 = 1/4
h = 2*np.pi * f_1 # field strength
B_0 = 2*np.pi * f_0 # Z-component of B-field
omega_0 = B_0 # resonance frequency

m = 100 # time steps
T = 4 # simulation time
tau = T / m # time step size
ts = np.arange(0,T+tau,tau) # time range

# T1, T2 decay step unitary
T_1_inv, T_2_inv = 1, 1
C = np.exp( - 0.5 * tau * np.array([T_2_inv, T_2_inv, T_1_inv]) )

Ms = [] # track M vector per time step
M = np.array([0,0,1]) # inital condition
Ms.append(M) # save initial M

# initial B-field phase offset
phi = np.pi / 8

for i in range(0,m):

    # time for B field update
    t = (i+0.5)*tau

    # update B field
    Bx = h * np.cos(omega_0 * t + phi)
    By = -h * np.sin(omega_0 * t + phi)
    Bz = B_0

    # precompute factors
    omega2 = Bx**2 + By**2 + Bz**2
    omega = np.sqrt(omega2)
    a = np.cos(omega * tau * gamma)
    b = 1 - a
    c = omega * np.sin(omega * tau * gamma)

    # precompute field products
    BxBy = Bx * By
    BxBz = Bx * Bz
    ByBz = By * Bz
    Bx2 = Bx**2
    By2 = By**2
    Bz2 = Bz**2

    # time step unitary
    U = 1/omega2 * np.array([[Bx2 + a*(By2 + Bz2), b*BxBy + c*Bz,            b*BxBz - c*By],
                             [b*BxBy - c*Bz,       By2 + a*(Bx2 + Bz2),      b*ByBz + c*Bx],
                             [b*BxBz + c*By,       b*ByBz - c*Bx,      Bz2 + a*(By2 + Bx2)]])

    # update magnetization
    M = C * M
    M = U @ M
    M = C * M

    # save magnetization
    Ms.append(M)

# Plotting
plt.plot(ts,Ms);

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)

# discretization parameters
delta = 0.1
L = 1001
tau = 0.001
m = 50_000

# 2x2 time-step block
X = np.array([[0,1],[1,0]])
I = np.eye(2)
a = tau/(4*delta**2)
A = np.cos(a)*I + 1j*np.sin(a)*X

# given parameters in l-space
l = np.array(range(L))
l_0 = 20/delta
q = 1*delta
sigma = 3/delta

# potentials in l-space
V1, V2 = np.zeros(L), np.zeros(L)
V2[int(50/delta):int(50.5/delta)] = 2
Vs = [V1,V2]

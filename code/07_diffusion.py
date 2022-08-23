import numpy as np
import matplotlib.pyplot as plt

D = 1
L = 1001
delta = 0.1
tau = 0.001
m = 10_000

idx0 = L // 2 # init condition 1
# idx0 = 0 # init condition 2

def expM(factor=1):
    '''Calculate exponential matrix e^aX. Also returns a.'''
    a = - tau * D / delta**2 * factor
    x = 1 + np.exp(2*a)
    y = 1 - np.exp(2*a)
    M = 0.5 * np.array([[x,y],[y,x]])
    return M, a

expB, aB = expM()
expA, aA = expM(factor=0.5)

N = np.zeros(L)
N[idx0] = 1 # start with all density at source location idx0

x1 = np.zeros(m) # Track 1st moment
x2 = np.zeros(m) # Track 2nd moment

diff0 = (np.array(range(L)) - idx0) * delta # Offset of each grid point to source

for i in range(m):
    
    ### Track moments ###

    x1[i] = np.sum(N * diff0) / np.sum(N) # 1st moment
    x2[i] = np.sum(N * diff0**2) / np.sum(N) # 2nd moment
    
    ### Update state ###

    # e^aA/2 @ N
    N[:-1] = np.hstack([eA @ v for v in np.split(N[:-1], 500)]).ravel()
    N[-1] *= np.exp(aA)
    
    # e^aB @ N
    N[1:] = np.hstack([eB @ v for v in np.split(N[1:], 500)]).ravel()
    N[0] *= np.exp(aB)
    
    # e^aA/2 @ N
    N[:-1] = np.hstack([eA @ v for v in np.split(N[:-1], 500)]).ravel() 
    N[-1] *= np.exp(aA)
    
var = (x2 - x1**2) / delta**2 # Calculate variance
plt.plot(np.arange(0,m*tau, tau), var); # Plot variance over time range with tau steps
plt.xlabel('t');
plt.ylabel('Var[i(t)]');

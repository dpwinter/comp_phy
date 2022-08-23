from common import *

N = 100
n = 128 # no. of oscillators
dt = 0.1
x = np.zeros((N,n))
v = np.zeros((N,n))
t = np.arange(0,N*dt, dt)

# x[0,(n-1)//2] = 1 #initial condition 1

# inital condition 2
j = (n-1) // 2 # and (n-1)//2 or 1
for k in range(n):
    x[0,j] = np.sin(np.pi*(k+1)*j / (n+2))

def force(x):  # Defining the force between oscillators
    res = np.empty_like(x)
    res[1:-1] = -2*x[1:-1] + x[:-2] + x[2:]
    res[0] = -x[0] + x[1]
    res[-1] = -x[-1] + x[-2]
    return res

for i in range(N-1):  # Velocity Verlet algorithm
    v_n = v[i,:] + force(x[i,:]) * dt / 2.0
    x[i+1,:] = x[i,:] + v_n * dt
    v[i+1,:] = v_n + force(x[i+1,:]) * dt / 2.0
    
E = 0.5 * np.sum(v**2, axis=1) + 0.5 * np.sum( (x[:,:-1] - x[:,1:])**2 , axis=1) # total energy

plot(t, [*(x.T), E], ylim=[-2,2], path='./10.png')
# plot(t, [*(x.T[:4]), E], lbls=[r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', 'E'], ylim=[-2,2], path='./10.png')

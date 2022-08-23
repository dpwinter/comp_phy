from common import *

dt = 0.1  # time spacing
N = 1000  # no. of space steps
x = np.zeros((N)) # positions
v = np.zeros((N)) # velocities
t = np.arange(0,N*dt,dt) # time

# initial conditions
x[0] = 0
v[0] = 1

for i in range(N-1):
    x[i+1] = x[i] + v[i]*dt # update position
    v[i+1] = v[i] - x[i]*dt # update velocity (f = -kx = -dVdx)

plot(t,[x,np.sin(t)],['Numerical solution', 'Analytical solution'], 
        ylabel='Position', path='./1.png')

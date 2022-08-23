from common import *

dt = 0.1 # time step
N = 1000 # no. of space steps
x = np.zeros((N)) # positions
v = np.zeros((N)) # velocities
t = np.arange(0,N*dt,dt) # time

x[0] = 0
v[0] = 1

# Variant 1:
# for i in range(N-1):
#     v[i+1] = v[i] - x[i]*dt # update velocity first
#     x[i+1] = x[i] + v[i+1]*dt # update position afterwards

# Variant 2:
for i in range(N-1):
    x[i+1] = x[i] + v[i]*dt # update position first
    v[i+1] = v[i] - x[i+1]*dt # update velocity afterwards

E = v**2/2 + x**2/2
x_ana = np.sin(t)
plot(t,[x,x_ana],lbls=['Numerical solution','Analytical solution'], markers=['-','--'], path='./2a.png', ylabel='Position')

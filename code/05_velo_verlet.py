from common import *

dt = 0.1 # time spacing
N = 1000 # no. of space steps
x = np.zeros(N) # positions
v = np.zeros(N) # velocities
t = np.arange(0,N*dt, dt) # time

x[0] = 0
v[0] = 1

for i in range(N-1):
    a = -x[i]           # force
    v_n = v[i] + a*dt/2.0   # velocity after dt/2 (half kick)
    x[i+1] = x[i] + v_n*dt
    a = -x[i+1]         # force at new position
    v[i+1] = v_n + a*dt/2.0 # velocity from t+dt/2 (remaining half kick)

E = v**2/2 + x**2/2
x_ana = np.sin(t)
plot(t,[v,E],lbls=['Velocity','Energy'],
        markers=['-','-'], path='./4.png')

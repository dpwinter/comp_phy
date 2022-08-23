import numpy as np
np.random.seed(1234)

N_steps = 32
N_samples = 10_000
HWP2 = 0
T0 = 1_000
W=1

cHWP2 = 1
sHWP2 = 0
n = 360
E1 = np.zeros((4,n))
E2 = np.zeros((4,n))
E12 = np.zeros((4,n))
count = np.zeros((2,2,2,n), dtype=int)
tot = np.zeros((2, n), dtype=int)

def analyzer(c, s, cHWP, sHWP, T0):
    c2 = cHWP * c + sHWP * s
    s2 = -sHWP * c + cHWP * s
    x = c2**2 - s2**2
    y = 2 * c2 * s2

    j = 0 if (x > 2 * np.random.uniform() - 1) else 1
    l = y**4 * T0 * np.random.uniform()

    return l, j

for i in range(N_steps):
    cHWP1 = np.cos(i * 2 * np.pi / N_steps)
    sHWP1 = np.sin(i * 2 * np.pi / N_steps)
    for j in range(N_samples):
        r = np.random.uniform()
        c1 = np.cos( r * 2 * np.pi )
        s1 = np.sin( r * 2 * np.pi )
        c2 = -s1
        s2 = c1
        l1, j1 = analyzer(c1,s1,cHWP1,sHWP1,T0)
        l2, j2 = analyzer(c2,s2,cHWP2,sHWP2,T0)

        count[j1,j2,0,i] += 1
        if abs(l1-l2) < W:
            count[j1,j2,1,i] += 1

for j in range(N_steps):
    for i in range(2):
        tot[i,j] = np.sum(count[:,:,i,j])
        E12[i,j] = count[0,0,i,j] + count[1,1,i,j] - count[1,0,i,j] - count[0,1,i,j]
        E1[i,j] = count[0,0,i,j] + count[0,1,i,j] - count[1,1,i,j] - count[1,0,i,j]
        E2[i,j] = count[0,0,i,j] + count[1,0,i,j] - count[1,1,i,j] - count[0,1,i,j]
        if tot[i,j] > 0:
            E12[i,j] = E12[i,j] / tot[i,j]
            E1[i,j] = E1[i,j] / tot[i,j]
            E2[i,j] = E2[i,j] / tot[i,j]

xs, ys = [], []
for i in range(1):
    print("HI")
    for j in range(N_steps):
        s = 3 * E12[i,j] - E12[i, np.mod(j*3, N_steps)]

        print(j * 360 / N_steps, E12[i,j], tot[i,j], s)
        xs.append(j*360/N_steps)

import matplotlib.pyplot as plt

print(len(xs))
plt.plot(xs, E12[0,:N_steps])
plt.plot(xs, E12[1,:N_steps])
plt.show()


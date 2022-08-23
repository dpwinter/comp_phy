from common import *
from scipy.fftpack import fft, fftfreq
import sim

psis = sim.run()

for i in range(len(Vs)):
    for j in [0,-2]:

        # Fourier transform state vector
        fx = fftfreq(L,delta)
        fy = fft(psis[i,j])

        # Convert to probability
        p = (fy.real**2 + fy.imag**2)
        p /= np.sum(p) # renormalize

        # Find maximum
        xmax = fx[np.argmax(p)]
        ymax = p.max()

        # Plot transform and maximum
        plt.plot(fx, p, label='t=%d'%T[j])
        plt.plot(xmax, ymax, 'x', c='red');

plt.xlabel(r'v [$\Delta/\tau$]');
plt.ylabel(r'P(v,t)');

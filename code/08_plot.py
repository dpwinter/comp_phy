from common import *
import sim

psis = sim.run()
probs = np.zeros_like(psis)

for i in range(len(Vs)):
    for j in range(len(T)):
    	# Calculate probabilit from state vector
        probs[i,j] = psis[i,j].real**2 + psis[i,j].imag**2

# normalize the tunneled portion
max_tot_prob = np.max(np.sum(probs[1,:,int(50.5/delta):], axis=1))
probs[1,:,int(50.5/delta):] = probs[1,:,int(50.5/delta):] / max_tot_prob

# plot probabilities
cols = ['tab:blue','tab:orange']
for i,v_col in enumerate(cols):
    for j,t in enumerate(T):
        plt.plot(x*delta, probs[i,j,:], c=v_col)

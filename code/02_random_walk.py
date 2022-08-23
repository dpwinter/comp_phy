import numpy as np
import matplotlib.pyplot as plt

n_steps = 1_000
n_particles = 10_000
np.random.seed(1380) # Set seed
X = np.zeros((n_particles, n_steps)) # initialize result matrix

for i in range(1,n_steps):
    r = np.random.uniform(size=(n_particles)) # random vector
    X[r>0.5,i] += X[r>=0.5,i-1] + 1 # Add previous value +1 for each "Heads"
    X[r<0.5,i] += X[r<0.5,i-1] - 1 # Add previous value -1 for each "Tails"

avg = np.sum(X, axis=0) / n_particles # Calculate expectation value of x numerically
var = np.sum(X**2, axis=0) / n_particles - avg ** 2 # Calculate variance numerically

# Plot variance as function of no. of events
plt.figure(figsize=(10,6))
plt.plot(range(n_steps), range(n_steps), '--', label="Analytical") # Analytical variance f(n)=n
plt.plot(range(n_steps), var, label="Numerical")
plt.xlabel('Number of steps N')
plt.ylabel(r'Variance of final position $Var[x]$')
plt.grid()
plt.legend()
plt.savefig("fig", dpi=300)

# Plot histogram of final position values
plt.figure(figsize=(10,4))
plt.hist(X[:,-1], bins=300)
plt.xlabel(r'Final position after $N$ steps')
plt.ylabel(r'Number of occurances')
plt.grid()
plt.savefig("fig2", dpi=300)

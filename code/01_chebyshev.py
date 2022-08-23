import numpy as np
import matplotlib.pyplot as plt

def cheby(x: np.ndarray, N: int) -> np.ndarray: # declare input/output types
    """Calculate T values up to (inclusive) N for values in x."""

    assert np.all(x >= -1) and np.all(x <= 1) and N >= 0 # assert if arguments in range

    def T(x,n): # We only need T(x,n) to be available in the context of cheby. This justifies a closure.
        """Recursively calculate T values."""

        # Guard clauses as defined in the exercise.
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return 2 * x * T(x, n-1) - T(x, n-2) # recursive function call

    res = np.zeros((len(x),N+1)) # create array of zeros of desired shape
    for n in range(N+1): # iterate over all n values up to N+1
        res[:,n:n+1] = T(x,n) # fill columns with T vector corresponding to one n.

    return res

x = np.linspace(-1,1,100)[:,None] # column vector of test values
N = 4 # inclusive max. order for which to generate Chebychev polynomials

res = cheby(x,N) # generate matrix of shape len(x)*(N+1) matrix to store T values

# plot the result
for n,col in zip(range(N+1),res.T):
    plt.plot(x,col,label=r'$T_{%d}$(x)'%n)

plt.grid()
plt.title("Chebychev polynomials of first kind")
plt.xlabel("x")
plt.ylabel("T")
plt.legend()
plt.savefig("cheby.png", dpi=300)

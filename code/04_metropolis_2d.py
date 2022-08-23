@numba.jit(nopython=True)
def energy_diff(S,N,i,j):
    '''Calculate energy difference to neighbors'''
    l = 0 if i-1 < 0 else S[i-1,j]
    r = 0 if i+1 > N-1 else S[i+1,j]
    u = 0 if j-1 < 0 else S[i,j-1]
    d = 0 if j+1 > N-1 else S[i,j+1]
    return 2*S[i,j]*(r+l+u+d)

@numba.jit(nopython=True)
def energy(S,N):
    '''Calculate energy of configuration S'''
    E = 0
    for i in range(N-1):
        for j in range(N):
            E -= S[i,j]*S[i+1,j]+S[j,i]*S[j,i+1] # see Eq. 1
    return E

@numba.jit(nopython=True)
def magnetization(S):
    '''Calculate magnetization of configuration S'''
    return S.sum()

@numba.jit(nopython=True)
def metropolis_step(S,N,p):
    '''Metropolis algorithm'''

    for _ in range(N*N):
        i = random.randrange(N)
        j = random.randrange(N)
        dE = energy_diff(S,N,i,j)
        if random.random() < np.exp(-dE/T):
            S[i,j] = -S[i,j]
    return S

@numba.jit(nopython=True)
def monte_carlo(S,T,N,N_samples,N_therm):  
    '''Repeatedly call Metropolis-Hastings and calculate observables from samples'''

    for _ in range(N_therm): # Thermalize
        S = metropolis_step(S,N,T)     

    E = np.zeros(N_samples)
    M = np.zeros(N_samples)
    for i in range(N_samples):
        S = metropolis_step(S,N,T)
        E[i] = energy(S,N)
        M[i] = magnetization(S)
    return E,M,S

Ts = np.arange(0.2,4.2,0.2)
N_samples = 10_000
N_therm = int(N_samples / 10)

Us = np.zeros(len(Ts))
Cs = np.zeros(len(Ts))
Ms = np.zeros(len(Ts))

observable = 'M' # for plotting
plt.figure(figsize=(6,6))
for N in [10,50,100]:
    for i,T in enumerate(Ts):
        S_init = np.random.choice([-1,1], size=(N,N)) # hot start
        E,M,S = monte_carlo(S_init,T,N,N_samples,N_therm)

        # track observables of interest
        Us[i] = np.mean(E) / N**2
        Cs[i] = np.var(E) / N**2 / T**2
        Ms[i] = np.mean(M) / N**2

    if observable == 'U':
        plt.plot(Ts,Us,label=r'MC, N=%d'%N)
    elif observable == 'C':
        plt.plot(Ts,Cs,label=r'MC, N=%d'%N)
    elif observable == 'M':
        plt.plot(Ts,abs(Ms),label=r'MC, N=%d'%N)


if observable == 'M':
    T_c = 2/np.log(1+np.sqrt(2))
    M = lambda T: (1 - np.sinh(2/T)**(-4))**(1/8)
    M_th = np.piecewise(Ts, [Ts < T_c, Ts >= T_c], [M, 0])
    plt.plot(Ts, M_th, '--', c='k', label='Th.')

plt.axvline(T_c, c='red', linestyle='--', label=r'$T_C$')
plt.legend()
plt.xlabel(r'T')
plt.ylabel(r'%s/N$^2$' % observable)
plt.minorticks_on()
plt.grid(which='major', color='#CCCCCC', linestyle='--')
plt.grid(which='minor', color='#CCCCCC', linestyle=':')

fname = '2D_%s_%d' % (observable,N_samples)
plt.savefig('./report/src/%s' % fname, dpi=300)



@numba.jit(nopython=True)
def energy(S,N): 
    '''Calculate energy of a configuration'''
    E = 0
    for i in range(1,N): # sum of neighbor products
        E -= S[i-1]*S[i] 
    return E

@numba.jit(nopython=True)
def energy_diff(S,N,i):
    '''Calculate energy difference of current config. S to proposed spin flip at index i'''
    l = 0 if i-1 < 0 else S[i-1]
    r = 0 if i+1 > N-1 else S[i+1]
    return 2*S[i]*(r+l) # l_pair_old+r_pair_old - (l_pair_new+r_pair_new), *_pair_new = -*_pair_old

@numba.jit(nopython=True)
def metropolis_step(S,N,T):
    '''Metropolis algorithm'''
    for _ in range(N):
        i = random.randrange(N) # must use randrange, numba can't handle np.random.choice
        dE = energy_diff(S,N,i) # calculate energy difference
        if random.random() < np.exp(-dE/T): # accept/reject
            S[i] = -S[i]
    return S

@numba.jit(nopython=True)
def monte_carlo(S,T,N,N_samples,N_therm):  
    '''Repeatedly call Metropolis-Hastings and calculate observables from samples'''
    for _ in range(N_therm):
        S = metropolis_step(S,N,T)     
    E = np.zeros(N_samples) # resulting configuration energies
    for i in range(N_samples):
        S = metropolis_step(S,N,T)
        E[i] = energy(S,N)
    return E

N_samples = 10000 # no. of samples after thermalization
N_therm = N_samples / 10 # no. of Metropolis algorithm calls for thermalization

Ts = np.arange(0.2, 4.2, 0.2) # temperature range
Us = np.zeros(len(Ts))
Cs = np.zeros(len(Ts))

plt.figure(figsize=(6,6))
observable = 'U' # used for easy plotting
for N_spins in [10,100,1000]:
    for i,T in enumerate(Ts):
        S_init = np.random.choice([-1,1], N_spins) # inital configuration
        E = monte_carlo(S_init,T,N_spins,N_samples,N_therm) # call monte carlo sampler
        Us[i] = np.mean(E) / N_spins # calculate average energy per spin
        Cs[i] = np.var(E) / N_spins / T**2  # calulate specific heat per spin

    # Plotting
    if observable=='U':
        U_th = -(N_spins-1)/N_spins * np.tanh(1/Ts)
        plt.plot(Ts,U_th,'--',label=r'Th., N=%d'%N_spins)
        plt.plot(Ts,Us,label=r"MC, N=%d"%N_spins)
    else:
        C_th = (N_spins-1)/N_spins * (Ts * np.cosh(1/Ts))**(-2)
        plt.plot(Ts,C_th,'--',label=r'Th., N=%d'%N_spins)
        plt.plot(Ts,Cs,label=r"MC, N=%d"%N_spins)

plt.ylim([-0.1,1.0])
plt.legend()
plt.xlabel(r'T')
plt.ylabel(r'%s/N' % observable)
plt.minorticks_on()
plt.grid(which='major', color='#CCCCCC', linestyle='--')
plt.grid(which='minor', color='#CCCCCC', linestyle=':')

fname = '1D_%s_%d' % (observable,N_samples)
plt.savefig('./report/src/%s' % fname, dpi=300)

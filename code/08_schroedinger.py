from common import *

def run():

    # Amount of 2-element pairs for block-update
    L_half = (L-1) // 2

    # Time steps for snapshots
    T = np.array([0,5,40,45,50])
    Ps = []

    for j, V in enumerate(Vs):

        # initial condition
        psi = np.zeros(L)
        psi = (2*np.pi*sigma**2)**(-1/4)*np.exp(1j*q*(l-l_0))*np.exp(-(l-l_0)**2/(4*sigma**2))

        P = []
        for i in range(m+1):

            # e^(-iTK2/2)e^(-iTK1/2)|psi>
            psi[:-1] = np.hstack([A @ v for v in np.split(psi[:-1], L_half)]).ravel()
            psi[1:] = np.hstack([A @ v for v in np.split(psi[1:], L_half)]).ravel()

            # e^(-iTV)|psi>
            psi = np.exp(-1j*tau*(delta**(-2) + V)) * psi

            # e^(-iTK1/2)e^(-iTK2/2)|psi>
            psi[1:] = np.hstack([A @ v for v in np.split(psi[1:], L_half)]).ravel()
            psi[:-1] = np.hstack([A @ v for v in np.split(psi[:-1], L_half)]).ravel()

            # Save |psi> for snapshot
            if i in T/tau:
                P.append(psi)
        Ps.append(P)
        
    return np.array(Ps)


import numpy as np
from qutip import Qobj, to_super, spre, spost, operator_to_vector, identity, liouvillian
from scipy.sparse.linalg import dsolve



def jump_ops(m_ops, method):

    if method == 'PD':
        L1 = [to_super(m_op) for m_op in m_ops]
        
    elif method == 'Homodyne':
         L1 = [spre(m_op) + spost(m_op.dag()) for m_op in m_ops]
    else:
        raise NameError("Check method. It should be 'PD' or 'Homodyne', not: {} ".format(method))

    return L1
    
def FCS_diffusion_matrix(H, c_ops, rho, m_ops, mu, method='PD'):
    
    # Get dimension of Hilbert space
    N = rho.shape[0]
    
    # Turn μ into numpy array
    mu = np.array(mu)
    
    # Initialise diffusion matrix D and Mαβ matrix 
    D = np.zeros((2, 2))
    
    # Vectorise density operator and identity
    rhovec = operator_to_vector(rho)
    Ivec = operator_to_vector(identity(N))
    
    # Compute Liouvillian super operator and ℒ1 measurement operator  
    L = liouvillian(H, c_ops)
    L1 = jump_ops(m_ops, method)
    
    # Compute average current
    Ja = np.array([np.real((Ivec.trans() * L1i * rhovec)[0,0]) for L1i in L1])
    
    # Compute Mαβ Matrix if n
    if method == 'Homodyne':
        M = mu@mu.T
    elif method == 'PD':
        M = mu@np.diag(Ja)@mu.T
    else:
        raise NameError("Check method. It should be 'PD' or 'Homodyne', not: {} ".format(method))
            
    # Compute ya
    ya = [(L1i - Ja[ix]) * rhovec for ix, L1i in enumerate(L1)]
    
    # If numbers dim is large, use sparse arrays:
    if N < 10:
        
        # Convert to numpy arrays
        Lf = L.full()
        Ivecf = Ivec.full()

        # create correct matrices to solve
        Ls = np.vstack((Lf, Ivecf.T))

        for ix, yf in enumerate(ya):
            # Convert to numpy arrays 
            yf = yf.full()
 
            yf = np.vstack((yf, [[0]]))
            beta = Qobj(np.linalg.lstsq(Ls, yf, rcond=None)[0], dims = [[[N], [N]], [1]])

            # Compute S(0) = D matrix
            for iy, L1i in enumerate(L1):

                D[ix, iy] = - np.real((Ivec.trans() * L1i * beta)[0,0])
        
    else:
        print("Using sparse Solver")
        
        # Get data 
        rhovecf = rhovec.data
        Lf = L.data
        Ivecf = Ivec.data
        
        for ix, yf in enumerate(ya):
            
            # Get yf data
            yf = yf.data
        
            # Compute β
            beta = dsolve.spsolve(Lf, yf, use_umfpack=False)
            beta = beta - rhovecf * (Ivecf.T * beta)
        
            for iy, L1i in enumerate(L1):
                # Now compute S(0) = D
                D[ix, iy] = - (np.real((Ivecf.T * L1i.data * beta)))[0]
    
    # return diffusion matrix
    return D + D.T + M

def TwoTimeCorrelationSS(H, t, c_ops, rho, m_ops, mu, method='PD'):
    
    # Get dimension of Hilbert space
    N = rho.shape[0]
    
    # Turn μ into numpy array
    mu = np.array(mu)
    rhovec = operator_to_vector(rho)
    Ivec = operator_to_vector(Qobj(identity(N), dims=rho.dims))

    L = liouvillian(H, c_ops)
    L1 = jump_ops(m_ops, method)

    # Compute measurement operators
    La = sum([mu[i]*L1[i] for i in range(len(m_ops))])

    # Compute average current
    Ja = np.array([np.real((Ivec.trans() * L1i * rhovec)[0,0]) for L1i in L1])
    
    # Compute two-time correlation function 
    Ft = np.array([np.real((Ivec.trans() * La * (L*ti).expm() * La * rhovec)[0,0]) for ti in t]) - np.sum(mu * Ja)**2

    return Ft

def FCSPowerSpectrumLinear(H, c_ops, rho, omega, m_ops, mu, method='PD'):
    
    # Get dimension of Hilbert space
    N = rho.shape[0]
    
    # Turn μ into numpy array
    mu = np.array(mu)
    
    # Vectorise density operator and identity
    rhovec = operator_to_vector(rho)
    Ivec = operator_to_vector(Qobj(identity(N), dims=rho.dims))
    
    # Compute Liouvillian super operator and ℒ1 measurement operator  
    L = liouvillian(H, c_ops)
    L1 = jump_ops(m_ops, method)
    
    # Compute average current
    Ja = np.array([np.real((Ivec.trans() * L1i * rhovec)[0,0]) for L1i in L1])
    
    # Compute Mαβ Matrix if n
    if method == 'Homodyne':
        M = mu@mu.T
    elif method == 'PD':
        M = mu@np.diag(Ja)@mu.T
    else:
        raise NameError("Check method. It should be 'PD' or 'Homodyne', not: {} ".format(method))
    
    # Define La
    La = sum([Li*mui for Li, mui in zip(L1, mu)])
    
    # Initialise S vector 
    S = np.zeros(len(omega))

    # define yf
    yf = La*rhovec
    
     # If numbers dim is large, use sparse arrays:
    if N < 10:
        
        # Compute frequency spectrum
        
        for i, omegai in enumerate(omega):
            v1 = Qobj(np.linalg.lstsq(L - 1j*omegai, yf, rcond=None)[0], dims = rhovec.dims)
            v2 = Qobj(np.linalg.lstsq(L + 1j*omegai, yf, rcond=None)[0], dims = rhovec.dims)
            
            S[i] = (M - np.real((Ivec.trans() * La*(v1 + v2))[0,0]))
            
    else:
        print("Using sparse Solver")
        
        # Get data
        Ivecf = Ivec.data
        yf = yf.data
        Lf = L.data
        Laf = La.data
        If = identity(N**2).data
        
        for i, omegai in enumerate(omega):
            
            v1 = dsolve.spsolve(Lf - 1j*omegai*If, yf, use_umfpack=False)
            v2 = dsolve.spsolve(Lf + 1j*omegai*If, yf, use_umfpack=False)
            
            spec = np.real((Ivecf.T * Laf * (v1 + v2)))[0]
            
            S[i] =  (M - spec)
 
    return S

def Partition(x, k, offset):
    "split array into subarrays of fixed length k"
    ix = int(np.floor((len(x)-k)/offset + 1))
    xs = np.zeros((ix, k))

    ki = 0
    i = 0
    while i < ix:
        xs[i,:] = x[ki:ki+k]

        ki += int(offset)
        i+=1

    return xs

def PowerSpectrum(x, dt, averaging=1, overlap=0.3):
    d = len(x)
    
    if averaging==1:
        k = d
    else:
        k = int(np.floor(2*d/(2*averaging - 1)))

    if k%2 != 0:
        k = k-1

    omegas = np.arange(1, k/2 +1)*(2*np.pi)/(k*dt)
    omegas = np.concatenate((-np.flip(omegas)[1:], omegas), axis=0)

    # Partition input signal into k steps 
    xs = Partition(x, k, int(np.floor(overlap*k)))
    xf = np.array([np.fft.fft(xi[:k] - np.mean(xi))/np.sqrt(k) for xi in xs])
    S = np.mean(np.array([dt*np.abs(xfi)**2 for xfi in xf]),axis=0)
    S = np.concatenate((S[int(k/2) +1 :k], S[1:int(k/2)+1]),axis=0)
    
    return omegas, S

def TwoTimeCorrelation(x, dt, averaging=1, overlap=0.3):
    d = len(x)
    
    if averaging==1:
        k = d
    else:
        k = int(np.floor(2*d/(2*averaging - 1)))

    if k%2 != 0:
        k = k-1

    xs = Partition(x, k, int(np.floor(overlap*k)))
    xf = np.array([xi[:k] - np.mean(xi) for xi in xs])
    Ft = np.mean(np.array([dt*np.abs(xfi)**2 for xfi in xf]),axis=0)
    
    return xs, Ft


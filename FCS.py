import numpy as np
from qutip import *
import scipy as sp
from sklearn.preprocessing import normalize

def jump_ops(m_ops, method):
    
    if method == 'PD':
        ℒ1 = [to_super(m_op) for m_op in m_ops]
        
    elif method == 'Homodyne':
         ℒ1 = [spre(m_op) + spost(m_op.dag()) for m_op in m_ops]
    
    return ℒ1
    
def FCS_diffusion_matrix(H, c_ops, ρ, m_ops, μ, method='PD'):
    
    # Get dimension of Hilbert space
    N = ρ.shape[0]
    
    # Turn μ into numpy array
    μ = np.array(μ)
    
    # Initialise diffusion matrix D and Mαβ matrix 
    D = np.zeros((2, 2))
    
    # Vectorise density operator and identity
    ρvec = operator_to_vector(ρ)
    Ivec = operator_to_vector(identity(N))
    
    # Compute Liouvillian super operator and ℒ1 measurement operator  
    ℒ = liouvillian(H, c_ops)
    ℒ1 = jump_ops(m_ops, method)
    
    # Compute average current
    Jα = np.array([np.real((Ivec.trans() * ℒ1i * ρvec)[0,0]) for ℒ1i in ℒ1])
    
    # Compute Mαβ Matrix if n
    if method == 'PD':
        M = μ@np.diag(Jα)@μ.T
        
    elif method == 'Homodyne':
        M = μ@μ.T
            
    # Compute yα
    yα = [(ℒ1i - Jα[ix]) * ρvec for ix, ℒ1i in enumerate(ℒ1)]
    
    # If numbers dim is large, use sparse arrays:
    if N < 10:
        
        # Convert to numpy arrays
        ℒf = ℒ.full()
        Ivecf = Ivec.full()

        # create correct matrices to solve
        ℒs = np.vstack((ℒf, Ivecf.T))

        for ix, yf in enumerate(yα):
            # Convert to numpy arrays 
            yf = yf.full()
 
            yf = np.vstack((yf, [[0]]))
            β = Qobj(np.linalg.lstsq(ℒs, yf, rcond=None)[0], dims = [[[N], [N]], [1]])

            # Compute S(0) = D matrix
            for iy, ℒ1i in enumerate(ℒ1):

                D[ix, iy] = - np.real((Ivec.trans() * ℒ1i * β)[0,0])
        
    else:
        print("Using sparse Solver")
        
        # Get data 
        ρvecf = ρvec.data
        ℒf = ℒ.data
        Ivecf = operator_to_vector(I).data
        
        for ix, yf in enumerate(yα):
            
            # Get yf data
            yf = yf.data
        
            # Compute β
            β = sp.sparse.linalg.dsolve.spsolve(ℒf, yf, use_umfpack=False)
            β = β - ρvecf * (Ivecf.T * β)
        
            for iy, ℒ1i in enumerate(ℒ1):
                # Now compute S(0) = D
                D[ix, iy] = - (np.real((Ivecf.T * ℒ1i.data * β)))[0]
    
    # return diffusion matrix
    return D + D.T + M



def TwoTimeCorrelationSS(H, t, c_ops, ρ, m_ops, μ, method='PD'):
    
    # Get dimension of Hilbert space
    N = ρ.shape[0]
    
    # Turn μ into numpy array
    μ = np.array(μ)
    ρvec = operator_to_vector(ρ)
    Ivec = operator_to_vector(Qobj(identity(N), dims=ρ.dims))

    ℒ = liouvillian(H, c_ops)
    ℒ1 = jump_ops(m_ops, method)

    # Compute measurement operators
    Lα = sum([μ[i]*ℒ1[i] for i in range(len(m_ops))])

    # Compute average current
    Jα = np.array([np.real((Ivec.trans() * ℒ1i * ρvec)[0,0]) for ℒ1i in ℒ1])
    
    # Compute two-time correlation function 
    Ft = np.array([np.real((Ivec.trans() * Lα * (ℒ*ti).expm() * Lα * ρvec)[0,0]) for ti in t]) - np.sum(μ * Jα)**2

    return Ft

def FCSPowerSpectrumLinear(H, c_ops, ρ, ω, m_ops, μ, method='PD'):
    
    # Get dimension of Hilbert space
    N = ρ.shape[0]
    
    # Turn μ into numpy array
    μ = np.array(μ)
    
    # Vectorise density operator and identity
    ρvec = operator_to_vector(ρ)
    Ivec = operator_to_vector(Qobj(identity(N), dims=ρ.dims))
    
    # Compute Liouvillian super operator and ℒ1 measurement operator  
    ℒ = liouvillian(H, c_ops)
    ℒ1 = jump_ops(m_ops, method)
    
    # Compute average current
    Jα = np.array([np.real((Ivec.trans() * ℒ1[i] * ρvec)[0,0]) for i in range(len(ℒ1))])
    
    # Compute Mαβ Matrix if n
    if method == 'PD':
        M = μ@np.diag(Jα)@μ.T
        
    elif method == 'Homodyne':
        M = μ@μ.T
    
    # Define ℒα
    ℒα = sum([L*μi for L, μi in zip(ℒ1, μ)])
    
    # Initialise S vector 
    S = np.zeros(len(ω))

    # define yf
    yf = ℒα*ρvec
    
     # If numbers dim is large, use sparse arrays:
    if N < 10:
        
        # Compute frequency spectrum
        
        for i, ωi in enumerate(ω):
            v1 = Qobj(np.linalg.lstsq(ℒ - 1j*ωi, yf, rcond=None)[0], dims = ρvec.dims)
            v2 = Qobj(np.linalg.lstsq(ℒ + 1j*ωi, yf, rcond=None)[0], dims = ρvec.dims)
            
            S[i] = (M - np.real((Ivec.trans() * ℒα*(v1 + v2))[0,0]))
            
    else:
        print("Using sparse Solver")
        
        # Get data 
        ρvecf = ρvec.data
        Ivecf = Ivec.data
        yf = yf.data
        ℒf = ℒ.data
        ℒαf = ℒα.data
        If = identity(N**2).data
        
        for i, ωi in enumerate(ω):
            
            v1 = sp.sparse.linalg.dsolve.spsolve(ℒf - 1j*ωi*If, yf, use_umfpack=False)
            v2 = sp.sparse.linalg.dsolve.spsolve(ℒf + 1j*ωi*If, yf, use_umfpack=False)
            
            spec = np.real((Ivecf.T * ℒαf * (v1 + v2)))[0]
            
            S[i] =  (M - spec)
 
    return S

def Diffusion_Gernot(H, c_ops, ρ, m_ops, μ, method="PD"):
    
    ## To DO: Figure out Homodyne detection
    ## Add sparse solver
    ## Only works for counting statistics 
    
    N = ρ.shape[0]
    
    ρvec = operator_to_vector(ρ)
    Ivec = operator_to_vector(Qobj(identity(N), dims=ρ.dims))
    
    # Define liouvillian
    ℒ = liouvillian(H, c_ops)
    ℒops = jump_ops(m_ops, method)
    
    # Create superoperators 
    ℒ1 = 1j*sum([L*μi for L, μi in zip(ℒops, μ)])
    ℒ2 = -1*sum([L*μi**2 for L, μi in zip(ℒops, μ)])
    
    # Convert to numpy arrays
    ℒf = ℒ.full()
    Ivecf = Ivec.full()
    Ibar = -1j*(Ivec.trans()*ℒ1*ρvec).full()[0,0]
    
    # create correct matrices to solve
    ℒs = np.vstack((ℒf, Ivecf.T))
    yf = ((1j*Ibar - ℒ1)*ρvec).full()
    yf = np.vstack((yf, [[0]]))
    y0 = np.vstack((np.zeros((N**2,1)),[[1]]))

    # Solve Gernot's equation
    ρv = Qobj(np.linalg.lstsq(ℒs, y0, rcond=None)[0], dims = ρvec.dims)
    σ = Qobj(np.linalg.lstsq(ℒs, yf, rcond=None)[0], dims = ρvec.dims)


    D = -(np.real((Ivec.trans()* (ℒ2 * ρvec + 2*ℒ1 * σ)).full()[0,0]))
    
    return D

def PowerSpectrum(x, dt, averaging=1, overlap=0.3):
    d = len(x)
    
    if averaging==1:
        k = d
    else:
        k = int(np.floor(2*d/(2*averaging - 1)))

    if k%2 != 0:
        k = k-1

    ωs = np.arange(1, k/2 +1)*(2*np.pi)/(k*dt)
    ωs = np.concatenate((-np.flip(ωs)[1:], ωs), axis=0)

    # Define partition function like Mathematics
    def Partition(x, n, offset):
        ix = int(np.floor((len(x)-k)/offset + 1))
        xs = np.zeros((ix, k))

        ki = 0
        i = 0
        while i < ix:
            xs[i,:] = x[ki:ki+k]

            ki += int(offset)
            i+=1

        return xs

    xs = Partition(x, k, int(np.floor(overlap*k)))
    xf = np.array([np.fft.fft(xi[:k] - np.mean(xi))/np.sqrt(k) for xi in xs])
    S = np.mean(np.array([dt*np.abs(xfi)**2 for xfi in xf]),axis=0)
    S = np.concatenate((S[int(k/2) +1 :k], S[1:int(k/2)+1]),axis=0)
    
    return ωs, S

def CrossPowerSpectrum(x, y, dt, averaging=1, overlap=0.3):
    d = len(x)
    
    if averaging==1:
        k = d
    else:
        k = int(np.floor(2*d/(2*averaging - 1)))

    if k%2 != 0:
        k = k-1

    ωs = np.arange(1, k/2 +1)*(2*np.pi)/(k*dt)
    ωs = np.concatenate((-np.flip(ωs)[1:], ωs), axis=0)

    # Define partition function like Mathematics
    def Partition(x, n, offset):
        ix = int(np.floor((len(x)-k)/offset + 1))
        xs = np.zeros((ix, k))

        ki = 0
        i = 0
        while i < ix:
            xs[i,:] = x[ki:ki+k]

            ki += int(offset)
            i+=1

        return xs

    xs = Partition(x, k, int(np.floor(overlap*k)))
    ys = Partition(y, k, int(np.floor(overlap*k)))
    xf = np.array([np.fft.fft(xi[:k] - np.mean(xi))/np.sqrt(k) for xi in xs])
    yf = np.array([np.fft.fft(yi[:k] - np.mean(yi))/np.sqrt(k) for yi in ys])
    
    
    S = np.mean(np.array([dt*np.abs(yf[i] * xf[i]) for i in range(len(xf))]),axis=0)
    S = np.concatenate((S[int(k/2) +1 :k], S[1:int(k/2)+1]),axis=0)
    
    return ωs, S

def TwoTimeCorrelation(x, dt, averaging=1, overlap=0.3):
    d = len(x)
    
    if averaging==1:
        k = d
    else:
        k = int(np.floor(2*d/(2*averaging - 1)))

    if k%2 != 0:
        k = k-1


    # Define partition function like Mathematics
    def Partition(x, n, offset):
        ix = int(np.floor((len(x)-k)/offset + 1))
        xs = np.zeros((ix, k))

        ki = 0
        i = 0
        while i < ix:
            xs[i,:] = x[ki:ki+k]

            ki += int(offset)
            i+=1

        return xs

    xs = Partition(x, k, int(np.floor(overlap*k)))
    xf = np.array([xi[:k] - np.mean(xi) for xi in xs])
    Ft = np.mean(np.array([dt*np.abs(xfi)**2 for xfi in xf]),axis=0)
    
    return ωs, S


# def FCSPowerSpectrumEigen(H, c_ops, ρ, ω, m_ops, μ, method='PD'):
    
#     # Get dimension of Hilbert space
#     N = ρ.shape[0]
    
#     # Turn μ into numpy array
#     μ = np.array(μ)
    
#     # Vectorise density operator and identity
#     ρvec = operator_to_vector(ρ)
#     Ivec = operator_to_vector(identity(N))
    
#     # Compute Liouvillian super operator and ℒ1 measurement operator  
#     ℒ = liouvillian(H, c_ops)
#     ℒ1 = jump_ops(m_ops, method)
    
#     # Compute average current
#     Jα = np.array([np.real((Ivec.trans() * ℒ1i * ρvec)[0,0]) for ℒ1i in ℒ1])
    
#     # Compute Mαβ Matrix if n
#     if method == 'PD':
#         M = μ@np.diag(Jα)@μ.T
        
#     elif method == 'Homodyne':
#         M = μ@μ.T
        
#     # Define ℒα
#     ℒα = sum([L*μi for L, μi in zip(ℒ1, μ)])
    
#     # Get eigenstates and eigenvalues
#     λ, Q = ℒ.eigenstates()

#     # drop the steady state 
#     minval, idx = min([(abs(val), idx) for (idx, val) in enumerate(λ)])
#     λ = np.delete(λ, idx)

#     # Compute |xi>
#     x = np.delete(Q, idx)
    
#     Qt = np.array([q for q in Q])
#     y = np.delete(np.linalg.inv(Qt.T).T, idx, axis=0)
#     y = [Qobj(yi, dims=Q[0].dims, type=Q[0].type) for yi in y]

#     # compute final terms
#     Υ = [(Ivec.dag() * ℒα * x[i]*y[i].dag() * ℒα * ρvec)[0,0] for i in range(N**2-1)]
    
#     Sω = M - sum([(1/(λ[i] - 1j*ω)  + 1/(λ[i] + 1j*ω)) * Υ[i] for i in range(N**2-1)])

#     return Sω

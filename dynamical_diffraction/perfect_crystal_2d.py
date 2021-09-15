import numpy as np

def laue_fixed_length(E_0, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):

    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_0))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 + alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j*np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = -1j*k/2/np.cos(alpha_h)*C*chi_h
    A_hh = -1j*k/2/np.cos(alpha_h)*(chi_0+beta) - 1j*np.tan(alpha_h)*q*2*np.pi

    # Eigenvalues for each decpuples 2x2 problem
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors
    v1 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Transmission coeff of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Transform initial condition
    ff = np.fft.fft(E_0)

    # Transmission and reflection
    E_0 = np.fft.ifft((v1*t1 - v2*t2)/(v1 - v2) * ff)
    E_h = np.fft.ifft((t1 - t2)/(v1 - v2) * ff)

    return E_0, E_h
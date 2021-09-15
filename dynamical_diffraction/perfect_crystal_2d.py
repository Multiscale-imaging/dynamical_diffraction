import numpy as np

def laue_fixed_length(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):

    ''' Perfect crystal propagator in the Laue case for a fixed rocking angle and crystal thickness.

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        phi (optional, float): rocking angle

    Returns:
        E_0 (N by 1 complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by 1 complex numpy array): Complex real space amplitudes of scattered beam.
    '''


    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
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
    ff = np.fft.fft(E_init)

    # Transmission and reflection
    E_0 = np.fft.ifft((v1*t1 - v2*t2)/(v1 - v2) * ff)
    E_h = np.fft.ifft((t1 - t2)/(v1 - v2) * ff)

    return E_0, E_h


def laue_fixed_rockingcurve(E_init, phi, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1):

    ''' Perfect crystal propagator in the Laue case for a fixed rocking angle and crystal thickness.

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        phi (M by 1, float array): rocking angle
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        
    Returns:
        E_0 (N by M complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by M complex numpy array): Complex real space amplitudes of scattered beam.
    '''


    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j*np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = -1j*k/2/np.cos(alpha_h)*C*chi_h
    A_hh = -1j*k/2/np.cos(alpha_h)*(chi_0+beta[np.newaxis,:]) - 1j*np.tan(alpha_h)*q[:, np.newaxis]*2*np.pi

    # Eigenvalues for each decpuples 2x2 problem
    squareroot_term = np.sqrt( A_00[:, np.newaxis]**2 + A_hh**2 - 2*A_00[:, np.newaxis]*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00[:, np.newaxis] + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00[:, np.newaxis] + A_hh )

    # Eigenvectors
    v1 = -(-A_00[:, np.newaxis] + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00[:, np.newaxis] + A_hh - squareroot_term) / 2 / A_h0

    # Transmission coeff of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Transform initial condition
    ff = np.fft.fft(E_init)

    # Transmission and reflection
    E_0 = np.fft.ifft((v1*t1 - v2*t2)/(v1 - v2) * ff[:, np.newaxis], axis = 0)
    E_h = np.fft.ifft((t1 - t2)/(v1 - v2) * ff[:, np.newaxis], axis = 0)

    return E_0, E_h
import numpy as np

def bragg_inf(phi, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1):
    ''' Calculate and return the complex reflectivity for an semi-infinite
     crystal in Bragg-geometry.
    Parameters:
        phi (numpy array): rocking angle
        alpha_0 (real float): glancing angle of incident radiation in radians IMPORTANTLY THIS IS NOT THE ANGLE OF INCIDENCE
        beta_h (real float): glancing angle of reflection in radians IMPORTANTLY THIS IS NOT THE ANGLE OF INCIDENCE
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (real float): Polarization factor
    '''

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    twotheta = alpha_0 + alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j/2/np.sin(alpha_0)*chi_0
    A_0h = -1j/2/np.sin(alpha_0)*C*chi_hm
    A_h0 = 1j/2/np.sin(alpha_h)*C*chi_h
    A_hh = 1j/2/np.sin(alpha_h)*(chi_0+beta)

    # Check which eigenvalue is negative
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    sign = -np.sign(np.real(eigval_1))

    # Find components of the corresponding eigenvector
    R =  2 * A_h0 / (-A_00 + A_hh + sign * squareroot_term)

    return R

def bragg_finite(phi, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1):
    ''' Calculate and return the complex reflectivity and transmission for a plate-like crystal in
     crystal in Bragg-geometry.
    Parameters:
        phi (numpy array): rocking angle
        L (real float): Thickness of crystal. Same unit as wavelength
        lmbd (real float): wavelength. Same unit as thickness.
        alpha_0 (real float): glancing angle of incident radiation in radians IMPORTANTLY THIS IS NOT THE ANGLE OF INCIDENCE
        beta_h (real float): glancing angle of reflection in radians IMPORTANTLY THIS IS NOT THE ANGLE OF INCIDENCE
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (real float): Polarization factor
    '''

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 + alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.sin(alpha_0)*chi_0
    A_0h = -1j*k/2/np.sin(alpha_0)*C*chi_hm
    A_h0 = 1j*k/2/np.sin(alpha_h)*C*chi_h
    A_hh = 1j*k/2/np.sin(alpha_h)*(chi_0+beta)

    # Eigenvalues
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors
    v1 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Transmission of of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Transmission coeff of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Calculate reflection and transmission
    R = (1 -t1 / t2) / (v1+ -t1 / t2*v2)
    T = (v1*t1 -t1 *v2) / (v1-t1 / t2*v2)

    return R, T

def laue_rockingcurve(phi, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1):
    ''' Calculate and return the complex reflectivity for a plate-like crystal in
     crystal in Bragg-geometry.
    Parameters:
        phi (numpy array): rocking angle
        L (real float): Thickness of crystal. Same unit as wavelength
        lmbd (real float): wavelength. Same unit as thickness.
        alpha_0 (real float): angle of incidence of the incident radiation in radians 
        beta_h (real float): angle of incidence of the reflection in radians
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (real float): Polarization factor
    '''

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 + alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = 1j*k/2/np.cos(alpha_0)*chi_0
    A_0h = 1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = 1j*k/2/np.cos(alpha_h)*C*chi_h
    A_hh = 1j*k/2/np.cos(alpha_h)*(chi_0+beta)

    # Eigenvalues
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors
    v1 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Transmission coeff of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Transmission and reflection
    T = (v1*t1 - v2*t2)/(v1 - v2)
    R = (t1 - t2)/(v1 - v2)

    return R, T

def laue_thickness(L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1):
    ''' Calculate and return the complex reflectivity for a plate-like crystal in
     crystal in Bragg-geometry at zero rocking angle (which might not be the peak of the reflectivity curve) as a function of the crystal length. 
    Parameters:
        L (numpy array): Thickness of crystal. Same unit as wavelength
        lmbd (real float): wavelength. Same unit as thickness.
        alpha_0 (real float): angle of incidence of the incident radiation in radians 
        beta_h (real float): angle of incidence of the reflection in radians
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (real float): Polarization factor
    '''

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    A_00 = 1j*k/2/np.cos(alpha_0)*chi_0
    A_0h = 1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = 1j*k/2/np.cos(alpha_h)*C*chi_h
    A_hh = 1j*k/2/np.cos(alpha_h)*chi_0

    # Eigenvalues
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors
    v1 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Transmission coeff of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Transmission and reflection
    T = (v1*t1 - v2*t2)/(v1 - v2)
    R = (t1 - t2)/(v1 - v2)

    return R, T
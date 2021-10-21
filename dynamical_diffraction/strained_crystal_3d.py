import numpy as np
import matplotlib.pyplot as plt
import tqdm

def laue_exponential_heun(E_init, u, stepsizes, gridshape, lmbd, k_0, k_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0, u_type = 'array'):

    ''' Strained crystal finite difference integrator in the Laue case for a fixed rocking angle and crystal thickness.
    The finite difference is "exponential euler" descibed in https://arxiv.org/abs/2106.12412

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        u (function): Function of x (array) and z(contant) that returns the scalar displacement field
                      in the direction of Q. Given in the same real space units as other inputs.
        del_z (float): Step size in thickness dimension.              
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction. Number of steps is M = floor(L/del_z).
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        phi (optional, float): rocking angle

    Returns:
        E_0 (N by M complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by M complex numpy array): Complex real space amplitudes of scattered beam.
    '''

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h
    
    # Build frequency arrays
    qx = np.fft.fftfreq(gridshape[0])/stepsizes[0]
    qy = np.fft.fftfreq(gridshape[1])/stepsizes[1]
    qx = qx[:, np.newaxis]
    qy = qy[np.newaxis, :]

    # Reciprocal space geometry. Uglier than it needs to be, but better than rewriting everything
    k = 2*np.pi / lmbd
    twotheta = np.arccos(np.dot(k_0, k_h)/np.linalg.norm(k_0)/np.linalg.norm(k_h))
    beta = 2*np.sin(twotheta)*phi
    Q = np.linalg.norm(k_h - k_0)

    psi0 = np.arctan2(k_0[1], k_0[0])
    alpha0 = np.arccos(k_0[2]/k) 
    q_parallel0 = (qx*np.cos(psi0)  + qy*np.sin(psi0))
    q_normal0 = (-qx*np.sin(psi0)  + qy*np.cos(psi0))

    psih = np.arctan2(k_h[1], k_h[0])
    alphah = np.arccos(k_h[2]/k)
    q_parallelh = (qx*np.cos(psih)  + qy*np.sin(psih))
    q_normalh = (-qx*np.sin(psih)  + qy*np.cos(psih))

    # Precalculate some trig. for prettier code later
    ct0 = np.cos(alpha0)
    cth = np.cos(alphah)
    tt0 = np.tan(alpha0)
    tth = np.tan(alphah)
    st0 = np.sin(alpha0)
    sth = np.sin(alphah)

    # transform initial condition
    E0_f = np.fft.fft2(E_init)

    # Build coefficient arrays (equations are derived in paper)
    #A0 = -1j / 2 / (ct0 * k + tt0 * q_parallel0) * (k**2*chi0 + 4 * np.pi * st0 * q_parallel0 * k + q_normal0**2 + (1-tt0**2) * q_parallel0**2)
    #Ah = -1j / 2 / (cth * k + tth * q_parallelh) * (k**2*chi0 + 4 * np.pi * sth * q_parallelh * k + q_normalh**2 + (1-tth**2) * q_parallelh**2)
    A0 = -1j/2/ct0 * (k*(chi_0) + 4 * np.pi*st0*q_parallel0 )
    Ah = -1j/2/cth * (k*(chi_0+beta) + 4 * np.pi*sth*q_parallelh )
    A = np.stack((A0, Ah), axis = 2)
    A = A.flatten()

    #B0 = -1j * k**2 / 2 / (ct0 * k + tt0 * 2* np.pi * q_parallel0)
    #Bh = -1j * k**2 / 2 / (cth * k + tth * 2* np.pi * q_parallelh)
    B0 = -1j * k / 2 / ct0 *np.ones(A0.shape)
    Bh = -1j * k / 2 / cth *np.ones(A0.shape)
    B = np.stack((B0, Bh), axis = 2)
    B = B.flatten()
    
    # Define sub function
    def B_fun(E, chih_slice, chihm_slice):

        # Perform fourier convolution
        arr = np.fft.ifft2(E.reshape((*gridshape[:2], 2)), axes = (0,1))
        arr = arr * np.stack((chih_slice, chihm_slice), axis = -1)
        arr = np.fft.fft2(arr, axes = (0,1))

        # Switch E_0 and E_h parts
        arr = np.flip(arr, axis=2)

        # Multiply by coefficient arrays
        arr[:,:,0] = B0 * arr[:,:,0]
        arr[:,:,1] = Bh * arr[:,:,1]

        # return flattened array
        return arr.flatten()

    # Pre-calculate exponentials
    h = stepsizes[2]
    phi0 = np.exp(h*A)
    phi1 = np.nan_to_num(1/h/A * (phi0 - 1))
    phi2 = np.nan_to_num(2/h/A * (phi1 - 1))

    # Prepare initial condition

    E_running = np.zeros((*gridshape[:2], 2), dtype=complex)
    E_running[..., 0] = E_init
    E_running = np.fft.fft2(E_running, axes=(0,1)).flatten()


    iz = 0
    if u_type == 'array':
        u_iz = u[:,:,iz]
        # Multiply by susceptibilities
        chih_slice = chi_h*np.exp(1j*u_iz*Q )
        chihm_slice = chi_hm*np.exp(-1j*u_iz*Q )


    ############           INTEGRATION     ##################
    for iz in tqdm.tqdm(range(gridshape[2]-1)): # loop over z slices

        # if iz%1 == 0:
        #     print(iz)

        # Do the first part of the finite difference step
        E1 = np.array(E_running)
        g1 = B_fun(E1, chih_slice, chihm_slice)
        E2 = phi0*E1 + h * phi1 * g1

        # Read/calculate the next slice in the u function
        u_iz = u[:,:,iz]
        chih_slice = chi_h*np.exp(1j*u_iz*Q )
        chihm_slice = chi_hm*np.exp(-1j*u_iz*Q )

        # Do the second part of the finite difference step
        g2 = B_fun(E2, chi_h, chi_hm)
        E_running = phi0*E1 + h/2 * ((2*phi1-phi2)*g1 + phi2*g2)

    # Transform output back
    Eh_out = np.fft.ifft2(E_running.reshape((*gridshape[:2], 2)), axes = (0,1))[:,:,1]
    E0_out = np.fft.ifft2(E_running.reshape((*gridshape[:2], 2)), axes = (0,1))[:,:,0]
    
    return E0_out, Eh_out

import numpy as np


def laue_exponential_heun(E_init, u, del_z, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):


    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Calculate useful quantitites
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    q_magnitude = np.abs(2*k*np.sin(twotheta/2))
    beta = 2*np.sin(twotheta)*phi
    Nz = int(L//del_z)
    Nx = len(E_init)


    # Build coordinate arrays
    x = np.arange(Nx) * del_x
    qx = np.fft.fftfreq(Nx)/del_x # Full period frequency

    # Calculate some geometrical factors
    ct0 = np.cos(alpha_0)
    cth = np.cos(alpha_h)
    tt0 = np.tan(alpha_0)
    tth = np.tan(alpha_h)

    # Build coefficient arrays (equations are derived in paper)
    A_00 = -1j*k*chi_0/2/ct0 + 1j*tt0*qx*2*np.pi
    A_hh = -1j*k*(chi_0+beta)/2/cth + 1j*tth*qx*2*np.pi
    A = np.append(A_00, A_hh)

    B0 = -1j * k**2 / 2 / (ct0 * k)*np.ones(Nx)
    Bh = -1j * k**2 / 2 / (cth * k)*np.ones(Nx)

    ################  Define B function #####################
    def B_fun(iz, E):

        # load relevant slice
        u_iz = u(x, del_z*iz) 

        # Multiply by susceptibilities
        chih = chi_h*np.exp(1j*u_iz*q_magnitude )
        chihm = chi_hm*np.exp(-1j*u_iz*q_magnitude )

        # Perform fourier convolution
        arr = np.fft.ifft(E.reshape((2, Nx)), axis = 1)
        arr = arr * np.stack((chih, chihm), axis = 0)
        arr = np.fft.fft(arr.reshape((2, Nx)), axis = 1)

        # Switch E_0 and E_h parts
        arr = np.flip(arr, axis=0)

        # Multiply by coefficient arrays
        arr[0,:] = B0 * arr[0,:]
        arr[1,:] = Bh * arr[1,:]

        # return flattened array
        return arr.flatten()


    # Output arrays
    E0 = np.zeros((Nz,Nx), dtype=complex)
    Eh = np.zeros((Nz,Nx), dtype=complex)

    ########## Define step function ###############

    # Pre-calculate exponentials
    phi0 = np.exp(del_z*A)
    phi1 = 1/del_z/A * (phi0 - 1)
    phi2 = 2/del_z/A * (phi1 - 1)

    def SingleStep(iz, E):
        E1 = E.flatten()
        g1 = B_fun(iz, E)
        E2 = phi0*E1 + del_z * phi1 * g1
        g2 = B_fun(iz+1,E2)
        arr = phi0*E1 + del_z/2 * ((2*phi1-phi2)*g1 + phi2*g2)
        return arr

    # Input initial condition
    E0[0,:] = E_init
    # Transform initial condition
    E = np.append( np.fft.fft(E0[0,:]), np.fft.fft(Eh[0,:]) ) 

    ############           INTEGRATION     ##################
    for ii in range(Nz-1): # loop over z slices
        # Do step
        E = SingleStep(ii, E )

        # Assign to ouput arrays 
        E0[ii+1,:] = np.fft.ifft(E[:Nx])
        Eh[ii+1,:] = np.fft.ifft(E[Nx:])

    return E0.transpose(), Eh.transpose()

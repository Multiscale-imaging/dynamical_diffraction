import numpy as np
import matplotlib.pyplot as plt
import dynamical_diffraction.perfect_crystal_2d as perf_2d

# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 + 1j*1e-8
chi_h = 1.88e-6 - 1j*1e-8
alpha_0 = -10*np.pi/180
alpha_h = 10*np.pi/180
C = 1
lmbd = 0.71*1e-7
L = 300*1e-3
phi = np.linspace(-4e-5, 4e-5, 5000)
mu = lmbd/np.pi/np.imag(chi_0)
x = np.linspace(0, 110*1e-3, 512)
del_x = x[1] - x[0]
x_mid = 55*1e-3
width = 0.5*1e-3
E_init = np.exp(-(x-x_mid)**2/2/width**2).astype(complex)

E_0, E_h = perf_2d.laue_fixed_length(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0)

fig = plt.figure(figsize = (15, 5))
plt.subplot(1,3,1)
plt.plot(x, np.abs(E_init)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$E_0(x,0)|^2$', fontsize = 15)
plt.title('Initial condition',fontsize = 20)

plt.subplot(1,3,2)
plt.plot(x, np.abs(E_0)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$E_0(x,0)|^2$', fontsize = 15)
plt.title('Transmitted beam',fontsize = 20)

plt.subplot(1,3,3)
plt.plot(x, np.abs(E_h)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$E_h(x,L)|^2$', fontsize = 15)
plt.title('Scattered beam',fontsize = 20)

fig.tight_layout()  
plt.show()

plt.show()

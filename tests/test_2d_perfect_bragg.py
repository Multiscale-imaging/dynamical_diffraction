import numpy as np
import matplotlib.pyplot as plt
import dynamical_diffraction.perfect_crystal_2d as perf_2d

# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 - 1j*1e-8
chi_h = 1.88e-6 - 1j*1e-8
alpha_0 = 10*np.pi/180
alpha_h = 5*np.pi/180
C = 1
lmbd = 0.71*1e-7
mu = lmbd/np.pi/np.imag(chi_0)
x = np.linspace(0, 400*1e-3, 1024)
del_x = x[1] - x[0]
x_mid = 50*1e-3
width = 1*1e-3
E_init = np.exp(-(x-x_mid)**2/2/width**2).astype(complex)

L = 10*1e-3
E_0, E_h = perf_2d.bragg_finite(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, phi = 0)

fig = plt.figure(figsize = (10, 6))
plt.subplot(2,3,1)
plt.plot(x*1e3, np.abs(E_init)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$|E_0(x,0)|^2$', fontsize = 15)
plt.title('Initial condition',fontsize = 20)

plt.subplot(2,3,2)
plt.plot(x*1e3, np.abs(E_0)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$|E_0(x,L)|^2$', fontsize = 15)
plt.title('Transmitted',fontsize = 20)


plt.subplot(2,3,3)
plt.plot(x*1e3, np.abs(E_h)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$|E_h(x,0)|^2$', fontsize = 15)
plt.title('Scattered beam',fontsize = 20)


E_h = perf_2d.bragg_inf(E_init, del_x, lmbd, alpha_0, alpha_h, chi_0, chi_h, phi = 0)


plt.subplot(2,3,4)
plt.plot(x*1e3, np.abs(E_init)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$|E_0(x,0)|^2$', fontsize = 15)
plt.title('Initial condition',fontsize = 20)

plt.subplot(2,3,6)
plt.plot(x*1e3, np.abs(E_h)**2)
plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$|E_h(x,0)|^2$', fontsize = 15)
plt.title('Scattered beam',fontsize = 20)

fig.tight_layout()  
plt.show()

E0_list = []
Eh_list = []

M = 300
phi_list = np.linspace(-1*1e-3, 1*1e-3, M)

for phi in phi_list:
    E_0, E_h = perf_2d.bragg_finite(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, phi = phi)
    E0_list.append(E_0)
    Eh_list.append(E_h)

fig = plt.figure(figsize = (6,6))


x_axis_ticks = np.array([0, 100, 200, 300, 400]) # microns
phi_axis_ticks = np.array([-1000, -500, 00, 500, 1000]) # micro radians
del_phi = phi_list[1] - phi_list[0]


plt.subplot(2,1,1)
plt.imshow(np.abs(np.stack(E0_list))**2)
plt.xticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.xlabel(r'$x (\mathrm{\mu m})$', fontsize = 15)
plt.yticks(phi_axis_ticks*1e-6 / del_phi + M/2, phi_axis_ticks, fontsize = 12)
plt.ylabel(r'$\phi (\mathrm{\mu rad})$', fontsize = 15)
plt.colorbar()
plt.title('Transmitted beam',fontsize = 20)

plt.subplot(2,1,2)
plt.imshow(np.abs(np.stack(Eh_list))**2)
plt.colorbar()
plt.xticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.xlabel(r'$x (\mathrm{\mu m})$', fontsize = 15)
plt.yticks(phi_axis_ticks*1e-6 / del_phi + M/2, phi_axis_ticks, fontsize = 12)
plt.ylabel(r'$\phi (\mathrm{\mu rad})$', fontsize = 15)
plt.title('Scattered beam',fontsize = 20)
fig.tight_layout()  
plt.show()
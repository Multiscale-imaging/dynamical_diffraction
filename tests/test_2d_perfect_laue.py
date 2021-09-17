import numpy as np
import matplotlib.pyplot as plt
import dynamical_diffraction.perfect_crystal_2d as perf_2d

# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 - 1j*1e-8
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

E_0, E_h = perf_2d.laue(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 20e-6)

fig1 = plt.figure(figsize = (15, 5))
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

fig1.tight_layout()  


M = 500
phi_range = 100e-6
phi = np.linspace(-phi_range, phi_range, M) 
E_0, E_h = perf_2d.laue_rockingcurve(E_init, phi, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1)

fig2 = plt.figure(figsize = (10, 4))
x_axis_ticks = np.array([0, 20, 40, 60, 80, 100]) # microns
phi_axis_ticks = np.array([-100, -50, 0, 50, 100]) # micro radians
del_phi = phi[1] - phi[0]

plt.subplot(1,3,1)
plt.imshow(np.abs(E_0))
plt.title('Transmitted beam',fontsize = 20)
plt.yticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.ylabel(r'$x (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(phi_axis_ticks*1e-6 / del_phi + M/2, phi_axis_ticks, fontsize = 12)
plt.xlabel(r'$\phi (\mathrm{\mu rad})$', fontsize = 15)

plt.subplot(1,3,2)
plt.imshow(np.abs(E_h))
plt.title('Scattered beam',fontsize = 20)
plt.yticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.ylabel(r'$x (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(phi_axis_ticks*1e-6 / del_phi + M/2, phi_axis_ticks, fontsize = 12)
plt.xlabel(r'$\phi (\mathrm{\mu rad})$', fontsize = 15)


ax1 = plt.subplot(1,3,3)
color = 'tab:red'
ax1.set_xlabel(r'$\phi (\mu\mathrm{rad})$', fontsize = 15)
ax1.set_ylabel(r'$\mathrm{Transmission}$', color = color, fontsize = 15)
ax1.plot(phi*1e6, np.sum(np.abs(E_0)**2, axis = 0) / np.sum(np.abs(E_init)**2) , color = color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=10)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'$\mathrm{Reflectivity}$', color = color, fontsize = 15)
ax2.plot(phi*1e6, np.sum(np.abs(E_h)**2, axis = 0) / np.sum(np.abs(E_init)**2), color = color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=10)
plt.title('Integrated intensity',fontsize = 20)


fig2.tight_layout()


M = 500

x = np.linspace(0, 420*1e-3, 1024)
del_x = x[1] - x[0]
x_mid = x[-1]/2

L = np.linspace(10*1e-3, 610*1e-3, M)

E_0, E_h = perf_2d.laue_depth_dependece(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h)

fig3 = plt.figure(figsize = (10, 4))
x_axis_ticks = np.array([0, 50, 100, 150, 200]) # microns
L_axis_ticks = np.array([0, 200, 400, 600]) # micro radians
del_L = L[1] - L[0]

plt.subplot(1,3,1)
plt.imshow(np.abs(E_0))
plt.title('Transmitted beam',fontsize = 20)
plt.yticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.ylabel(r'$x (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(L_axis_ticks*1e-3 / del_L, L_axis_ticks, fontsize = 12)
plt.xlabel(r'$\mathrm{Thickness} (\mathrm{\mu m})$', fontsize = 15)

plt.subplot(1,3,2)
plt.imshow(np.abs(E_h))
plt.title('Scattered beam',fontsize = 20)
plt.yticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.ylabel(r'$x (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(L_axis_ticks*1e-3 / del_L, L_axis_ticks, fontsize = 12)
plt.xlabel(r'$\mathrm{Thickness} (\mathrm{\mu m})$', fontsize = 15)


ax1 = plt.subplot(1,3,3)
color = 'tab:red'
ax1.set_xlabel(r'$L (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(L_axis_ticks*1e-3 / del_L, L_axis_ticks, fontsize = 12)
ax1.set_ylabel(r'$\mathrm{Transmission}$', color = color, fontsize = 15)
ax1.plot(L*1e3, np.sum(np.abs(E_0)**2, axis = 0) / np.sum(np.abs(E_init)**2) , color = color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=10)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'$\mathrm{Reflectivity}$', color = color, fontsize = 15)
ax2.plot(L*1e3, np.sum(np.abs(E_h)**2, axis = 0) / np.sum(np.abs(E_init)**2), color = color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=10)
plt.title('Integrated intensity',fontsize = 20)


fig3.tight_layout()
plt.show()


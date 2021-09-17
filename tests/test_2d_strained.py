import numpy as np
import matplotlib.pyplot as plt
import dynamical_diffraction.strained_crystal_2d as strained_2d

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
del_z = 0.1e-3

# Calculated quantities to make a valid displacement field
twotheta = np.abs(alpha_0 -alpha_h)
k = 2*np.pi/lmbd
q_magnitude = np.abs(2*k*np.sin(twotheta/2))
d = 2*np.pi/q_magnitude

print(L//del_z)

def u(x,z):
    # Sum a few simple dislocations
    uu = np.zeros(x.shape)
    
    for ii in range(7):
        uu += np.arctan2(x-x_mid-x_mid*0.15*ii, z-0.5*L+L*0.03*ii)
    return uu/2/np.pi*d


E_0, E_h = strained_2d.laue_exponential_heun(E_init, u, del_z, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 20e-6)


u_array = [u(x, iz*del_z) for iz in range(int(L//del_z))]
u_array = np.stack(u_array).transpose()

fig = plt.figure(figsize = (6,8))

x_axis_ticks = np.array([0, 25, 50, 75, 100]) # microns
z_axis_ticks = np.array([0, 50, 100, 150, 200, 250, 300]) # microns


plt.subplot(3,1,1)
plt.imshow(u_array*1e7, aspect = del_x/del_z)
plt.colorbar()
plt.yticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.ylabel(r'$\mathrm{x} (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(z_axis_ticks*1e-3 / del_z, z_axis_ticks, fontsize = 12)
plt.xlabel(r'$\mathrm{z} (\mathrm{\mu m})$', fontsize = 15)
plt.title('Displacement field',fontsize = 20)

plt.subplot(3,1,2)
plt.imshow(np.abs(E_h), aspect = del_x/del_z)
plt.colorbar()
plt.yticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.ylabel(r'$\mathrm{x} (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(z_axis_ticks*1e-3 / del_z, z_axis_ticks, fontsize = 12)
plt.xlabel(r'$\mathrm{z} (\mathrm{\mu m})$', fontsize = 15)
plt.title('Scattered beam',fontsize = 20)

plt.subplot(3,1,3)
plt.imshow(np.abs(E_0), aspect = del_x/del_z)
plt.colorbar()
plt.yticks(x_axis_ticks*1e-3 / del_x, x_axis_ticks, fontsize = 12)
plt.ylabel(r'$\mathrm{x} (\mathrm{\mu m})$', fontsize = 15)
plt.xticks(z_axis_ticks*1e-3 / del_z, z_axis_ticks, fontsize = 12)
plt.xlabel(r'$\mathrm{z} (\mathrm{\mu m})$', fontsize = 15)
plt.title('Transmitted beam',fontsize = 20)

plt.tight_layout()
plt.show()
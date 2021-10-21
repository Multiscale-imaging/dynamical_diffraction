import numpy as np
import matplotlib.pyplot as plt
import dynamical_diffraction.strained_crystal_3d as strained_3d
from scipy.special import erf

# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 - 1j*1e-7
chi_h = 1.88e-6 - 1j*1e-7

alpha_h = 20*np.pi/180
eta = 15*np.pi/180

C = 1
lmbd = 0.71*1e-7

k = 2*np.pi / lmbd
k_0 = np.array([0, 0, k])
k_h = np.array([np.cos(eta)*np.sin(alpha_h), np.sin(eta)*np.sin(alpha_h), np.cos(alpha_h)])*k

stepsizes = [0.25e-3, 0.25e-3, 0.5e-3]
gridshape = [256, 256, 300]
phi = -0.6e-4

# make real space coords
x = np.arange(gridshape[0])*stepsizes[0] 
y = np.arange(gridshape[1])*stepsizes[1]
x,y  = np.meshgrid(x,y, indexing= 'ij')

# Make init condition
xmid = 5e-3
x_thickness =0.3e-3
y_width = 35e-3
ymid = 25e-3
y_smooth_param = 1e-3

E_init = np.exp(-(x-xmid)**2/2/x_thickness**2)*(1-erf((np.abs(y-ymid)-y_width/2)/y_smooth_param))
E_init = E_init.astype(complex)

# plt.imshow(np.abs(E_init))
# plt.show()


# Make dusplacement field array
z = np.arange(gridshape[2])*stepsizes[2]

mid_yz = [37e-3, 44e-3]
mid_yz_2 = [22e-3, 30e-3]
q_magnitude = np.linalg.norm(k_h - k_0)

u_array = np.zeros(gridshape)
for iz in range(gridshape[2]):
    u_array[:,:,iz] = y**2*2e-5 - np.arctan2(y-mid_yz[0], z[iz]-mid_yz[1])/q_magnitude*2 + np.arctan2(y-mid_yz_2[0], z[iz]-mid_yz_2[1])/q_magnitude*2

# plt.imshow(u_array[128,:,:])
# plt.show()

# Do the integration 
E0_out, Eh_out = strained_3d.laue_exponential_heun(E_init, u_array, stepsizes, gridshape, lmbd, k_0, k_h, chi_0, chi_h, phi = phi)

plt.figure(figsize = (10,6))
plt.subplot(1,2,2)
plt.imshow(np.abs(Eh_out))
plt.title('Scattered Beam')

plt.subplot(1,2,1)
plt.imshow(np.abs(E0_out))
plt.title('Transmitted beam')
plt.show()
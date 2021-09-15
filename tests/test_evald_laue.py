import dynamical_diffraction.ewald_laue_theory as elt
import numpy as np
import matplotlib.pyplot as plt

# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 + 1j*1e-8
chi_h = 1.88e-6 - 1j*1e-8
theta = 10*np.pi/180
C = 1
phi = np.linspace(-1e-5, 3e-5, 500)


# Semi infinite Bragg
R = elt.bragg_inf(phi, theta, theta, chi_0, chi_h)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel(r'$\phi (\mu\mathrm{rad})$', fontsize = 15)
ax1.set_ylabel(r'$|E_h/E_0|^2$', color = color, fontsize = 15)
ax1.plot(phi*1e6, np.abs(R)**2, color = color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=10)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'$\mathrm{Angle}(E_h/E_0)$', color = color, fontsize = 15)
ax2.plot(phi*1e6, np.angle(R), color = color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=10)
plt.title('Reflectivity',fontsize = 20)

fig.tight_layout()  
plt.show()


# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 + 1j*1e-8
chi_h = 1.88e-6 - 1j*1e-8
alpha_0 = 5*np.pi/180
alpha_h = 15*np.pi/180
C = 1
lmbd = 0.71
L = 20*1e4
phi = np.linspace(-2e-5, 5e-5, 5000)

# Finite Bragg
R, T = elt.bragg_finite(phi, L, lmbd, alpha_0, alpha_h, chi_0, chi_h)

fig = plt.figure(figsize = (10, 6))
ax1 = plt.subplot(1, 2, 1)
color = 'tab:red'
ax1.set_xlabel(r'$\phi (\mu\mathrm{rad})$', fontsize = 15)
ax1.set_ylabel(r'$|E_h(0)/E_0(0)|^2$', color = color, fontsize = 15)
ax1.plot(phi*1e6, np.abs(R)**2, color = color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=10)
plt.title('Reflectivity',fontsize = 20)
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel(r'$\mathrm{Angle}(E_h(0)/E_0(0))$', color = color, fontsize = 15)
# ax2.plot(phi*1e6, np.angle(R), color = color, linewidth = 2)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='minor', labelsize=10)

ax1 = plt.subplot(1, 2, 2)
color = 'tab:red'
ax1.set_xlabel(r'$\phi (\mu\mathrm{rad})$', fontsize = 15)
ax1.set_ylabel(r'$|E_0(L)/E_0(0)|^2$', color = color, fontsize = 15)
ax1.plot(phi*1e6, np.abs(T)**2, color = color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=10)
plt.title('Transmission',fontsize = 20)

# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel(r'$\mathrm{Angle}(E_h(L)/E_0(0))$', color = color, fontsize = 15)
# ax2.plot(phi*1e6, np.angle(T), color = color, linewidth = 2)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='minor', labelsize=10)

fig.tight_layout()  
plt.show()


# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 + 5j*1e-7
chi_h = 1.88e-6 - 3j*1e-7
alpha_0 = 5*np.pi/180
alpha_h = -15*np.pi/180
C = 1
lmbd = 0.71
L = 200*1e4
phi = np.linspace(-4e-5, 4e-5, 5000)
mu = lmbd/np.pi/np.imag(chi_0)

print(f'Attentuation length is {mu*1e-4:.2f}µm')
print(f'µL is {L/mu:.1f}')

# Fixed length Laue rocking curve
R, T = elt.laue_rockingcurve(phi, L, lmbd, alpha_0, alpha_h, chi_0, chi_h)

fig = plt.figure(figsize = (10, 6))
ax1 = plt.subplot(1, 2, 1)
color = 'tab:red'
ax1.set_xlabel(r'$\phi (\mu\mathrm{rad})$', fontsize = 15)
ax1.set_ylabel(r'$|E_h(L)/E_0(0)|^2$', color = color, fontsize = 15)
ax1.semilogy(phi*1e6, np.abs(R)**2, color = color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=10)

# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel(r'$\mathrm{Angle}(E_h(0)/E_0(0))$', color = color, fontsize = 15)
# ax2.plot(phi*1e6, np.angle(R), color = color, linewidth = 2)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='minor', labelsize=10)

plt.title('Reflectivity',fontsize = 20)

ax1 = plt.subplot(1, 2, 2)
color = 'tab:red'
ax1.set_xlabel(r'$\phi (\mu\mathrm{rad})$', fontsize = 15)
ax1.set_ylabel(r'$|E_0(L)/E_0(0)|^2$', color = color, fontsize = 15)
ax1.semilogy(phi*1e6, np.abs(T)**2, color = color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=10)

# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel(r'$\mathrm{Angle}(E_h(L)/E_0(0))$', color = color, fontsize = 15)
# ax2.plot(phi*1e6, np.angle(T), color = color, linewidth = 2)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='minor', labelsize=10)

plt.title('Transmission',fontsize = 20)

fig.tight_layout()  
plt.show()

# Fixed length Laue rocking curve
# Approximately Diamond at 17 keV, 111 reflection
chi_0 = -2e-6 + 3j*1e-7
chi_h = 1.88e-6 - 2j*1e-7
alpha_0 = -5*np.pi/180
alpha_h = 15*np.pi/180
C = 1
lmbd = 0.71
L = 200*1e4
phi = np.linspace(-4e-5, 4e-5, 5000)
mu = lmbd/np.pi/np.imag(chi_0)

print(f'Attentuation length is {mu*1e-4:.2f}µm')
L = np.linspace(0*1e4, 300*1e4, 500)
R, T = elt.laue_thickness(L,  lmbd, alpha_0, alpha_h, chi_0, chi_h)

fig = plt.figure(figsize = (6, 4))


plt.xlabel(r'$L (\mu\mathrm{m})$', fontsize = 15)
plt.ylabel(r'$|E_h(L)/E_0(0)|^2$', fontsize = 15)
plt.semilogy(L*1e-4, np.abs(R)**2, linewidth = 2)
plt.semilogy(L*1e-4, np.abs(T)**2, color = color, linewidth = 2)
plt.legend(['Reflectivity', 'Transmission'])
plt.grid()
fig.tight_layout()  
plt.show()
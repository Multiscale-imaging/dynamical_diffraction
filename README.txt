This is just a small collection of functions used to calculate dynamical diffraction patterns from plate-like crytals.

The formalism is based on the "symmetric Takagi Taupin Equations". See https://arxiv.org/pdf/1703.04100.pdf for example.

The perfect crystal functions are all based on Fourier transforming the incident radiation along the surface direction 
and solving the TTE-equations, which then reduce to a set of 2x2 eigenvalue problems which are solved exactly.

The finite difference algorithms for strained crystals, , which are also based on transverse Fourier transforms 
are documented in https://arxiv.org/abs/2106.12412 

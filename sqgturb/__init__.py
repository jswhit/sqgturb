"""
constant PV f-plane QG turbulence (a.k.a surface QG turbulence).
Doubly periodic geometry with sin(2*pi/L) jet basic state.
References:
http://journals.ametsoc.org/doi/pdf/10.1175/2008JAS2921.1 (section 3)
http://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281978%29035%3C0774%3AUPVFPI%3E2.0.CO%3B2

includes Ekman damping, linear thermal relaxation back
to equilibrium jet, and hyperdiffusion.

pv has units of meters per second.
scale by f*theta0/g to convert to temperature.

FFT spectral collocation method with 4th order Runge Kutta
time stepping (dealiasing with 2/3 rule, hyperdiffusion treated implicitly).

Jeff Whitaker December, 2016 <jeffrey.s.whitaker@noaa.gov>
"""
from .enkf_utils import gaspcohn, enkf_update, cartdist, lgetkf_update
from .sqg import SQG, rfft2, irfft2

__all__=['SQG','rfft2','irfft2',gaspcohn, enkf_update, cartdist]

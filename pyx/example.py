"""
example.py  -  demonstrates the sqg_cy Cython wrapper.

Replicates the __main__ block from the original sqg.py so that
results are directly comparable.

Usage:
    python example.py
"""

import numpy as np
from sqg_cy import SQG          # the compiled Cython extension

# ---- model parameters ------------------------------------------------
N          = 64
dt         = 1800.0             # time step (seconds)
diff_efold = 24.0 * 3600.0      # hyperdiff e-folding time for shortest wave
norder     = 8                  # hyperdiffusion order
r          = 0.0                # Ekman damping
nsq        = 1.0e-4
f          = 1.0e-4
g          = 9.8
theta0     = 300.0
H          = 10.0e3             # lid height (m)
U          = 20.0               # jet speed  (m/s)
L          = 20.0e6             # domain size (m)
tdiab      = 10.0 * 86400.0    # thermal relaxation (s)
scalefact  = f * theta0 / g     # PV -> temperature conversion

outputinterval = 6.0  * 3600.0  # output every 6 hours
tmax           = 300.0 * 86400.0 # run for 300 days

# ---- initial PV: isolated blob on lid --------------------------------
pv = np.zeros((2, N, N), dtype=np.float32)

nexp = 20
x = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=np.float32)
y = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=np.float32)
xx, yy = np.meshgrid(x, y)
pv[1] += 2000.0 * np.sin(xx / 2)**(2*nexp) * np.sin(yy)**nexp

# remove area mean from each level
for k in range(2):
    pv[k] -= pv[k].mean()

# ---- create model ----------------------------------------------------
model = SQG(pv, f=f, nsq=nsq, L=L, H=H, U=U, r=r,
            tdiab=tdiab, diff_order=norder, diff_efold=diff_efold,
            theta0=theta0, g=g, dt=dt)

print(model)

ntimesteps = int(outputinterval / model.dt)
nsteps     = int(tmax / outputinterval)

print(f"Running {nsteps} output steps, {ntimesteps} timesteps each …")

# ---- optionally write NetCDF output ----------------------------------
try:
    from netCDF4 import Dataset
    nc = Dataset("sqg_cy.nc", "w")
    nc.r          = float(model.r[0])
    nc.f          = float(model.f)
    nc.U          = float(model.U)
    nc.L          = float(model.L)
    nc.H          = float(model.H)
    nc.g          = g
    nc.theta0     = theta0
    nc.nsq        = float(model.nsq)
    nc.tdiab      = float(model.tdiab)
    nc.dt         = float(model.dt)
    nc.diff_efold = float(model.diff_efold)
    nc.diff_order = int(model.diff_order)
    nc.createDimension("x", N)
    nc.createDimension("y", N)
    nc.createDimension("z", 2)
    nc.createDimension("t", None)
    pvvar = nc.createVariable("pv", "f4", ("t", "z", "y", "x"), zlib=True)
    pvvar.units = "K"
    tvar  = nc.createVariable("t",  "f4", ("t",))
    tvar.units = "seconds"
    xvar  = nc.createVariable("x",  "f4", ("x",))
    xvar[:] = np.arange(N, dtype=np.float32) * model.L / N
    yvar  = nc.createVariable("y",  "f4", ("y",))
    yvar[:] = np.arange(N, dtype=np.float32) * model.L / N
    zvar  = nc.createVariable("z",  "f4", ("z",))
    zvar[:] = [0.0, model.H]
    use_nc = True
    print("Writing output to sqg_cy.nc")
except ImportError:
    use_nc = False
    print("netCDF4 not available – skipping file output")

# ---- time loop -------------------------------------------------------
nout = 0
while model.t < tmax:
    pv_out = model.advance(timesteps=ntimesteps)

    print(f"hr={model.t/3600:8.2f}  "
          f"min/max pv  {scalefact*pv_out.min():10.4f}  "
          f"{scalefact*pv_out.max():10.4f}")

    if use_nc:
        pvvar[nout, :, :, :] = pv_out * scalefact
        tvar[nout] = model.t
        nc.sync()

    nout += 1

if use_nc:
    nc.close()
    print("Done – output written to sqg_cy.nc")

# ---- demonstrate lower-level API ------------------------------------
print("\n--- low-level API demo ---")
# Forward FFT
spec = model.rfft2(pv_out)
print(f"rfft2 output shape: {spec.shape}, dtype: {spec.dtype}")

# Invert PV to get streamfunction
psispec = model.invert(spec)
print(f"invert output shape: {psispec.shape}")

# Compute one tendency
tend = model.gettend()
print(f"gettend output shape: {tend.shape}")

# Padded derivatives
xd, yd = model.xyderiv(spec)
print(f"xyderiv output shapes: {xd.shape}, {yd.shape}")

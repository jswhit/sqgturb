"""Single-ob increments side by side: 576 members / 6000 km, 16 members / 2400 km
(single scale), and 16 members / two-scale R localization (lgetkf_ms, multi-scale
Nerger factor, 4200/1400 km, band cutoff n=20, cross-band 0.35), for a 1 K lid
observation (ob error 1 K) at the domain centre and at the largest-spread point
within 2000 km of it.  Last background ensemble of cv620_6000km_cvb0.75-1.5.
See single_ob_increment.py for the individual versions.

  PYTHONPATH=../sqgturb_pylibs:. python compare_single_ob_increments.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sqgturb import cartdist, gaspcohn, lgetkf_ms

D = '/scratch4/BMC/gsienkf/whitaker/python/sqgturb/'
nc = Dataset(D + 'cv620_6000km_cvb0.75-1.5.nc')
ens_full = nc['pv_b'][-1].astype(np.float64)   # (576, z, y, x), K
x = nc['x'][:]; y = nc['y'][:]; L = nc.L; R = nc.oberrstdev**2
_, nz, ny, nx = ens_full.shape
X, Y = np.meshgrid(x, y)
IZ, INNOV = 1, 1.0


def single_scale(ens, iy, ix, loc_km):
  pert = ens - ens.mean(0)
  n = len(ens)
  hp = pert[:, IZ, iy, ix]
  cov = np.tensordot(hp, pert, axes=(0, 0)) / (n - 1)
  rho = gaspcohn(cartdist(X[iy, ix], Y[iy, ix], X, Y, L, L) / (loc_km * 1e3))
  return INNOV * cov / (hp.var(ddof=1) + R) * rho[None]


def two_scale_rloc(ens, iy, ix, scales=(4200., 1400.), cutoffs=(20.,), facts=(0.35,)):
  # as sqg_lgetkf_cvms.py: wavenumber bands, cross-band mixing, rescaled to the
  # original domain-mean variance; then lgetkf_ms with one ob (mean increment).
  n, nl = len(ens), len(scales)
  pert = ens - ens.mean(0)
  cmat = np.ones((nl, nl))
  for i in range(nl):
    for j in range(nl):
      if i != j:
        cmat[j, i] = facts[abs(i - j) - 1]
  kk = np.arange(nx // 2 + 1)[None, :]; ll = (ny * np.fft.fftfreq(ny))[:, None]
  wn = np.sqrt(kk**2 + ll**2)
  spec = np.fft.rfft2(pert)
  bands, prev = [], 0.
  for c in cutoffs:
    filt = np.fft.irfft2(np.where(wn < c, spec, 0.), s=(ny, nx))
    bands.append(filt - prev); prev = filt
  bands.append(pert - prev)
  z = np.einsum('jn...,jl->ln...', np.asarray(bands), cmat)
  var_l = (z**2).sum(1) / (n - 1)
  z *= np.sqrt(((pert**2).sum(0) / (n - 1)).mean() / var_l.sum(0).mean())
  ndim = nx * ny
  dist = cartdist(X[iy, ix], Y[iy, ix], X, Y, L, L)
  xens = ens.reshape(n, 2, ndim).astype(np.float32)
  covl = np.stack([gaspcohn(dist / (s * 1e3)).reshape(1, ndim) for s in scales]).astype(np.float32)
  xa = lgetkf_ms(nl, xens.copy(), z.reshape(nl * n, 2, ndim).astype(np.float32),
                 z[:, :, IZ, iy, ix].reshape(nl * n, 1).astype(np.float32),
                 pert[:, IZ, iy, ix].reshape(n, 1).astype(np.float32),
                 np.array([INNOV], np.float32), np.array([R], np.float32), covl, ngroups=0)
  return (xa.mean(0) - xens.mean(0)).reshape(2, ny, nx).astype(np.float64)


# the two ob points (chosen with the full ensemble's lid spread)
sprd = ens_full[:, IZ].std(0, ddof=1)
near = cartdist(X[ny//2, nx//2], Y[ny//2, nx//2], X, Y, L, L) <= 2000e3
points = [(ny // 2, nx // 2), np.unravel_index(np.where(near, sprd, -1).argmax(), sprd.shape)]
ens16 = ens_full[:16]
cases = [('576 members, 6000 km', lambda iy, ix: single_scale(ens_full, iy, ix, 6000.)),
         ('16 members, 2400 km', lambda iy, ix: single_scale(ens16, iy, ix, 2400.)),
         ('16 members, two-scale R-loc\n4200/1400 km, n<20, cross-band 0.35',
          lambda iy, ix: two_scale_rloc(ens16, iy, ix))]

ZOOM = 2500
km = (x - x[nx // 2]) / 1e3
fig, axs = plt.subplots(4, 3, figsize=(12.5, 16), constrained_layout=True)
for p, (iy, ix) in enumerate(points):
  incs = [np.roll(f(iy, ix), (ny // 2 - iy, nx // 2 - ix), axis=(1, 2)) for _, f in cases]
  vmax = np.ceil(max(np.abs(i).max() for i in incs) * 20) / 20
  lev = np.linspace(-vmax, vmax, 21)
  s576, s16 = sprd[iy, ix], ens16[:, IZ, iy, ix].std(ddof=1)
  for k, (iz, lname) in enumerate([(1, 'lid'), (0, 'surface')]):
    row = 2 * p + k
    for c, ((title, _), inc) in enumerate(zip(cases, incs)):
      ax = axs[row, c]
      cs = ax.contourf(km, km, inc[iz], lev, cmap='RdBu_r', extend='both')
      ax.contour(km, km, inc[iz], lev[np.abs(lev) > 1e-6 * vmax], colors='k', linewidths=0.3)
      ax.plot(0, 0, 'k+', ms=12, mew=2)
      ax.set_xlim(-ZOOM, ZOOM); ax.set_ylim(-ZOOM, ZOOM); ax.set_aspect('equal')
      ax.set_title((title + '\n' if row == 0 else '') +
                   f'{lname}: max {inc[iz].max():.2f}, min {inc[iz].min():.2f} K', fontsize=10)
      if row == 3: ax.set_xlabel('x from ob (km)')
      if c == 0:
        ax.set_ylabel(f'point ({iy}, {ix}), {lname}\nspread {s576:.2f} K (576) / {s16:.2f} K (16)\n'
                      'y from ob (km)')
  fig.colorbar(cs, ax=axs[2 * p:2 * p + 2], shrink=0.8, label='increment (K)',
               ticks=np.linspace(-vmax, vmax, 9))
fig.suptitle('Single observation at the lid, 1 K warmer than the background mean (ob error 1 K)\n'
             'last background ensemble of cv620_6000km_cvb0.75-1.5 (16 members = the first 16 of 576)')
out = D + 'single_ob_increment_comparison.png'
fig.savefig(out, dpi=100)
print(out)

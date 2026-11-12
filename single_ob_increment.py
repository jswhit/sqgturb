import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
sys.path.insert(0, '/scratch4/BMC/gsienkf/whitaker/python/sqgturb')
from sqgturb import cartdist, gaspcohn, lgetkf_ms
f = '/scratch4/BMC/gsienkf/whitaker/python/sqgturb/cv620_6000km_cvb0.75-1.5.nc'
nc = Dataset(f)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('search_km', nargs='?', type=float, default=0.,
                help='ob at the largest lid spread within this distance of the centre (0: centre)')
ap.add_argument('--nanals', type=int, default=None, help='use only the first NANALS members')
ap.add_argument('--loc-km', type=float, default=6000., help='Gaspari-Cohn cutoff (km)')
ap.add_argument('--scales', default=None,
                help='multiscale: km per band, large to small, e.g. 4200,1400 (as sqg_lgetkf_cvms.py)')
ap.add_argument('--band-cutoffs', default='20', help='total wavenumber band cutoffs (multiscale)')
ap.add_argument('--crossband', default='0.35', help='cross-band covariance factors (multiscale)')
ap.add_argument('--obloc', action='store_true',
                help='multiscale: observation-space (R) localization with the multi-scale Nerger '
                'factor, i.e. the ensemble-mean increment from sqgturb.lgetkf_ms itself (as in the '
                'runs); the third column then shows the model-space (B) localization version')
ap.add_argument('--point', default=None, help='iy,ix: ob grid point (overrides search_km)')
args = ap.parse_args()
ens = nc['pv_b'][-1].astype(np.float64)        # (nanals, z, y, x), K (scaled by scalefact)
ens_full = ens
if args.nanals:
  ens = ens[:args.nanals]
x = nc['x'][:]; y = nc['y'][:]; L = nc.L; R = nc.oberrstdev**2
nanals, nz, ny, nx = ens.shape
print('nanals', nanals, 'L', L/1e3, 'km', 'oberrstdev', nc.oberrstdev, 'hcov', nc.hcovlocal_scale)
X, Y = np.meshgrid(x, y)
iz = 1                                          # model lid (upper boundary)
pert = ens - ens.mean(0)
sprd = pert[:, iz].std(0, ddof=1)
# location: the largest lid spread within SEARCH_KM of the domain centre
SEARCH_KM = args.search_km
if args.point:
  iy, ix = (int(v) for v in args.point.split(','))
else:  # chosen from the full ensemble's spread, so the location does not depend on --nanals
  sprd_full = ens_full[:, iz].std(0, ddof=1)
  near = cartdist(X[ny//2, nx//2], Y[ny//2, nx//2], X, Y, L, L) <= SEARCH_KM * 1e3
  iy, ix = np.unravel_index(np.where(near, sprd_full, -1).argmax(), sprd_full.shape)
print('ob at (iy, ix) = (%d, %d), %.0f km from centre; lid spread there %.3f K (centre %.3f, domain mean %.3f, max %.3f)' % (
    iy, ix, cartdist(X[ny//2, nx//2], Y[ny//2, nx//2], X[iy, ix], Y[iy, ix], L, L) / 1e3,
    sprd[iy, ix], sprd[ny//2, nx//2], sprd.mean(), sprd.max()))
x0, y0 = x[ix], y[iy]
hp = pert[:, iz, iy, ix]
cov = np.tensordot(hp, pert, axes=(0, 0)) / (nanals - 1)   # (z, y, x)
gain = cov / (hp.var(ddof=1) + R)
innov = 1.0
dist = cartdist(X[iy, ix], Y[iy, ix], X, Y, L, L)
if args.scales:
  # sqg_lgetkf_cvms.py: split the perturbations into wavenumber bands, mix them with
  # the cross-band matrix, rescale to the original total variance, and localize band l
  # with its own scale.  Single-ob gain:
  #   sum_l rho_l cov(z_l, z_l(ob)) / (sum_l var(z_l(ob)) + R)
  scales = [float(v) for v in args.scales.split(',')]
  cutoffs = [float(v) for v in args.band_cutoffs.split(',')]
  facts = [float(v) for v in args.crossband.split(',')]
  nl = len(scales)
  cmat = np.ones((nl, nl))
  for i in range(nl):
    for j in range(nl):
      if i != j:
        cmat[j, i] = facts[abs(i - j) - 1]
  kk = np.arange(nx // 2 + 1)[None, :]; ll = (ny * np.fft.fftfreq(ny))[:, None]
  wavenums = np.sqrt(kk**2 + ll**2)
  spec = np.fft.rfft2(pert)
  bands, prev = [], 0.
  for c in cutoffs:
    filt = np.fft.irfft2(np.where(wavenums < c, spec, 0.), s=(ny, nx))
    bands.append(filt - prev); prev = filt
  bands.append(pert - prev)
  bands = np.asarray(bands)                                  # (nl, nanals, z, y, x)
  z = np.einsum('jn...,jl->ln...', bands, cmat)
  var_l = (z**2).sum(1) / (nanals - 1)
  z *= np.sqrt(((pert**2).sum(0) / (nanals - 1)).mean() / var_l.sum(0).mean())
  num = 0.; den = R
  for l in range(nl):
    hz = z[l][:, iz, iy, ix]
    num = num + gaspcohn(dist / (scales[l] * 1e3))[None] * np.tensordot(hz, z[l], axes=(0, 0)) / (nanals - 1)
    den = den + hz.var(ddof=1)
  inc_loc = innov * num / den
  inc_bloc = inc_loc
  if args.obloc:
    ndim = nx * ny
    xens = ens.reshape(nanals, 2, ndim).astype(np.float32)
    xprime = z.reshape(nl * nanals, 2, ndim).astype(np.float32)
    hxprime = z[:, :, iz, iy, ix].reshape(nl * nanals, 1).astype(np.float32)
    hxprime_orig = pert[:, iz, iy, ix].reshape(nanals, 1).astype(np.float32)
    covl = np.stack([gaspcohn(dist / (sc * 1e3)).reshape(1, ndim) for sc in scales]).astype(np.float32)
    xa = lgetkf_ms(nl, xens.copy(), xprime, hxprime, hxprime_orig,
                   np.array([innov], np.float32), np.array([R], np.float32), covl, ngroups=0)
    inc_loc = (xa.mean(0) - xens.mean(0)).reshape(2, ny, nx).astype(np.float64)
    for k, name in ((1, 'lid'), (0, 'surface')):
      line = f'  R-loc vs B-loc, {name}: rms ratio (corr) by distance:'
      for a, b in ((0, 500), (500, 1000), (1000, 1500), (1500, 2500), (2500, 4200)):
        m = (dist >= a * 1e3) & (dist < b * 1e3)
        r, q = inc_loc[k][m], inc_bloc[k][m]
        line += f'  {a}-{b} km {np.sqrt((r**2).mean() / (q**2).mean()):.2f} ({np.corrcoef(r, q)[0, 1]:.2f})'
      print(line)
  loc_label = f'{"/".join(f"{v:g}" for v in scales)} km, n<{args.band_cutoffs}, cb {args.crossband}'
  if args.obloc:
    loc_label = 'R-loc, ' + loc_label
  print('band variances at ob:', [round(float(v), 3) for v in var_l[:, iz, iy, ix]])
else:
  scales = [args.loc_km]
  inc_bloc = None
  inc_loc = innov * gain * gaspcohn(dist / (args.loc_km * 1e3))[None]
  loc_label = loc_label
inc_raw = innov * gain
print('background spread at ob %.3f K, ob increment %.3f K' % (hp.std(ddof=1), inc_loc[iz, iy, ix]))
print('surface increment range %.3f..%.3f K' % (inc_loc[0].min(), inc_loc[0].max()))
# doubly periodic: roll so the ob sits at the centre grid point
shift = (ny//2 - iy, nx//2 - ix)
inc_loc, inc_raw = (np.roll(a, shift, axis=(1, 2)) for a in (inc_loc, inc_raw))
if inc_bloc is not None: inc_bloc = np.roll(inc_bloc, shift, axis=(1, 2))
kmx = lambda v: (v - v[len(v)//2]) / 1e3
kmy = kmx
fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
vmax = np.ceil(np.abs(inc_loc).max() * 20) / 20
lev = np.linspace(-vmax, vmax, 21)
panels = [(inc_loc, loc_label, None), (inc_loc, loc_label + ', zoom', 2500),
          ((inc_bloc, 'B-loc, same settings', None) if args.obloc else (inc_raw, 'no localization', None))]
for j, (inc, ttl, zoom) in enumerate(panels):
  for i, (k, name) in enumerate([(1, 'lid'), (0, 'surface')]):
    ax = axs[i, j]
    cs = ax.contourf(kmx(x), kmy(y), inc[k], lev, cmap='RdBu_r', extend='both')
    ax.contour(kmx(x), kmy(y), inc[k], lev[np.abs(lev) > 1e-6 * vmax], colors='k', linewidths=0.3)
    ax.plot(0, 0, 'k+', ms=12, mew=2)
    if j < 2:
      for sc in scales:
        ax.add_patch(plt.Circle((0, 0), sc, fill=False, ls='--', lw=1, color='0.3'))
    if zoom:
      ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    else:
      ax.set_xticks([-10000, -5000, 0, 5000]); ax.set_yticks([-10000, -5000, 0, 5000])
    ax.set_aspect('equal'); ax.set_xlabel('x from ob (km)'); ax.set_ylabel('y from ob (km)')
    ax.set_title(f'{name}: {ttl}\nmax {inc[k].max():.2f} K, min {inc[k].min():.2f} K', fontsize=10)
cb = fig.colorbar(cs, ax=axs, shrink=0.6, label='increment (K)', ticks=np.linspace(-vmax, vmax, 9))
fig.suptitle(f'Single observation at the lid, grid point ({iy}, {ix}) (lid spread {sprd[iy, ix]:.2f} K), 1 K warmer than the background mean (ob error {nc.oberrstdev:g} K)\n'
             f'{nanals}-member background ensemble' + (f' (first {nanals} of {ens_full.shape[0]})' if args.nanals else '') + f' at the last cycle of cv620_6000km_cvb0.75-1.5   '
             f'(+ ob location, dashed circles = localization cutoffs)')
out = '/scratch4/BMC/gsienkf/whitaker/python/sqgturb/single_ob_increment_cv620_6000km_cvb0.75-1.5' + (f'_{iy}_{ix}' if (SEARCH_KM or args.point) else '') + (f'_n{nanals}' if args.nanals else '') + (f'_ms{args.scales.replace(",", "-")}_cb{args.crossband}' + ('_rloc' if args.obloc else '') if args.scales else (f'_loc{args.loc_km:g}' if args.nanals else '')) + '.png'
fig.savefig(out, dpi=110); print(out)

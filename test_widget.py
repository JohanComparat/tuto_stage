"""
Standalone test for the Section 8 widget logic.
Run from the activated conda environment:

    conda activate tuto_stage
    python test_widget.py

Each test is independent and prints [OK  ] or [FAIL].
"""
import sys, traceback, math

PIXEL_SCALE = 0.2   # arcsec / px
STAMP_PX    = 81
RN          = 5.0
SKY         = 200.0
N_FREE      = 7     # free Sersic2D parameters

def mark(label, ok, note=''):
    status = 'OK  ' if ok else 'FAIL'
    suffix = f'  ({note})' if note else ''
    print(f'  [{status}]  {label}{suffix}')
    return ok

PASSED = []
FAILED = []

def check(label, ok, note=''):
    r = mark(label, ok, note)
    (PASSED if r else FAILED).append(label)
    return r

# ── imports ──────────────────────────────────────────────────────────────────
print('\n[1] Core imports')
try:
    import numpy as np
    check('numpy', True, np.__version__)
except Exception as e:
    check('numpy', False, str(e)); sys.exit(1)

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check('matplotlib', True, matplotlib.__version__)
except Exception as e:
    check('matplotlib', False, str(e)); sys.exit(1)

try:
    from numpy.fft import fft2, ifft2, fftfreq
    check('numpy.fft', True)
except Exception as e:
    check('numpy.fft', False, str(e))

try:
    from astropy.modeling import models, fitting
    import astropy
    check('astropy.modeling', True, astropy.__version__)
except Exception as e:
    check('astropy.modeling', False, str(e))

try:
    import warnings, io
    check('warnings / io', True)
except Exception as e:
    check('warnings / io', False, str(e))

# ── pure-numpy helpers (same as widget) ─────────────────────────────────────
def _sersic_numpy(xx, yy, cx, cy, re_px, n, q, pa_rad, flux):
    dx  = xx.astype(np.float64) - float(cx)
    dy  = yy.astype(np.float64) - float(cy)
    cp  = float(np.cos(pa_rad)); sp = float(np.sin(pa_rad))
    xp  =  dx * cp + dy * sp
    yp  = -dx * sp + dy * cp
    r   = np.sqrt(xp**2 + (yp / max(float(q), 1e-3))**2)
    bn  = max(2.0 * float(n) - 0.327, 0.1)
    arg = -bn * ((r / max(float(re_px), 1e-3)) ** (1.0 / max(float(n), 0.05)) - 1.0)
    profile = np.exp(np.clip(arg, -50.0, 50.0))
    s = profile.sum()
    if s > 0:
        profile *= float(flux) / s
    return profile

def _gauss_convolve(image, sigma_px):
    if float(sigma_px) < 0.05:
        return image.copy()
    ny, nx = image.shape
    fy = fftfreq(ny)[:, None]; fx = fftfreq(nx)[None, :]
    kernel = np.exp(-2.0 * np.pi**2 * float(sigma_px)**2 * (fx**2 + fy**2))
    return np.real(ifft2(fft2(image) * kernel))

def _simulate(n_val, re_arcsec, flux_val, q_val, pa_deg,
              psf_fwhm_arcsec, sky_val, add_noise, seed):
    sz  = STAMP_PX; cx = sz / 2.0
    yy_, xx_ = np.mgrid[0:sz, 0:sz]
    re_px   = max(float(re_arcsec) / PIXEL_SCALE, 0.5)
    psf_sig = (float(psf_fwhm_arcsec) / PIXEL_SCALE) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sky_f   = float(sky_val)
    profile = _sersic_numpy(xx_, yy_, cx, cx, re_px, n_val, q_val,
                             np.deg2rad(float(pa_deg)), float(flux_val))
    conv    = _gauss_convolve(profile, psf_sig)
    s = conv.sum()
    if s > 0:
        conv *= float(flux_val) / s
    if add_noise:
        rng_  = np.random.default_rng(int(seed))
        lam   = np.maximum(conv + sky_f, 0.0)
        noisy = rng_.poisson(lam).astype(np.float64) - sky_f
        noisy += rng_.normal(0.0, RN, noisy.shape)
    else:
        noisy = conv.copy()
    sigma = np.sqrt(np.abs(noisy) + sky_f + RN**2).clip(1e-3)
    return conv, noisy, sigma

# ── [2] Sérsic profile: finiteness and flux conservation ────────────────────
print('\n[2] Pure-numpy Sérsic profile')
sz = STAMP_PX; cx = sz / 2.0
yy, xx = np.mgrid[0:sz, 0:sz]

for n_val, re_arcsec, flux_val, q_val, pa_deg in [
    (1.0, 0.8,  3162.0, 0.7,  45.0),   # typical disk
    (4.0, 0.4, 10000.0, 0.8,  60.0),   # elliptical
    (0.3, 3.0,   316.0, 1.0,   0.0),   # diffuse dwarf
    (6.0, 0.4, 10000.0, 0.3,  90.0),   # compact high-n
]:
    try:
        re_px   = max(re_arcsec / PIXEL_SCALE, 0.5)
        profile = _sersic_numpy(xx, yy, cx, cx, re_px, n_val, q_val,
                                 np.deg2rad(pa_deg), flux_val)
        finite  = np.all(np.isfinite(profile))
        fluxok  = abs(profile.sum() - flux_val) < 0.01 * flux_val
        check(f'n={n_val} r_e={re_arcsec}" flux={flux_val:.0f}',
              finite and fluxok,
              f'sum={profile.sum():.1f} peak={profile.max():.2f}')
    except Exception as e:
        check(f'n={n_val}', False, f'{type(e).__name__}: {e}')

# ── [3] Gaussian convolution: finiteness and flux conservation ───────────────
print('\n[3] Gaussian FFT convolution')
for psf_fwhm in [0.4, 0.8, 2.0]:
    try:
        profile  = _sersic_numpy(xx, yy, cx, cx, 4.0, 1.0, 0.7, 0.0, 3162.0)
        sigma_px = (psf_fwhm / PIXEL_SCALE) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        conv     = _gauss_convolve(profile, sigma_px)
        finite   = np.all(np.isfinite(conv))
        fluxok   = abs(conv.sum() - 3162.0) < 0.01 * 3162.0
        check(f'PSF FWHM={psf_fwhm}"', finite and fluxok,
              f'sigma={sigma_px:.2f}px  sum={conv.sum():.1f}')
    except Exception as e:
        check(f'PSF FWHM={psf_fwhm}"', False, f'{type(e).__name__}: {e}')

# ── [4] Poisson noise: no overflow for any slider-range flux ─────────────────
print('\n[4] Poisson noise — no overflow for full flux range')
rng_test = np.random.default_rng(0)
for flux_val in [316.0, 1000.0, 3162.0, 10000.0]:
    try:
        conv  = _sersic_numpy(xx, yy, cx, cx, 4.0, 1.0, 0.7, 0.0, flux_val)
        lam   = np.maximum(conv + SKY, 0.0)
        noisy = rng_test.poisson(lam).astype(np.float64) - SKY
        ok    = np.all(np.isfinite(noisy))
        check(f'flux={flux_val:.0f}', ok, f'peak={conv.max():.1f}')
    except Exception as e:
        check(f'flux={flux_val:.0f}', False, f'{type(e).__name__}: {e}')

# ── [5] Full _simulate: no overflow ─────────────────────────────────────────
print('\n[5] Full _simulate — all slider combinations')
CASES = [
    (1.0, 0.8,  3162.0, 0.7,  45.0, 0.8, 200.0, True,  1),
    (4.0, 0.4, 10000.0, 0.8,  60.0, 0.4, 200.0, True,  2),
    (6.0, 0.4, 10000.0, 0.3,   0.0, 2.0,  50.0, False, 3),
    (0.3, 3.0,   316.0, 1.0,  90.0, 0.4, 500.0, True,  4),
    (1.0, 0.4,  3162.0, 1.0,   0.0, 0.4, 200.0, True,  5),  # compact disk
]
for args in CASES:
    n_val, re_arcsec, flux_val, q_val, pa_deg, psf_fwhm, sky_val, noise, seed = args
    try:
        _, noisy, sigma = _simulate(*args)
        ok = np.all(np.isfinite(noisy)) and np.all(np.isfinite(sigma))
        check(f'n={n_val} r_e={re_arcsec}" flux={flux_val:.0f} psf={psf_fwhm}"',
              ok, f'peak={noisy.max():.1f}')
    except Exception as e:
        check(f'n={n_val} r_e={re_arcsec}"', False, f'{type(e).__name__}: {e}')
        traceback.print_exc()

# ── [6] astropy Sersic2D evaluation ─────────────────────────────────────────
print('\n[6] astropy.modeling.Sersic2D evaluation')
for n_val, re_arcsec in [(1.0, 0.8), (4.0, 0.4), (6.0, 0.4), (0.3, 3.0)]:
    try:
        re_px   = max(re_arcsec / PIXEL_SCALE, 0.5)
        m       = models.Sersic2D(amplitude=1.0, r_eff=re_px, n=n_val,
                                  x_0=cx, y_0=cx, ellip=0.3, theta=0.0)
        profile = m(xx, yy).astype(np.float64)
        ok      = np.all(np.isfinite(profile))
        check(f'Sersic2D n={n_val} r_e={re_arcsec}"', ok,
              f'max={profile.max():.2e}')
    except Exception as e:
        check(f'Sersic2D n={n_val}', False, f'{type(e).__name__}: {e}')
        traceback.print_exc()

# ── [7] Fitting: correct recovery and no auto-reset of sliders ──────────────
print('\n[7] LevMarLSQFitter: parameter recovery')

def _fit_galaxy(arr, sigma, n0, re0_arcsec, fix_n=False, fix_re=False):
    """Mirror of the widget _fit logic (post-fix version)."""
    sz  = arr.shape[0]; cx = sz / 2.0
    yy_, xx_ = np.mgrid[0:sz, 0:sz]
    re0px = re0_arcsec / PIXEL_SCALE
    eps   = 1e-4
    bounds = {
        'ellip':     (0.0, 0.95),
        'amplitude': (0.0, None),
        'theta':     (-math.pi / 2.0, math.pi / 2.0),
        'n':     (n0-eps, n0+eps) if fix_n  else (0.3, 6.0),
        'r_eff': (re0px*(1-eps), re0px*(1+eps)) if fix_re else (0.3, sz*0.45),
    }
    m0 = models.Sersic2D(amplitude=max(float(arr.max()), 1.0),
                          r_eff=re0px, n=n0, x_0=cx, y_0=cx,
                          ellip=0.3, theta=0.0, bounds=bounds)
    fitter_lm = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m_fit = fitter_lm(m0, xx_, yy_, arr, weights=1.0/sigma, maxiter=500)
    n_fit  = m_fit.n.value
    re_fit = m_fit.r_eff.value * PIXEL_SCALE
    return n_fit, re_fit, m_fit(xx_, yy_)

# 7a. Good initial guess → good recovery
for n_true, re_true in [(1.0, 0.8), (4.0, 0.6), (2.0, 1.2)]:
    try:
        _, arr, sigma = _simulate(n_true, re_true, 5000.0, 0.7, 45.0, 0.8, 200.0, True, 42)
        n_fit, re_fit, _ = _fit_galaxy(arr, sigma, n0=n_true, re0_arcsec=re_true)
        n_ok  = abs(n_fit  - n_true)  < 0.5
        re_ok = abs(re_fit - re_true) < 0.3
        check(f'recovery n_true={n_true} r_e_true={re_true}"',
              n_ok and re_ok,
              f'n_fit={n_fit:.2f} r_e_fit={re_fit:.2f}"')
    except Exception as e:
        check(f'recovery n={n_true}', False, f'{type(e).__name__}: {e}')
        traceback.print_exc()

# 7b. Deliberately bad initial guess → convergence should be checked (no crash)
print('\n[7b] Fit with bad initial guess (should converge or fail gracefully)')
for n0_bad, re0_bad in [(5.0, 2.5), (0.3, 0.4)]:
    try:
        _, arr, sigma = _simulate(1.0, 0.8, 5000.0, 0.7, 45.0, 0.8, 200.0, True, 7)
        n_fit, re_fit, _ = _fit_galaxy(arr, sigma, n0=n0_bad, re0_arcsec=re0_bad)
        check(f'bad init n0={n0_bad} re0={re0_bad}" → no crash',
              True, f'n_fit={n_fit:.2f} r_e_fit={re_fit:.2f}"')
    except Exception as e:
        check(f'bad init n0={n0_bad}', False, f'{type(e).__name__}: {e}')

# 7c. Verify theta bounding breaks π-ambiguity
print('\n[7c] Theta bound [-π/2, π/2] prevents sign flip')
try:
    _, arr, sigma = _simulate(1.0, 0.8, 5000.0, 0.6, 45.0, 0.8, 200.0, False, 1)
    n_fit, re_fit, _ = _fit_galaxy(arr, sigma, n0=1.0, re0_arcsec=0.8)
    # theta must stay in [-π/2, π/2]
    re_px = max(0.8 / PIXEL_SCALE, 0.5)
    m0 = models.Sersic2D(amplitude=arr.max(), r_eff=re_px, n=1.0,
                          x_0=cx, y_0=cx, ellip=0.3, theta=0.0,
                          bounds={'ellip': (0.0, 0.95), 'amplitude': (0.0, None),
                                  'theta': (-math.pi/2, math.pi/2),
                                  'n': (0.3, 6.0), 'r_eff': (0.3, sz*0.45)})
    fitter_lm = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yy_, xx_ = np.mgrid[0:sz, 0:sz]
        m_fit = fitter_lm(m0, xx_, yy_, arr, weights=1.0/sigma, maxiter=300)
    theta_ok = -math.pi/2 <= m_fit.theta.value <= math.pi/2
    check('theta in [-pi/2, pi/2]', theta_ok, f'theta={m_fit.theta.value:.3f} rad')
except Exception as e:
    check('theta bound', False, f'{type(e).__name__}: {e}')
    traceback.print_exc()

# ── [8] Galaxy-mask chi² is sensitive to model errors ───────────────────────
print('\n[8] Galaxy-mask chi² detects wrong model (n=4 fit on n=1 galaxy)')
try:
    # True: n=1 disk.  Bad model: n=4 de Vaucouleurs.
    r  = np.sqrt((xx - cx)**2 + (yy - cx)**2)
    true_gal  = _sersic_numpy(xx, yy, cx, cx, 4.0, 1.0, 0.7, 0.0, 3162.0)
    bad_model = _sersic_numpy(xx, yy, cx, cx, 4.0, 4.0, 0.7, 0.0, 3162.0)  # wrong n
    rng_  = np.random.default_rng(99)
    noisy = rng_.poisson(np.maximum(true_gal + SKY, 0)).astype(float) - SKY
    noisy += rng_.normal(0, RN, noisy.shape)
    sigma = np.sqrt(np.abs(noisy) + SKY + RN**2).clip(1e-3)
    resid = noisy - bad_model

    # OLD metric (all pixels) — should be near 1 even for wrong model
    chi2_all = float(np.sum((resid/sigma)**2) / (sz*sz - N_FREE))

    # NEW metric (galaxy mask only)
    thresh    = max(bad_model.max() * 0.05, 1e-10)
    gal_mask  = bad_model > thresh
    r_gal     = resid[gal_mask]; s_gal = sigma[gal_mask]
    chi2_gal  = float(np.sum((r_gal/s_gal)**2) / max(gal_mask.sum() - N_FREE, 1))
    max_nsig  = float(np.max(np.abs(r_gal) / np.clip(s_gal, 1e-6, None)))

    check('OLD chi2 (all px) ~ 1 despite wrong model',
          chi2_all < 2.0,
          f'chi2_all={chi2_all:.3f}  (confirms background dilution)')
    check('NEW chi2 (gal mask) > 2 for wrong model',
          chi2_gal > 2.0,
          f'chi2_gal={chi2_gal:.3f}  n_gal={gal_mask.sum()}')
    check('max |r|/sigma > 3 in galaxy region',
          max_nsig > 3.0,
          f'max_nsig={max_nsig:.2f}')
except Exception as e:
    check('galaxy-mask chi²', False, f'{type(e).__name__}: {e}')
    traceback.print_exc()

# ── [9] Matplotlib figure → PNG (no display backend needed) ─────────────────
print('\n[9] Matplotlib figure → PNG bytes')
try:
    _, noisy, _ = _simulate(1.0, 0.8, 3162.0, 0.7, 45.0, 0.8, 200.0, True, 99)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(noisy, origin='lower', cmap='inferno')
    ax.set_title('test')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0); nbytes = len(buf.read())
    plt.close(fig)
    check('fig → PNG', nbytes > 500, f'{nbytes} bytes')
except Exception as e:
    check('fig → PNG', False, f'{type(e).__name__}: {e}')
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(PASSED) + len(FAILED)
print(f'\n{"="*60}')
print(f'Results: {len(PASSED)}/{total} passed')
if FAILED:
    print(f'FAILED:')
    for f in FAILED:
        print(f'  - {f}')
else:
    print('All tests passed — the widget should work correctly.')
print('='*60)

# Provenance — `sz_actxdesi` (Module 3, N10c)

Data used by `notebooks/sz_actxdesi_bgs_measurement.ipynb` (stacked thermal-SZ measurement,
DESI DR1 BGS × ACT DR6). **All products are public.** Nothing here is redistributed in the repo:
the notebook downloads each file into `cache/` on first run (cached afterwards). This file records
exactly what is fetched, from where, and how it is used.

## ACT DR6 + Planck CMB maps — NASA LAMBDA

Pixelisation: plate carrée (CAR), 0.5′ pixels, read with `pixell`. Beam FWHM per product noted below.

| Product | File | Size | Used for |
|---|---|---|---|
| NILC Compton-*y* (fiducial) | `act-planck_dr6.02_nilc_ComptonY.fits` | 1.78 GB | main *y* signal |
| NILC *y*, CIB-deprojected (β=1.0, T=10.7 K) | `..._nilc_ComptonY_deproj_cib_1.0_10.7.fits` | 1.78 GB | dust-robust *y* |
| NILC *y*, dβ-deprojected | `..._nilc_ComptonY_deproj_cib_cibdBeta_1.0_10.7.fits` | 1.78 GB | 2nd deprojection |
| NILC *y*, CIB-deproj β=1.6 | `..._nilc_ComptonY_deproj_cib_1.6_10.7.fits` | 1.78 GB | Fig. 21 (β sensitivity) |
| NILC *y*, CIB-deproj T=24 K | `..._nilc_ComptonY_deproj_cib_1.0_24.0.fits` | 1.78 GB | Fig. 22 (T sensitivity) |
| Single-freq co-adds f090/f150/f220 | `act-planck_dr6.02_coadd_AA_daynight_{f090,f150,f220}_map.fits` | 5.35 GB each | dust/CIB & 220 GHz tests |

- NILC *y*-maps: `https://lambda.gsfc.nasa.gov/data/act/nilc/published/` — beam FWHM 1.6′
  (transfer function `ilc_beam.txt` from the Compton-*y* product page).
- Single-frequency co-adds: `https://lambda.gsfc.nasa.gov/data/act/maps/published/` — TQU cubes
  `(3, 10320, 43200)`, beam FWHM 2.1′/1.4′/1.0′. The notebook **streams only the temperature (Stokes I)
  plane** via HTTP range requests, so each frequency is cached as a 1.78 GB 2-D FITS (`coadd_<f>_T.fits`)
  instead of the full 5.35 GB.
- Product pages: [Compton-y](https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_compton_maps_info.html) ·
  [DR6.02 maps](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_maps_info.html)
- References: Coulton et al. 2024, PRD 109, 063530 ([arXiv:2307.01258](https://arxiv.org/abs/2307.01258));
  Naess et al. 2025, JCAP 2025, 061 ([arXiv:2503.14451](https://arxiv.org/abs/2503.14451)).

## DESI DR1 BGS galaxies — FastSpecFit value-added catalogue

- File(s): `fastspec-iron-main-bright-nside1-hp<NN>.fits` (nside=1 HEALPix; the notebook loads the
  pixel(s) in `CONFIG['FASTSPEC_HP']`, default `[6]` = NGC-equatorial, 5.92 GB).
- URL base: `https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/`
- HDUs (row-matched): `METADATA` (TARGETID, RA, DEC, Z, ZWARN, SPECTYPE, BGS_TARGET),
  `SPECPHOT` (**LOGMSTAR** = stellar mass), `FASTSPEC` (spectral fit).
- Selection applied in the notebook: `LOGMSTAR > 10`, `0.01 < z < 0.6`, `ZWARN == 0`,
  `SPECTYPE == 'GALAXY'`, BGS_BRIGHT bit set. Yields ⟨z⟩≈0.24, ⟨logM*⟩≈10.7 (matches the target sample).
- References: DESI DR1 (DESI Collaboration 2025, AJ 171, 285,
  [arXiv:2503.14745](https://arxiv.org/abs/2503.14745)); FastSpecFit (Moustakas et al.,
  [docs](https://fastspecfit.readthedocs.io)); Siudek et al. 2024, A&A 691, A308
  ([arXiv:2409.19066](https://arxiv.org/abs/2409.19066)).

## NVSS radio catalogue — VizieR

- Queried live via `astroquery.vizier` catalogue `VIII/65/nvss` (cached to `cache/nvss_box.npz`).
- Reference: Condon et al. 1998, AJ 115, 1693.

## Published comparison values (hard-coded in the notebook, with citations)

- $\bar y$–$\bar\tau$ scaling for BGS (SIMBA-calibrated): $\ln\tau_0=-9.67$, $m=0.93$, $\bar y_0=10^{-7}$
  (Battaglia 2016 form, [arXiv:1607.02442](https://arxiv.org/abs/1607.02442); SIMBA: Davé et al. 2019,
  [arXiv:1901.10203](https://arxiv.org/abs/1901.10203)).
- Pairwise-kSZ optical depths for the BGS mass bins: Hadzhiyska et al. 2026, PRD 113, 063565
  ([arXiv:2510.14135](https://arxiv.org/abs/2510.14135)).

These are *literature reference values* used only for the comparison figures (14 & 17); the notebook's own
measurements come entirely from the public maps + catalogue above.

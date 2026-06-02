# Astronomy & Astrophysics Tutorials for Bachelor Interns
### Université Grenoble Alpes — Stage L3

A self-contained set of Jupyter notebooks covering astronomy, astrophysics, and cosmology through hands-on work with real observational data. Each notebook is designed for a bachelor-level intern (L3) with no prior astronomy experience but a solid background in mathematics and classical physics.

---

## Table of contents

1. [Who is this for?](#1-who-is-this-for)
2. [Prerequisites](#2-prerequisites)
3. [Environment setup](#3-environment-setup)
4. [Repository structure](#4-repository-structure)
5. [Full notebook catalog](#5-full-notebook-catalog)
6. [Suggested learning paths](#6-suggested-learning-paths)
7. [Data sources and download](#7-data-sources-and-download)
8. [Troubleshooting](#8-troubleshooting)
9. [How to add a new notebook](#9-how-to-add-a-new-notebook)
10. [References and further reading](#10-references-and-further-reading)

---

## 1. Who is this for?

These tutorials target **bachelor-level interns** at UGA doing a research internship in astrophysics or cosmology. They are also appropriate for students encountering observational astronomy/cosmology for the first time.

The notebooks assume you are comfortable with:
- Python 3 (basic syntax, functions, loops)
- NumPy arrays and matplotlib plots
- Physics: mechanics, thermodynamics, and electromagnetism

No prior astronomy knowledge is required. Each notebook starts from first principles and provides relevant physical background before the coding exercises.

---

## 2. Prerequisites

### Python knowledge

You need to be able to read and modify Python code. If you are new to scientific Python, work through the following free resources before starting:

| Resource | What to cover |
|----------|--------------|
| [Software Carpentry — Python](https://swcarpentry.github.io/python-novice-inflammation/) | Python basics, NumPy, matplotlib |
| [astropy tutorials](https://learn.astropy.org/) | astropy Tables, units, coordinates |
| [Scipy Lecture Notes](https://scipy-lectures.org/) | NumPy, SciPy, matplotlib in depth |

### Physics background

| Notebook series | Recommended prior knowledge |
|----------------|-----------------------------|
| Module 0 (Tools) | None beyond basic Python |
| Module 1 (Stars) | Thermodynamics, electromagnetic radiation, calculus |
| Module 2 (Galaxies) | Special relativity (Doppler), Newton's law of gravitation |
| Module 3 (Galaxy statistics) | Basic probability and statistics |
| Module 4 (Large-scale structure) | Fourier analysis, basic statistics |
| Module 5 (Advanced) | Modules 1–4, familiarity with cosmological models |

---

## 3. Environment setup

### 3.1 Install Miniforge (once per machine)

[Miniforge](https://github.com/conda-forge/miniforge) is the recommended conda distribution. It ships with `mamba`, a fast drop-in replacement for `conda`. On the lab machines, Miniforge is already installed at `/home/comparat/miniforge3` and the environments live under `/home/comparat/mamba/envs/`.

If you are setting up a new machine:

```bash
# Download the installer (Linux x86_64)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Run the installer (follow prompts, accept defaults)
bash Miniforge3-Linux-x86_64.sh

# Reload shell
source ~/.bashrc
```

Verify the installation:

```bash
mamba --version   # should print 2.x.x
```

### 3.2 Existing environments on the lab machines

The supervisor's research environments are already available. Do **not** modify them.

| Environment | Purpose |
|-------------|---------|
| `sum_stat` | Galaxy stellar mass function and summary statistics (N01, N02, N09) |
| `hod_mod` | Halo Occupation Distribution model (N11) |
| `gga_model` | Galaxy/gas/AGN X-ray model + JAX (N13) |
| `jax-cpu` | JAX CPU-only environment |

To list all available environments:

```bash
mamba env list
```

### 3.3 Create the tutorial environment (for Modules 0–2 and new notebooks)

From the root of this repository, create a self-contained environment for Modules 0–2 and all proposed new notebooks (N03–N15):

```bash
mamba env create -f environment.yml
```

This installs Python 3.11 and all required packages (astropy, healpy, scipy, specutils, astroquery, …). It takes a few minutes the first time.

Activate the environment **every time** you open a new terminal:

```bash
mamba activate tuto_stage
```

Deactivate when you are done:

```bash
mamba deactivate
```

### 3.4 Which environment to use for each notebook

| Notebook | Environment | Notes |
|----------|-------------|-------|
| N01 `file_formats_formats` | `sum_stat` or `tuto_stage` | — |
| N02 `astronomical_coordinates` | `sum_stat` or `tuto_stage` | — |
| N04 `stellar_luminosity_mass` | `tuto_stage` | needs `astroquery` |
| N09 `stellar_mass_function` | `sum_stat` | needs `sum_stat` package |
| N11 `angular_power_spectrum` | `hod_mod` + `sum_stat` | needs both; see note below |
| N13 `xray-galaxy-cross-correlation` | `gga_model` | needs JAX + `gga_model` |
| All proposed new notebooks | `tuto_stage` | — |

> **Note for N11:** the notebook requires both `sum_stat` and `hod_mod`, which live in separate environments. A combined environment can be created by the supervisor on request.

### 3.5 Launch Jupyter

```bash
mamba activate tuto_stage       # or the appropriate environment
cd /path/to/tuto_stage
jupyter lab
```

Your browser will open automatically. Navigate to the `notebooks/` folder and open any notebook.

### 3.6 Register a kernel so JupyterLab sees your environment

If JupyterLab opens but the kernel is not found, register it explicitly:

```bash
mamba activate tuto_stage
python -m ipykernel install --user --name tuto_stage --display-name "Python (tuto_stage)"
```

Then restart Jupyter and select **Python (tuto_stage)** from the kernel menu.

### 3.7 Update the environment

If new packages are added to `environment.yml`:

```bash
mamba env update -f environment.yml --prune
```

### 3.8 Remove the environment

```bash
mamba env remove -n tuto_stage
```

---

## 4. Repository structure

```
tuto_stage/
├── README.md               ← this file
├── environment.yml         ← mamba environment specification
├── notebooks/              ← source notebooks (edit these)
│   │
│   │── Module 0: Tools
│   ├── file_formats_formats.ipynb
│   ├── astronomical_coordinates.ipynb
│   │
│   │── Module 1: Stars
│   ├── stellar_luminosity_mass.ipynb
│   ├── blackbody_stellar_colors.ipynb          [proposed]
│   │
│   │── Module 2: Galaxies
│   ├── galaxy_spectra_redshift.ipynb           [proposed]
│   ├── hubble_law.ipynb                        [proposed]
│   ├── galaxy_rotation_curves.ipynb            [proposed]
│   ├── galaxy_color_magnitude.ipynb            [proposed]
│   │
│   │── Module 3: Galaxy populations
│   ├── stellar_mass_function.ipynb
│   ├── luminosity_function.ipynb               [proposed]
│   │
│   │── Module 4: Large-scale structure
│   ├── angular_power_spectrum.ipynb
│   ├── two_point_correlation.ipynb             [proposed]
│   │
│   └── Module 5: Multi-wavelength & advanced
│       ├── xray-galaxy-cross-correlation.ipynb
│       ├── photometric_redshifts.ipynb         [proposed]
│       └── cosmic_microwave_background.ipynb   [proposed]
│
└── notebooks_evaluated/    ← pre-executed notebooks with all outputs
    ├── file_formats_formats.ipynb
    ├── astronomical_coordinates.ipynb
    ├── stellar_luminosity_mass.ipynb
    ├── stellar_mass_function.ipynb
    ├── angular_power_spectrum.ipynb
    └── xray-galaxy-cross-correlation.ipynb
```

Notebooks marked `[proposed]` are planned and may be added progressively.

The `notebooks_evaluated/` folder contains fully executed versions of all existing notebooks, generated by running each notebook end-to-end with no errors. They serve as a reference showing expected outputs and figures before an intern runs the notebooks themselves. They are generated with:

```bash
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=600 \
  --output-dir notebooks_evaluated/ \
  notebooks/<name>.ipynb
```

Each notebook requires its specific environment (see Section 3.4).

---

## 5. Full notebook catalog

Each notebook entry lists: learning objectives, physical prerequisites, Python packages used, data source, and estimated working time.

---

### Module 0 — Tools for Observational Astronomy

These two notebooks are the entry point for every intern. Do them first, in order.

---

#### N01 · `file_formats_formats.ipynb` — Astronomical Data File Formats
**Status:** complete  
**Level:** introductory | **Time:** 2–3 h

**Learning objectives**
1. Understand the structure and purpose of FITS, VOTable, ECSV, HDF5, and CSV files.
2. Read and write astronomical catalogs with `astropy.table.Table`.
3. Filter, sort, and add columns to a catalog.
4. Preserve physical units and metadata through format conversions.
5. Choose the right file format for a given science case.

**Prerequisites:** basic Python, no astronomy  
**Packages:** `astropy`, `h5py`, `matplotlib`, `numpy`  
**Data:** Gaia DR2 subsample (bundled with notebook)

**Key concepts:** FITS header, HDU, binary table, astropy Table, units

---

#### N02 · `astronomical_coordinates.ipynb` — Astronomical Coordinate Systems
**Status:** complete  
**Level:** introductory | **Time:** 3–4 h

**Learning objectives**
1. Explain the four main coordinate systems: ICRS, Galactic, Ecliptic, Alt-Az.
2. Use `astropy.coordinates.SkyCoord` to convert between systems.
3. Understand parallax and proper motion from Gaia astrometry.
4. Visualise star positions in multiple sky projections.
5. Build a stellar density map with HEALPix pixelisation.
6. Read a Hertzsprung–Russell diagram directly from Gaia photometry.

**Prerequisites:** N01  
**Packages:** `astropy`, `healpy`, `matplotlib`, `numpy`  
**Data:** Gaia DR2 (bundled subsample)

**Key concepts:** RA/Dec, Galactic latitude/longitude, parallax, proper motion, HEALPix, colour–magnitude diagram

---

### Module 1 — Stars: Physics and Measurement

---

#### N03 · `blackbody_stellar_colors.ipynb` — Blackbody Radiation and Stellar Colors
**Status:** proposed  
**Level:** introductory–intermediate | **Time:** 2–3 h

**Learning objectives**
1. Derive and plot the Planck blackbody function B_ν(T) and B_λ(T).
2. Apply Wien's displacement law to relate peak wavelength to stellar temperature.
3. Apply the Stefan–Boltzmann law to compute total luminosity from radius and temperature.
4. Understand the photometric magnitude system and the definition of a colour index.
5. Map the B−V colour index to effective temperature using the Ballesteros (2012) formula.
6. Locate stellar spectral types (O, B, A, F, G, K, M) on a colour–temperature diagram.

**Prerequisites:** N01, N02; university thermodynamics  
**Packages:** `numpy`, `scipy`, `matplotlib`, `astropy`  
**Data:** none (pure computation)

**Key concepts:** Planck function, Wien's law, Stefan–Boltzmann, magnitude system, colour index, spectral type

---

#### N04 · `stellar_luminosity_mass.ipynb` — Measuring Stellar Luminosity and Mass
**Status:** in progress  
**Level:** intermediate | **Time:** 3–4 h

**Learning objectives**
1. Convert a Gaia parallax measurement into a distance with its uncertainty.
2. Compute the distance modulus and convert apparent magnitude to absolute magnitude.
3. Apply bolometric corrections to obtain the total luminosity in solar units.
4. Use the empirical main-sequence mass–luminosity relation (L ∝ M^α) to estimate stellar mass.
5. Propagate measurement uncertainties through each step.
6. Position a sample of Gaia stars on the HR diagram and read off their properties.

**Prerequisites:** N02, N03; error propagation  
**Packages:** `astropy`, `numpy`, `matplotlib`, `scipy`  
**Data:** Gaia DR3 queried via `astroquery.gaia`

**Key equations**
```
distance [pc]       d = 1 / ω_mas × 1000
distance modulus    μ = m − M = 5 log₁₀(d / 10 pc)
luminosity          L / L☉ = 10^[0.4 (M☉_bol − M_bol)]
main-sequence mass  L / L☉ ≈ (M / M☉)^4   [solar-type stars]
```

**Key concepts:** parallax, distance modulus, bolometric correction, mass–luminosity relation, uncertainty propagation

---

### Module 2 — Galaxies: Observation and Kinematics

---

#### N05 · `galaxy_spectra_redshift.ipynb` — Galaxy Spectra and Redshift Measurement
**Status:** proposed  
**Level:** intermediate | **Time:** 3 h

**Learning objectives**
1. Read and plot an SDSS galaxy spectrum with `specutils`.
2. Identify prominent emission (Hα, [OII]) and absorption (Ca H&K) lines.
3. Measure the observed wavelength of a spectral line and compute the redshift z.
4. Convert redshift to recession velocity and, for small z, to distance using Hubble's law.
5. Compare spectra of an elliptical galaxy, a spiral galaxy, and a quasar.

**Prerequisites:** N01; special relativity (Doppler effect)  
**Packages:** `specutils`, `astropy`, `astroquery`, `matplotlib`, `numpy`  
**Data:** SDSS spectra via `astroquery.sdss`

**Key concepts:** spectral line, Doppler shift, cosmological redshift, recession velocity, spectral type (galaxy)

---

#### N06 · `hubble_law.ipynb` — Hubble's Law and the Expanding Universe
**Status:** proposed  
**Level:** intermediate | **Time:** 3 h

**Learning objectives**
1. Build a Hubble diagram (recession velocity vs. distance) from a nearby galaxy sample.
2. Fit a straight line to extract the Hubble constant H₀.
3. Understand why the scatter in the Hubble diagram comes from peculiar velocities.
4. Compute look-back time and comoving distance for a flat ΛCDM cosmology.
5. Compare your H₀ estimate to the CMB (Planck 2018) and Cepheid (SH0ES) measurements.

**Prerequisites:** N05; basic kinematics  
**Packages:** `astropy`, `astroquery`, `numpy`, `scipy`, `matplotlib`  
**Data:** NASA/IPAC Extragalactic Database (NED) via `astroquery.ipac.ned`

**Key concepts:** Hubble constant, Hubble tension, peculiar velocity, look-back time, flat ΛCDM

---

#### N07 · `galaxy_rotation_curves.ipynb` — Galaxy Rotation Curves and Dark Matter
**Status:** proposed  
**Level:** intermediate | **Time:** 3–4 h

**Learning objectives**
1. Read a HI 21 cm rotation curve from a published dataset.
2. Predict the rotation curve from the visible stellar disc alone (Keplerian expectation).
3. Identify the discrepancy that requires the existence of a dark matter halo.
4. Fit an NFW halo profile to the observed rotation curve.
5. Estimate the dark matter fraction within the optical radius.

**Prerequisites:** N02, N06; Newtonian mechanics, circular motion  
**Packages:** `numpy`, `scipy`, `matplotlib`, `astropy`  
**Data:** THINGS survey rotation curves (publicly available ASCII tables)

**Key concepts:** circular velocity, Keplerian rotation, NFW profile, dark matter, mass decomposition

---

#### N08 · `galaxy_color_magnitude.ipynb` — The Galaxy Color–Magnitude Diagram
**Status:** proposed  
**Level:** intermediate | **Time:** 2–3 h

**Learning objectives**
1. Download SDSS photometry for a volume-limited galaxy sample.
2. Plot the g − r colour vs. r absolute magnitude diagram.
3. Identify the red sequence, the blue cloud, and the green valley.
4. Fit a Gaussian mixture to the colour distribution to quantify bimodality.
5. Discuss the physical mechanisms (star formation quenching) that produce bimodality.

**Prerequisites:** N05, N06  
**Packages:** `astropy`, `astroquery`, `numpy`, `scipy`, `matplotlib`, `pandas`  
**Data:** SDSS DR18 photometric catalog (queried with `astroquery.sdss`)

**Key concepts:** colour bimodality, red sequence, blue cloud, quenching, stellar populations

---

### Module 3 — Galaxy Populations: Statistical Descriptions

---

#### N09 · `stellar_mass_function.ipynb` — The Galaxy Stellar Mass Function
**Status:** complete  
**Level:** intermediate–advanced | **Time:** 4–5 h

**Learning objectives**
1. Load and explore a real galaxy survey catalog in FITS format (DESI BGS LS10).
2. Understand the concept of the Stellar Mass Function Φ(M★).
3. Apply three independent non-parametric estimators: 1/V_max, SWML, and C⁻.
4. Estimate Poisson uncertainties and compare estimators.
5. Compare the measured SMF to the double-Schechter analytical form.

**Prerequisites:** N06; probability, statistics  
**Packages:** `astropy`, `healpy`, `numpy`, `matplotlib`, `sum_stat` (local)  
**Data:** DESI BGS Legacy Survey DR10 — Comparat et al. (2025), A&A 697, A173

**Key concepts:** stellar mass function, Schechter function, 1/V_max, SWML, C⁻, volume-limited sample

---

#### N10 · `luminosity_function.ipynb` — The Galaxy Luminosity Function
**Status:** proposed  
**Level:** intermediate–advanced | **Time:** 3–4 h

**Learning objectives**
1. Understand the difference between the luminosity function Φ(L) and the stellar mass function.
2. Compute absolute magnitudes from SDSS photometry and Hubble distances.
3. Apply the 1/V_max estimator to measure Φ(M_r) in the r band.
4. Fit a single Schechter function and interpret M★ and the faint-end slope α.
5. Compare the luminosity function across SDSS photometric bands (u, g, r, i, z).

**Prerequisites:** N09  
**Packages:** `astropy`, `numpy`, `scipy`, `matplotlib`  
**Data:** SDSS DR18 spectroscopic catalog

**Key concepts:** luminosity function, absolute magnitude, Schechter parameters, mass-to-light ratio

---

### Module 4 — Large-Scale Structure

---

#### N11 · `angular_power_spectrum.ipynb` — The Angular Power Spectrum of Galaxies
**Status:** complete  
**Level:** advanced | **Time:** 5–6 h

**Learning objectives**
1. Pixelise a galaxy catalog onto a HEALPix sphere map and build the overdensity field.
2. Measure the angular power spectrum C_ℓ with the pseudo-C_ℓ (anafast) method.
3. Subtract shot noise and estimate Gaussian covariance.
4. Compare the measurement to a Limber-approximation HOD theory prediction.
5. Connect C_ℓ to the angular two-point correlation function w(θ).

**Prerequisites:** N09; Fourier analysis, basic probability  
**Packages:** `astropy`, `healpy`, `numpy`, `matplotlib`, `hod_mod` (local), `sum_stat` (local)  
**Data:** DESI BGS Legacy Survey DR10 — Comparat et al. (2025)

**Key concepts:** spherical harmonics, angular power spectrum, HEALPix, pseudo-C_ℓ, Limber approximation, HOD

---

#### N12 · `two_point_correlation.ipynb` — The 3D Two-Point Correlation Function
**Status:** proposed  
**Level:** advanced | **Time:** 4–5 h

**Learning objectives**
1. Understand the definition of the two-point correlation function ξ(r) as excess probability.
2. Generate a random catalog that matches the survey geometry and redshift distribution.
3. Measure ξ(r) and the projected correlation function w_p(r_p) using the Landy–Szalay estimator.
4. Fit a power law ξ(r) = (r / r₀)^−γ and extract the clustering length r₀.
5. Compare your ξ(r) to the angular power spectrum result from N11.

**Prerequisites:** N11  
**Packages:** `numpy`, `scipy`, `matplotlib`, `astropy`, `Corrfunc`  
**Data:** DESI BGS LS10 subsample (same as N09/N11)

**Key concepts:** two-point correlation function, Landy–Szalay estimator, random catalog, clustering length, projected correlation function

---

### Module 5 — Multi-wavelength Observations and Advanced Topics

---

#### N13 · `xray-galaxy-cross-correlation.ipynb` — X-ray / Galaxy Angular Cross-correlation
**Status:** complete  
**Level:** advanced | **Time:** 5–6 h

**Learning objectives**
1. Understand the physical sources of X-ray emission associated with galaxies (AGN, X-ray binaries, CGM).
2. Load and interpret the eROSITA all-sky survey X-ray data.
3. Measure and plot the angular cross-correlation w(θ) between soft X-rays and galaxies.
4. Build a halo model decomposing the signal into one-halo and two-halo terms.
5. Fit the LX–M500c scaling relation parameters to the observed profile.

**Prerequisites:** N11, N12; halo model basics  
**Packages:** `astropy`, `jax`, `numpy`, `matplotlib`, `gga_model` (local)  
**Data:** eROSITA + DESI BGS LS10 — Comparat et al. (2025), Zenodo 15111974

**Key concepts:** angular cross-correlation, eROSITA, CGM, halo model, HOD, scaling relation

---

#### N14 · `photometric_redshifts.ipynb` — Photometric Redshifts
**Status:** proposed  
**Level:** advanced | **Time:** 4 h

**Learning objectives**
1. Understand why photometric redshifts (photo-z) are necessary for large imaging surveys.
2. Build a library of galaxy SED templates (elliptical, spiral, starburst).
3. Implement a template-fitting algorithm: find z and template type that minimise χ².
4. Evaluate photo-z accuracy: bias, scatter (σ_NMAD), and catastrophic outlier rate.
5. Compare template fitting to a simple machine-learning approach (k-NN or random forest).

**Prerequisites:** N08, N10; basic statistics  
**Packages:** `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `astropy`  
**Data:** COSMOS photometric catalog (publicly available)

**Key concepts:** SED template, χ² minimisation, photometric redshift, scatter, catastrophic failure

---

#### N15 · `cosmic_microwave_background.ipynb` — The Cosmic Microwave Background
**Status:** proposed  
**Level:** advanced | **Time:** 4–5 h

**Learning objectives**
1. Understand the origin of the CMB as the surface of last scattering (z ≈ 1100).
2. Download and display the Planck 2018 CMB temperature map using HEALPix.
3. Measure the CMB temperature power spectrum C_ℓ^TT and identify the acoustic peaks.
4. Relate the positions of the peaks to cosmological parameters (Ω_b, Ω_DM, H₀).
5. Compare the measured spectrum to the best-fit ΛCDM model.

**Prerequisites:** N11; basic thermodynamics, cosmology  
**Packages:** `healpy`, `numpy`, `matplotlib`, `astropy`  
**Data:** Planck 2018 CMB maps (ESA public release)

**Key concepts:** surface of last scattering, acoustic oscillations, CMB power spectrum, ΛCDM, Planck mission

---

## 6. Suggested learning paths

Choose a path depending on your background and the duration of your internship.

### Path A — Short internship (3–4 weeks, first contact with astronomy)

```
N01 → N02 → N03 → N04 → N05 → N06
```

Covers tools, stellar physics, and basic galaxy observations. Gives a solid foundation without the statistical or cosmological modules.

### Path B — Standard internship (6–8 weeks, galaxy survey science)

```
N01 → N02 → N03 → N04
                     ↓
              N05 → N06 → N07 → N08
                                  ↓
                             N09 → N10
```

Suitable for a project related to galaxy photometry or spectroscopic surveys.

### Path C — Full curriculum (internship + first year of Master)

```
N01 → N02 → N03 → N04
                     ↓
              N05 → N06 → N07 → N08
                                  ↓
                         N09 → N10 → N11 → N12
                                              ↓
                                    N13 → N14 → N15
```

Covers the entire curriculum from data formats to CMB science.

### Path D — Focus on large-scale structure (existing research context)

```
N01 → N02 → N09 → N11 → N12 → N13
```

Streamlined path for interns working directly on a clustering or cross-correlation project.

---

## 7. Data sources and download

Most notebooks query online databases automatically. The table below lists data sources, access method, and download size.

| Notebook | Dataset | Access | Size |
|----------|---------|--------|------|
| N01, N02 | Gaia DR2 subsample | bundled in repo | ~10 MB |
| N03 | none | — | — |
| N04 | Gaia DR3 | `astroquery.gaia` (online) | ~50 MB |
| N05 | SDSS spectra | `astroquery.sdss` (online) | ~5 MB |
| N06 | NED galaxy distances | `astroquery.ipac.ned` (online) | ~1 MB |
| N07 | THINGS HI rotation curves | direct URL (public) | ~1 MB |
| N08 | SDSS DR18 photometry | `astroquery.sdss` (online) | ~50 MB |
| N09, N11 | DESI BGS LS10 | Zenodo 15111974 | ~500 MB |
| N10 | SDSS DR18 spectroscopic | `astroquery.sdss` | ~100 MB |
| N12 | DESI BGS LS10 | same as N09 | — |
| N13 | eROSITA + DESI BGS LS10 | Zenodo 15111974 | ~500 MB |
| N14 | COSMOS photometry | direct URL (public) | ~200 MB |
| N15 | Planck 2018 CMB maps | ESA PLA (public) | ~500 MB |

### Downloading the DESI BGS LS10 data (N09, N11, N12, N13)

```bash
# Install zenodo_get if needed
pip install zenodo_get

# Download from Zenodo record 15111974
zenodo_get 15111974
```

### Downloading Planck CMB maps (N15)

```bash
# Download the SMICA CMB map (Commander is also fine)
wget https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits \
     -O planck_cmb_smica_2048.fits
```

---

## 8. Troubleshooting

### The environment fails to solve

`conda-forge` packages occasionally have solver conflicts. Try:

```bash
mamba env create -f environment.yml --solver=libmamba
```

If a specific package causes the conflict, install it separately via pip after creating the environment:

```bash
mamba activate tuto_stage
pip install <package>
```

### ImportError for a local package (sum_stat, hod_mod, gga_model)

These packages are not on PyPI or conda-forge. You must clone their repositories and install them manually:

```bash
mamba activate tuto_stage
pip install -e /path/to/package   # editable install
```

Ask your supervisor for access to the repositories.

### Jupyter kernel not found / wrong Python version

Make sure the environment is activated **before** launching Jupyter:

```bash
mamba activate tuto_stage
jupyter lab
```

If the kernel still shows the wrong Python, register it explicitly:

```bash
python -m ipykernel install --user --name tuto_stage --display-name "Python (tuto_stage)"
```

Then restart Jupyter and select the `Python (tuto_stage)` kernel.

### astroquery times out or returns no results

Online queries depend on external servers. If a query fails:
- Check your internet connection.
- Re-run the cell — transient failures are common.
- Use the cached/bundled data if provided (check the notebook's data-loading section).

### Plots do not appear in JupyterLab

Interactive matplotlib requires `ipympl`. At the top of the notebook cell, use:

```python
%matplotlib widget   # interactive, zoomable plots
# or
%matplotlib inline   # static plots (always works)
```

---

## 9. How to add a new notebook

Follow this checklist when contributing a new tutorial:

1. **Choose a slot** in the module sequence (N01–N15) or propose a new one.
2. **Start from the template** — copy the structure of an existing notebook:
   - Title, subtitle, data reference
   - "Learning objectives" section (5–7 bullet points)
   - Numbered theoretical background sections
   - Setup and imports cell
   - Data loading section
   - Analysis sections with explanations
   - Discussion, summary, and exercises
3. **Use only packages in `environment.yml`**. If you need a new package, add it there and test that the environment still builds.
4. **Download data programmatically** — never commit large data files to the repository. Query online services or point to a Zenodo record.
5. **Clear all outputs before committing** — run `Kernel → Restart & Clear Output` in JupyterLab.
6. **Update this README** — add the notebook to the catalog (Section 5) and check the learning paths (Section 6).

---

## 10. References and further reading

### Textbooks

| Title | Authors | Level | Notes |
|-------|---------|-------|-------|
| *An Introduction to Modern Astrophysics* (Carroll & Ostlie) | Carroll & Ostlie | Undergraduate | The standard reference; covers stars, galaxies, cosmology |
| *Galactic Dynamics* | Binney & Tremaine | Graduate | Detailed treatment of dynamics and dark matter |
| *Galaxy Formation and Evolution* | Mo, van den Bosch & White | Graduate | Statistical descriptions of galaxy populations |
| *Modern Cosmology* | Dodelson & Schmidt | Graduate | CMB, power spectra, perturbation theory |
| *Statistics, Data Mining, and Machine Learning in Astronomy* | Ivezić et al. | Undergraduate–Graduate | Statistical methods with Python examples |

### Key papers cited in the notebooks

- Gaia Collaboration, Brown et al. (2018), A&A 616, A1 — Gaia DR2
- Comparat et al. (2025), A&A 697, A173 — DESI BGS LS10 (N09, N11, N13)
- Schmidt (1968), ApJ 151, 393 — 1/V_max estimator
- Efstathiou, Ellis & Peterson (1988), MNRAS 232, 431 — SWML
- Predehl et al. (2021), A&A 647, A1 — eROSITA
- Planck Collaboration (2020), A&A 641, A1 — Planck 2018 results

### Online resources

- [astropy documentation](https://docs.astropy.org/) — the primary Python toolkit for astronomy
- [HEALPix primer](https://healpix.sourceforge.io/documentation.php) — spherical pixelisation
- [SDSS Sky Server](https://skyserver.sdss.org/) — browse and query SDSS data
- [Gaia Archive](https://gea.esac.esa.int/archive/) — full Gaia catalog access
- [NASA/IPAC NED](https://ned.ipac.caltech.edu/) — extragalactic database
- [Zenodo record 15111974](https://zenodo.org/records/15111974) — DESI BGS LS10 data for this tutorial series

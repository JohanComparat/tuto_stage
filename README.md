# Astronomy & Astrophysics Tutorials for Bachelor Interns
### Université Grenoble Alpes — Stage License

A self-contained set of Jupyter notebooks covering astronomy, astrophysics, and cosmology through hands-on work with real observational data. Each notebook is designed for a bachelor-level intern with no prior astronomy experience but a solid background in mathematics and classical physics.

---

## Table of contents

1. [Who is this for?](#1-who-is-this-for)
2. [Prerequisites](#2-prerequisites)
3. [Environment setup](#3-environment-setup) — full (`tuto_stage`) and minimal (`pycl`)
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
| Module 1 (Stars & Galaxy Spectra) | Thermodynamics, electromagnetic radiation, calculus, Doppler effect |

---

## 3. Environment setup

### 3.1 Install Miniforge (once per machine)

[Miniforge](https://github.com/conda-forge/miniforge) is the recommended conda distribution. It ships with `mamba`, a fast drop-in replacement for `conda`. On the lab machines, Miniforge may be already installed at `$HOME/miniforge3` and the environments live under `$HOME/mamba/envs/`.

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

### 3.2 Create the environment

From the root of this repository:

```bash
mkdir $HOME/software
cd $HOME/software
git clone https://github.com/JohanComparat/tuto_stage.git
cd tuto_stage
mamba env create -f environment.yml
```

This installs Python 3.11 and all required packages — `astropy`, `healpy`, `jax`, `pyccl`, `camb`, `Corrfunc`, `pymaster`, `ppxf`, and more. It pulls everything from conda-forge plus a handful of pip-only packages. Expect 5–10 minutes on a first install.

Activate the environment **every time** you open a new terminal:

```bash
mamba activate tuto_stage
```

Deactivate when you are done:

```bash
mamba deactivate
```

### 3.3 Post-install steps

After `mamba env create`, two extra steps are needed.

**Step A — pymaster (NaMaster).** The `pymaster` wheel must be compiled from source. The C libraries it needs (`cfitsio`, `fftw`, `gsl`) are already installed by the conda step above; you just need to pass their location to the build:

```bash
mamba activate tuto_stage
LDFLAGS="-L$CONDA_PREFIX/lib" CFLAGS="-I$CONDA_PREFIX/include" pip install pymaster
```

**Step B — local research packages.** Two packages to install with pip.

```bash
pip install sum_stat    # summary statistics: 1/Vmax, SWML, C⁻ estimators
pip install hod_mod     # Halo Occupation Distribution model + Limber integral
```

### 3.4 Register a kernel so JupyterLab sees the environment

Before launching Jupyter, register the environment as a kernel so JupyterLab can find the correct Python:

```bash
mamba activate tuto_stage
python -m ipykernel install --user --name tuto_stage --display-name "Python (tuto_stage)"
```

This only needs to be done once. After registration, select **Python (tuto_stage)** from the kernel menu in JupyterLab.

### 3.5 Launch Jupyter

```bash
mamba activate tuto_stage
cd $HOME/software/tuto_stage
jupyter lab
```

Your browser will open automatically. Navigate to the `notebooks/` folder and open any notebook.

### 3.6 Update the environment

If new packages are added to `environment.yml`:

```bash
mamba env update -f environment.yml --prune
```

### 3.7 Remove and reinstall from scratch

```bash
mamba env remove -n tuto_stage
mamba env create -f environment.yml
```

### 3.8 Minimal environment for the angular power spectrum notebook (`pycl`)

The notebook `notebooks_evaluated/angular_power_spectrum.ipynb` (BGS angular power spectrum with NaMaster + More+2015 HOD) can also run in a lean, self-contained environment called `pycl`. It contains only the packages that notebook actually requires — about half the size of `tuto_stage` — making it easier to reproduce on a fresh Ubuntu machine.

**Step 1 — create the environment:**

```bash
mamba env create -f environment_pycl.yml
mamba activate pycl
```

**Step 2 — install local research packages** (point to the actual paths on your machine):

```bash
pip install -e /path/to/sum_stat
pip install -e /path/to/hod_mod
```

**Step 3 — pymaster source-build fallback** (only needed if the PyPI manylinux wheel fails, which is rare on Ubuntu x86_64):

```bash
LDFLAGS="-L$CONDA_PREFIX/lib" CFLAGS="-I$CONDA_PREFIX/include" \
  pip install --force-reinstall pymaster
```

**Step 4 — register as a Jupyter kernel:**

```bash
python -m ipykernel install --user --name pycl --display-name "pycl"
```

**Step 5 — smoke-test:**

```bash
python -c "import pymaster, healpy, camb, jax; print('OK')"
```

After these steps, open JupyterLab, select the **pycl** kernel, and run `notebooks_evaluated/angular_power_spectrum.ipynb` from top to bottom.

---

## 4. Repository structure

```
tuto_stage/
├── README.md               ← this file
├── environment.yml         ← full mamba environment (all tutorials)
├── environment_pycl.yml    ← minimal environment for angular_power_spectrum.ipynb
├── notebooks/              ← source notebooks (edit these)
│   │
│   │── Module 0: Tools
│   ├── file_formats_formats.ipynb
│   ├── astronomical_coordinates.ipynb
│   │
│   └── Module 1: Stars and Galaxy Spectra
│       ├── emission_stars_galaxies.ipynb
│       └── redshift_measurement.ipynb
│
└── notebooks_evaluated/    ← pre-executed notebooks with all outputs
    ├── file_formats_formats.ipynb
    ├── astronomical_coordinates.ipynb
    ├── emission_stars_galaxies.ipynb
    └── redshift_measurement.ipynb
```

The `notebooks_evaluated/` folder contains fully executed versions of all notebooks, generated by running each notebook end-to-end with no errors. They serve as a reference showing expected outputs and figures. They are generated with:

```bash
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=600 \
  --output-dir notebooks_evaluated/ \
  notebooks/<name>.ipynb
```

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

**Prerequisites:** N01  
**Packages:** `astropy`, `healpy`, `matplotlib`, `numpy`  
**Data:** Gaia DR2 (bundled subsample)

**Key concepts:** RA/Dec, Galactic latitude/longitude, parallax, proper motion, HEALPix

---

### Module 1 — Stars and Galaxy Spectra

---

#### N03 · `emission_stars_galaxies.ipynb` — From Stellar Emission to Galaxy Spectra
**Status:** complete  
**Level:** introductory–intermediate | **Time:** 4–5 h

**Learning objectives**
1. Derive and plot the Planck blackbody function B_λ(T) and B_ν(T).
2. Apply Wien's displacement law and the Stefan–Boltzmann law to stellar parameters.
3. Compare Planck functions to real MILES stellar spectra; identify spectral absorption features (Balmer series, Ca H&K, Mg b, Na D).
4. Build a Simple Stellar Population (SSP) spectrum by integrating over an IMF, using both blackbody and empirical MILES templates.
5. Load stacked SDSS galaxy spectra (star-forming ELG and quiescent LRG) and identify characteristic emission and absorption lines.
6. Fit blackbody and MILES SSP models to observed galaxy templates by χ² minimisation over SSP age.
7. Perform a penalized pixel-fitting (`ppxf`) analysis of the LRG spectrum to extract stellar velocity and velocity dispersion.

**Prerequisites:** N01, N02; thermodynamics, electromagnetic radiation  
**Packages:** `numpy`, `scipy`, `matplotlib`, `astropy`, `pyneb`, `ppxf`  
**Data:** MILES stellar library v9.1 (auto-downloaded from IAC, ~15 MB); SDSS DR16 galaxy stacks from [qmost_templates](https://github.com/JohanComparat/qmost_templates) (~2 MB)

**Key concepts:** Planck function, Wien's law, Stefan–Boltzmann, spectral type, IMF, SSP, 4000 Å break, emission lines, ppxf, stellar kinematics

---

#### N04 · `redshift_measurement.ipynb` — Measuring Galaxy Redshifts: Spectroscopic and Photometric Methods
**Status:** complete  
**Level:** intermediate–advanced | **Time:** 4–5 h

**Learning objectives**
1. Load real astrophysical SED templates (star-forming, quiescent, AGN type 1 & 2) from the qmost library.
2. Simulate observed spectra at z = [0.1, 0.3, 0.5], rescaled to r = 20, with noise levels of 1%, 10%, 50%.
3. Compute synthetic g, r, i broadband photometry and propagate uncertainties.
4. Understand how **redrock** measures spectroscopic redshifts on DESI spectra: template χ² fitting in log-wavelength space.
5. Understand how **EAZY-py** measures photometric redshifts: χ² minimisation and Bayesian P(z).
6. Compare the precision and limitations of spectroscopic vs. photometric redshifts as a function of SNR and galaxy type.

**Prerequisites:** N03; Doppler effect, basic statistics  
**Packages:** `astropy`, `scipy`, `numpy`, `matplotlib`, `eazy-py`  
**Data:** qmost templates from [github.com/JohanComparat/qmost_templates](https://github.com/JohanComparat/qmost_templates) (~5 MB)

**Key concepts:** cosmological redshift, SED template, χ² fitting, spectroscopic z, photometric z, DESI/redrock, EAZY-py, P(z), σ_NMAD, outlier rate

**References:** Guy et al. (2023) AJ 165, 144 (redrock); Brammer et al. (2008) ApJ 686, 1503 (EAZY); Comparat et al. (2020) (qmost templates)

---

## 6. Suggested learning path

Work through the notebooks in order:

```
N01 → N02 → N03 → N04
```

N01 and N02 cover data formats and coordinate systems — the tools you will use in every subsequent notebook. N03 builds physical intuition from blackbody radiation through stellar population synthesis to real galaxy spectra. N04 applies that spectral knowledge to measure galaxy redshifts, connecting observations to cosmological distances.

---

## 7. Data sources and download

Most notebooks query online databases automatically. The table below lists data sources, access method, and download size.

| Notebook | Dataset | Access | Size |
|----------|---------|--------|------|
| N01, N02 | Gaia DR2 subsample | bundled in repo | ~10 MB |
| N03 | MILES stellar library v9.1 | auto-downloaded from IAC server | ~15 MB |
| N03 | SDSS DR16 galaxy stacks (ELG, LRG) | auto-downloaded from GitHub | ~2 MB |
| N04 | qmost SED templates | auto-downloaded from GitHub | ~5 MB |

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

These packages are not on PyPI. Follow the instructions in Section 3.3 to install them with `pip install -e`. Ask your supervisor for the source paths.

### Jupyter kernel not found / wrong Python version

Make sure the kernel is registered (Section 3.4) and the environment is activated before launching Jupyter:

```bash
mamba activate tuto_stage
python -m ipykernel install --user --name tuto_stage --display-name "Python (tuto_stage)"
jupyter lab
```

Then select the **Python (tuto_stage)** kernel from the kernel menu.

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

1. **Choose a slot** in the module sequence or propose a new one.
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
6. **Update this README** — add the notebook to the catalog (Section 5) and check the learning path (Section 6).

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
- Sánchez-Blázquez et al. (2006), MNRAS 371, 703 — MILES stellar library
- Cappellari & Emsellem (2004), PASP 116, 138 — ppxf
- Comparat et al. (2020) — qmost SED templates
- Guy et al. (2023), AJ 165, 144 — redrock (DESI spectroscopic pipeline)
- Brammer et al. (2008), ApJ 686, 1503 — EAZY photometric redshifts

### Online resources

- [astropy documentation](https://docs.astropy.org/) — the primary Python toolkit for astronomy
- [MILES stellar library](http://miles.iac.es/) — empirical stellar spectra
- [qmost_templates](https://github.com/JohanComparat/qmost_templates) — SDSS DR16 galaxy spectral stacks
- [ppxf documentation](https://pypi.org/project/ppxf/) — penalized pixel-fitting
- [HEALPix primer](https://healpix.sourceforge.io/documentation.php) — spherical pixelisation
- [SDSS Sky Server](https://skyserver.sdss.org/) — browse and query SDSS data
- [Gaia Archive](https://gea.esac.esa.int/archive/) — full Gaia catalog access

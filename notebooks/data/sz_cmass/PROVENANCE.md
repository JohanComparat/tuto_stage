# Provenance of `notebooks/data/sz_cmass/`

Inputs used by `notebooks/sunyaev_zeldovich_prediction.ipynb` (Part 2: reproducing the
ACT × BOSS CMASS thermal-SZ measurement of Schaan et al. 2021 / Amodeo et al. 2021).

## Public model files — vendored from Mop-c-GT

Source repository: **Mop-c-GT** — "Model-to-observable projection code for galaxy thermodynamics"
by S. Amodeo et al., https://github.com/samodeo/Mop-c-GT (accessed 2026-07-09 via the jsDelivr CDN
mirror `cdn.jsdelivr.net/gh/samodeo/Mop-c-GT@master`). The repository has no explicit `LICENSE`
file; it is the companion code of the peer-reviewed paper below and its small numeric input tables
are reused here for teaching, with attribution. Please cite the paper if you reuse these.

| file (here) | original (Mop-c-GT `data/`) | content |
|---|---|---|
| `act_beam_150.txt` | `beam_example.txt` | ACT f150 instrument beam: col1 = θ [rad], col2 = B(θ) [sr⁻¹], normalised so ∫B dΩ = 1 |
| `PthGNFW_M2e+13_z0.56.txt` | same | Battaglia-2012 GNFW thermal-pressure profile for the CMASS stack (M≈2×10¹³ M⊙, z=0.56): col1 R [Mpc], col2 R/R₂₀₀c, col3 P_1h, col4 P_2h, col5 P_1h+2h, in cgs [g cm⁻¹ s⁻²] |
| `twohalo_cmass_average.txt` | same | two-halo term for the CMASS stack: col1 R [Mpc], col2 ρ_2h [g cm⁻³], col3 P_2h [g cm⁻¹ s⁻²] |

The notebook re-implements the forward model (line-of-sight projection → beam convolution →
compensated aperture photometry) natively in numpy/scipy; it does **not** import Mop-c-GT.

## Measured datapoints — `schaan2021_tsz_cmass_illustrative.csv`

**These are illustrative, not the official measurement.** Schaan et al. 2021 (arXiv:2009.05557,
PRD 103, 063513) and Amodeo et al. 2021 (arXiv:2009.05558, PRD 103, 063514) publish the stacked
tSZ profile **only as figures** (Schaan Fig. 8), with no numerical table. The exact measured
values are recoverable only by re-running the public **ThumbStack** pipeline
(https://github.com/EmmanuelSchaan/ThumbStack) on the ACT DR5 + Planck maps and the BOSS DR12
CMASS catalog. The CSV here is the **output of the notebook's own full forward model**
(vendored Battaglia-2012 pressure → line-of-sight projection → ACT beam → CAP) plus fixed-seed
Gaussian noise, tuned to the published shape and total S/N ≈ 11 — i.e. it illustrates a
measurement that the full model reproduces (χ² ≈ 3/6). The *real* CMASS data prefer ≈0.5–0.6× this
thermal pressure (feedback expelling gas from group-scale halos; Amodeo et al. 2021) — reproduced
in the notebook as Exercise 6. Use for pedagogy only; regenerate from ThumbStack for research.

## Sample / instrument facts (from the two papers)

- CMASS: ⟨z⟩ = 0.55 (0.4 < z < 0.7), ⟨M_halo⟩ ≈ 3×10¹³ M⊙, ⟨M*⟩ ≈ 3×10¹¹ M⊙, ≈ 4×10⁵ galaxies.
- ACT DR5 bands: f090 (≈98 GHz, FWHM 2.1′) and f150 (≈150 GHz, FWHM 1.3′), combined with Planck.
- CAP disk radii θ_d = 1–6 arcmin (≈1–4 R_vir). tSZ detected at 11σ, kSZ at 8σ.
- Amodeo best-fit GNFW pressure (their Table 2): P₀ = 2.0₋₀.₈⁺²·⁰, α = 0.8, β = 2.6, x_c fixed.
- Battaglia et al. 2012 default pressure (arXiv:1109.3711, ApJ 758, 75), used for the vendored
  table: P₀ = 18.1(M/10¹⁴)⁰·¹⁵⁴(1+z)⁻⁰·⁷⁵⁸, α = 1, β = 4.35(M/10¹⁴)⁰·⁰³⁹³(1+z)⁰·⁴¹⁵, γ = −0.3,
  x_c = 0.497(M/10¹⁴)⁻⁰·⁰⁰⁸⁶⁵(1+z)⁰·⁷³¹, with cosmology Ω_m=0.25, Ω_b=0.044, h=0.7.

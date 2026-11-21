# readLIGO cleaning pipeline v2: how it works, what was wrong, what changed

November 2026. H1 only: the 56 local O3a files, 5.8 days, 2.17 days of DQ-good data.

> **Status 2026-11-21.** This report describes v2 as of 2026-11-15. The pipeline has since been
> hardened (ISSUES.md section A): Doppler band planning, parallel chunked processing, 4 s gate and
> 8 s edge tapers, in-band flattening, an excess-power veto, a new line finder and stricter checks.
> The current reference is [`cleaning_pipeline.md`](cleaning_pipeline.md); numbers below are from
> 2026-11-15 unless marked.

## 1. Summary

The v1 chain cleaned each 4096 s file on its own: complex heterodyne at the band centre, then
decimation to roughly B, then an STFT χ² glitch cleaner. Its output did not match what the
searches read. Several faults damaged the data:
- fake lines from aliased low-frequency content;
- sample rates that alias for many B;
- a phase restart at every file;
- spikes at every file start;
- a glitch cleaner that fails at 0.1 Hz.

v2 replaces the chain. Each detector's data are processed as contiguous runs, in this order:
1. Broadband conditioning and gating, once.
2. An FFT channelizer, which serves every band from one FFT per 2 h block.
3. One product per band: a real series at **exactly 2B**, heterodyned at **fmin** against **one global epoch**.
   This is the Makefakedata_v5 (MFDv5) convention rustyprune and pyloki use for their simulated data.
   Each product also stores a mask, per-sample weights, the noise per 1800 s segment, a line list, and every setting.

| | v1 | v2 |
|---|---|---|
| Output | complex, at about B, heterodyned at band centre, phase restarts per file, one file per 4096 s | real, at exactly 2B, heterodyned at fmin, one epoch (1238319104), one file per band |
| Low-frequency aliasing (C1) | 450–900× noise in 56–57, 71–72, 519–520 Hz; fake lines | ratio 1.0000000 ± 2e-7, no fake lines |
| Band edges (C2) | −6 dB at the edges; out-of-band tones alias in at 0.13 | flat to 7e-7 up to 2.2 mHz from each edge; nothing aliases in above −123.6 dB |
| Sample rate (C3) | never 2B; 7 of 19 tested widths (0.1–1 Hz) alias | exactly 2B for any B on the 1/1800 Hz grid |
| Glitches (C4) | 36 % of samples lost at 0.1 Hz, 233 Pn left marked good | 0.24 % lost; no sample above 30σ² in the default band |
| File starts (C6) | spikes of up to 884 Pn | none; 3 files and 1 file give identical output (1e-12) without gates; with gates, gate edges move by up to 4 ms |
| MFDv5 equivalence | – | 1.4–2.2e-6 relative rms, phase < 3e-8 rad |
| Hardware injections | P6 only (2F ≈ 80) | P04, P01, P06, P14, P16 and P05 louder than all ~900 off-source trials; amplitude within 0.82–1.22 |
| Runtime (5.8 days, 32 bands) | the full decimation chain repeated per band and per file (E9) | 2.2 min in total, 1.1 GB memory |

## 2. What the searches read

From reading rustyprune (`src/allsky/search.rs`, `src/fft.rs`, `src/simulation/cw_mfdv5.rs`)
and pyloki (`pyloki/simulation/cw_mfdv5.py`, `pyloki/config.py`). Neither code was changed.

```
ROOT/frames/data.npy   float64, shape (N,) or (n_ifo, N), N = duration * 2B
sample n at            t_n = 1238319104 + n / (2B)      (tstart hard-coded)
content                x = 2 Re{ y },  y = [ h(t) exp(-2πi fmin (t - T_h)) ] filtered to [0, B]
normalisation          one scalar variance = sqrt_sx^2 * B for all data and detectors
band                   B = k/10 Hz (k = 1..10), fmin on the 1/1800 Hz grid
```

- The search divides the data by `sqrt(variance)`, takes the analytic signal and resamples to the SSB.
  It then removes `exp(-2πi fmin delay)`.
- So the heterodyne frequency must be **exactly fmin**, and the epoch T_h must be **one constant**.
  Any constant only adds a phase, which the F-statistic absorbs.
- A phase jump inside a segment is not absorbed. A jump of π gives 2F = 0.
- There is no mask input. Zero-filled gaps count as data, so noise 2F ≈ 4η and signal 2F ≈ η²ρ² (ISSUES D1).

MFDv5 conventions, measured directly (`scratchpad/v2/mfdv5`):
- **Epoch:** the heterodyne epoch is the series start time, with no restart at SFT boundaries.
- **Amplitude:** preserved. `A cos(2πft + φ)` maps to `A cos(2π(f − fmin)(t − T0) + …)`.
- **Spectrum:** the band maps to [0, B] with no inversion.
- **Noise:** white to both edges, with variance Sx·B.

## 3. The v2 pipeline, step by step

```
 raw GWOSC files (4096 Hz, NaN where DATA = 0)
   │  inventory: check each file against its own metadata, group into runs of back-to-back files
   ▼
 per run, as ONE series ─────────────────────────────────────────────── conditioning.py
   │  DQ stretches (DATA by default), drop < 16 s
   │  zero-phase FIR high-pass: ≤10 Hz at 120 dB, ≥15 Hz flat (odd-reflection padding at stretch ends)
   │  1 s half-Hann taper at each stretch end
   │  gating: per file with 64 s context, whitened 20–1700 Hz, seed 20σ / extent 6σ,
   │          zero ±0.25 s + 0.25 s Hann tapers  →  one GPS gate list for all bands
   ▼
 conditioned strain (disk, temporary)  ×  gate window  ─────────────── channelize.py
   │  blocks of 7200 s on an epoch-anchored grid, one real FFT per block for ALL bands
   │  per band: bins × Kaiser band-pass H(f), fold to 2B, inverse FFT, global phase, 2·Re
   ▼
 per band: x[n] at t_n = t0 + n/(2B) ───────────────────────── pipeline.py, noise.py
   │  window[n] = RMS of W(t) over the sample's slot; mask[n] = W ≡ 1 over the slot
   │  x = 0 exactly where window = 0
   │  noise: Sx per 1800 s segment = median(periodogram over flat band)/ln 2; lines
   │  narrowband veto: runs with x²/s² > 9 that peak above 25 → zeroed (s² robust per segment)
   ▼
 product  H1_<fmin>-<fmax>Hz_<gps>-<dur>.h5  ───────────────────────── products.py
   │  data, window, mask, sqrt_sx, noise_segments, spectrum, lines, gates, vetoes,
   │  stretches, runs, files, all settings, git commit and versions
   ▼
 export_rustyprune → data.npy (+ mask.npy, window.npy, band.json, readligo_meta.json)
```

### 3.1 Inventory and runs (`inventory.py`, `io.read_gwosc`)
- **Checks per file.** The name must agree with `meta`, Xspacing must be 1/integer, the sample count must match the duration, `Xstart` must equal `GPSstart`, and the DQ mask must have the right length.
- **Runs.** A run is a maximal set of files that start exactly where the previous one ends.
- **Local data:** 56 files in 12 runs [13, 7, 5, 11, 2, 12, 1, 1, 1, 1, 1, 1]. Each run is processed as one series, so file boundaries are never data edges.

### 3.2 DQ stretches
- **Good seconds.** A second is good when every flag in `dq_flags` passes. The default is `("DATA",)`, the CW baseline.
  - `CBC_CAT1` equals `DATA` in O3 and is no longer used (C9).
  - `BURST_CAT2` can be added (about 0.7 % excess noise at 100 Hz without it).
- **NaN guard.** A NaN or inf in a good second stops the run with its GPS time (E5).
- **Hardware injections.** `NO_CW_HW_INJ` is 0 in every good second, so CW injections were on throughout. Each product records this as `cw_hwinj_on_fraction` = 1.0.

### 3.3 High-pass
- **Why.** GWOSC strain is dominated by a 7.93 Hz line (ASD 2e-18) and 8–12 Hz noise, about 110 dB above the 100 Hz floor. Any window multiplies that content and smears it into the analysis bands, including:
  - DQ edges;
  - gates;
  - in v1, the decimation stages.
- **Filter.** A linear-phase Kaiser FIR, applied centred (zero phase, no delay):
  - 6395 taps;
  - −124 dB at 7.93 Hz and −120 dB at 10 Hz;
  - flat to 1.2e-6 above 15 Hz.
- **Stretch ends.** Odd reflection is continuous in value and slope, so no step is filtered.
- **Constraint.** Bands must start at or above 15 Hz.

### 3.4 Gating (`conditioning.detect_gates_run`)
- **Method.** This is gatestrain / PyCBC-style autogating, adapted to O3a.
  - Each file is searched with 64 s of its neighbours.
  - A median-Welch PSD (4 s segments) gives a 4 s whitening FIR over 20–1700 Hz. GWOSC's anti-alias roll-off starts above 1700 Hz.
  - The whitened data are divided by a robust σ (1.4826·median|x|).
  - A transient is a cluster of samples above 6σ that contains a sample above 20σ.
  - It is zeroed from 0.25 s before its first to 0.25 s after its last crossing, with 0.25 s Hann tapers outside.
- **Why these values** (study on these 56 files):
  - **No false gates.** On coloured Gaussian noise with the measured PSD, the statistic is N(0, 1): the maximum in 13 h is 5.74σ, so a 20σ seed never fires by chance.
  - **Glitch power removed.** Glitches carry 4.06× the Gaussian whitened power before gating and none measurable after; T = 20 is at that floor.
  - **Leakage.** Tapers of ≥ 0.25 s leave no leakage above the 0.02 measurement floor.
  - **CW signals untouched.** Software CWs are recovered with amplitude 1.000 ± 0.01 through the gates, and no CW ever seeds a gate: their whitened amplitude is ≤ 0.03σ.
- **Result.** 274 gates (5.25 per hour), 346 s zeroed, 0.24 % of DQ-good time lost (Σ(1 − w²)). The peaks have median 144σ and reach 5900σ.
- **No iteration.** Re-whitening gated data creates 7–42σ artefacts at the tapers, so a second pass would only gate those.

### 3.5 Channelizer (`channelize.py`)
For each band, `x(t_n) = 2 Re[ z(t_n) e^{−2πi fmin (t_n − t0)} ]` on `t_n = t0 + n/(2B)`. Here z is the
positive-frequency band-pass of the conditioned, gated strain.

```
  kernel  h(t) = Kaiser_L(t) · ideal complex band-pass,  L = 3600 s, β = 12.3053 (120 dB)
  block b covers [t0 + b·hop − L/2, t0 + b·hop + hop + L/2),  hop = 3600 s   (T = 7200 s)

  X = rfft(block)                                   one FFT, all bands
  c_k = X[fmin·T + k] · H(k/T)/N,  k = −B·T/2 … B·T + B·T/2
  Y_j = Σ_{k ≡ j mod 2BT} c_k        (exactly how sampling at 2B folds frequencies)
  x   = 2 Re[ e^{−2πi fmin (s_b − t0)} · M · ifft(Y) ]    keep the central hop seconds
```

- **Exact rate for any B.** fmin and B on the 1/1800 Hz grid, with hop and L/2 multiples of 900 s, make fmin·T and 2B·T integers. Every band edge is then an FFT bin, and 2B is reached with no resampling or rounding.
  - Off-grid values are rejected with the two nearest valid values.
  - 6-decimal forms such as `1390.885556` snap to the exact fraction.
- **Band edges.** A real series at exactly 2B has no room for a transition band. With `edge="inside"`, the transition sits inside the band:
  - |H| ≤ 1e-6 everywhere outside [fmin, fmax], so nothing can alias in;
  - the response is flat to 1e-6 on [fmin + g, fmax − g], with g = 7.84/L = 2.18 mHz;
  - the noise variance is therefore 1 − 2.52 mHz/B of Sx·B (0.975 at 0.1 Hz).
  - A signal is fully in the data only while its detector-frame frequency stays in [fmin + g, fmax − g]. For an all-sky search the Doppler shift (±1.06e-4·f) and spin-down move it, so bands must overlap by 2(1.06e-4·f + |ḟ|T + g), not 2g, and each band has a narrower *usable* source range (2026-11-21: `planning.plan_bands`, ISSUES.md A2).
- **Global phase.** `e^{−2πi fmin (s_b − t0)}` is computed with exact fractions. One epoch for everything means no per-file restart.
- **Chunk independence.** Each output sample comes from exactly one epoch-anchored block, so any split of the channelizer calls gives bit-identical output. (Splitting the pipeline's GPS range does not: the cut becomes a tapered data edge; ISSUES.md A14.)
- **Cost.** About 7 s per day for 100 bands. Blocks that are all zero (gaps) skip the FFT.

### 3.6 Window, mask and gaps
- **Window.** A sample stands for the slot [t_n − 1/(4B), t_n + 1/(4B)). Its `window` is the RMS of the strain window W(t) = stretch × edge taper × gates over that slot.
- **Mask.** `mask` is True only if W ≡ 1 over the whole slot.
- **Gaps.** Samples with window 0 are set to exactly 0.0, which is what the searches expect for gaps.
- **Duty.** The duty factor (mean window²) is 0.3739 in every band, because gating is broadband. The mask fraction is 0.3710 at 0.1 Hz and 0.3735 at 1 Hz: a 1 s gate masks a whole 5 s slot at 0.1 Hz.

### 3.7 Noise and lines (`noise.estimate_noise`)
- **Per segment.** Each 1800 s segment, aligned to the epoch, gets a Hann periodogram normalised by its window.
  - Sx_seg = median over the flat band / ln 2: the mean of exponential bins, insensitive to lines and outliers.
  - Segments with duty < 0.5 are skipped.
- **Lines.** A bin is a line if its power, averaged over segments after normalising each by its own Sx, exceeds max(1 + 6/√n_seg, 1.5). Line bins (±1) are excluded and Sx is re-estimated.
- **sqrt_sx.** The product's `sqrt_sx` is √(live-time mean of Sx_seg). This is the normalisation that gives noise 2F a mean of 4 in a search with one variance.
  - The harmonic mean is also stored (the effective noise of a noise-weighted search), with the median and the per-segment table.
- **Check.** The variance of good samples over Sx·B is 0.95–0.99 in clean bands, against 0.975–0.998 expected from the edge roll-off. The v1-era problem was a 3.4 % bias, and errors of 5–1500× in line bands (O4).

### 3.8 Narrowband outlier veto (`noise.narrowband_veto`)
- **What it catches.** Two things broadband gating cannot see:
  - transients concentrated in one band (scattered light at 20–60 Hz);
  - brief leakage of strong lines next to gates (seen in 519–520 Hz).
- **Statistic.** z = x²/s², where s² = median(x²)/0.4549 over the segment's live samples, a robust variance.
  - A stationary line raises s² as much as x², so lines are never vetoed.
  - A short transient leaves s² unchanged.
- **Rule.** Runs of z > 9 that peak above 25 are zeroed, plus one sample on each side.
  - For Gaussian noise, P(z > 25) = 5.7e-7 per sample. CW signals are far below the noise per sample and are never vetoed.
- **Effect.**
  - No vetoes in any hardware-injection band above 50 Hz, and 0–2 in the other bands above 50 Hz.
  - 11 and 40 in the 1 Hz bands at 71 and 56 Hz.
  - 8–33 in 26–38 Hz (155–610 s).
  - 112 in 519–520 Hz (270 s).

### 3.9 Products and export (`products.py`)
- **One product per detector per band.** HDF5, written atomically, with existing files refused unless `overwrite`. It holds:
  - the exact band and heterodyne/amplitude convention strings;
  - the data, window and mask;
  - the noise tables, line list, gate and veto lists, stretches, runs and files;
  - the full configuration, the channelizer and gating settings, the git commit (and whether the tree was dirty), and library versions.
- **`export_rustyprune`.** Writes the layout rustyprune reads:
  - `data.npy` from any `tstart` on the products' grid; rustyprune needs 1238319104.
  - `band.json`, with only keys rustyprune's `deny_unknown_fields` accepts.
  - `mask.npy`, `window.npy` and `readligo_meta.json` for later search changes.
  - Example: `python -m readligo export products/v2_H1/H1_100.68-100.78Hz_*.h5 --out-dir ROOT_frames --duration 345600 --tseg 86400` gives (69120,) float64 with duty 0.297.
- **`sqrt_sx_mode`.** `"live"` (default) writes the true ASD of the live data. `"zero_filled"` writes √duty × ASD, a stopgap that brings unchanged searches' noise 2F back to ≈ 4. Signals still lose about a factor duty, so the real fix is the mask hook (D1).

## 4. Problems in v1 and what was done

| Issue | Problem (evidence) | Fix | Check |
|---|---|---|---|
| C1 | `decimate(ftype="fir")` gave 53–66 dB per alias path; the unfiltered < 10 Hz content (110 dB above floor) flooded bands at 64k ± 8 Hz, 512k ± 8 Hz: 789×, 493×, 901× in 71–72, 56–57, 519–520 Hz; fake lines at 56.07, 71.93, 120.07, 135.93, 199.93, 519.93, 1016.07 Hz | zero-phase 10/15 Hz FIR high-pass, 120 dB; FFT channelizer with −123.6 dB worst-case and −260 dB at 50 Hz offsets | var(no high-pass)/var(high-pass) = 1.0000000 ± 2e-7 in all 11 test bands; no fake line in any product |
| C2 | fs = B, −6 dB at the edges, aliasing of edge content (gain 0.13), ghost lines ±B | output at 2B with the transition inside the band | flat to 7e-7 beyond 2.18 mHz; ≤ −123.6 dB for every out-of-band frequency (unit test: < −120 dB at 0.1 mHz – 3 Hz offsets) |
| C3 | rounding of decimation factors: B = 0.15, 0.35, 0.55, 0.6, 0.75, 0.9 always alias, 0.4 sometimes | exact grid arithmetic (`Fraction`), fs = 2B by construction | B = 0.1, 0.15, 0.37, 0.55, 1.0 at fmin 20–2000 Hz: exact rate, flat gain (unit tests); real products at 0.15, 0.37, 0.55 Hz |
| C4 | STFT cleaner at 0.1 Hz: 14 columns per file, 63.6 % usable, samples up to 233 Pn kept as good | broadband gating once at 4 kHz + narrowband outlier veto | default band: 0 samples > 30σ² (max 18.4); 0.24 % lost; duty factor equal at 0.1 and 1 Hz |
| C5 | nearly empty files marked all-good | fixed in v1 (2026-11-11); in v2 stretches < 16 s are dropped | – |
| C6 | each file processed alone: start spikes up to 884 Pn | runs of contiguous files processed as one series | 3 files vs 1 file: identical to 1e-12 (unit test); z near 44 internal boundaries 0.91–1.07 (expected 1) |
| C7 | heterodyne phase restarted per file: frequency bias up to 1.2e-4 Hz, SNR² loss up to 60 % | one global epoch, exact phase | MFDv5 phase error ≤ 3e-8 rad; every recovered injection peaks at exactly its frequency (0/T) |
| C8 | 410 samples per file at 0.1 Hz: 6 s junctions | single grid t_n = t0 + n/(2B) | by construction; checked |
| C9 | `DATA & CBC_CAT1` (= `DATA` in O3), misleadingly named | `dq_flags=("DATA",)` default, configurable (`BURST_CAT2`) | recorded in every product |
| O1, O2 | filename collisions; no metadata | fixed in v1; v2 names `H1_100.68-100.78Hz_<gps>-<dur>.h5` and records everything | round-trip test |
| O3 | complex, band-centre, per-file, ≈B output | real, fmin, global epoch, 2B, one series; `export_rustyprune` | matches the input contract; MFDv5 equivalence |
| O4 | no noise estimate; sample variance 3.4 % low, 5–1500× off in line bands | Sx per 1800 s from the flat band, lines excluded | variance ratio as expected; lines listed |
| O5 | amplitude convention undocumented | `amplitude_convention`, `heterodyne_convention` attrs; this report | – |
| V1 | stored v1 products used bound = 15, default 10 | default `bound = 15` in v1, recorded | – |
| V3 | STFT vs gatestrain comparison unfinished | redone on O3a: gating loses 0.24 % vs 36 % (v1 at 0.1 Hz); after gating the STFT cleaner finds 0 artefact columns (248 before) | – |
| V4 | only 100–101 Hz validated | default band 100.68–100.78 characterised; band-edge tests | – |
| V7 | baseline P6 only | new baseline table (§5.3) | – |
| E2 | `rich` undeclared, `pycbc` unused, pins above the tested versions | core deps numpy/scipy/h5py; `legacy`, `validation`, `dev` extras | – |
| E3 | docs drifted | docs rewritten for v2; v1 doc corrected | – |
| E4 | no tests | 47 pytest tests (channelizer, gating, noise, veto, products, export, end to end) | `pytest` passes |
| E6 | `np.repeat(..., 4096)`; unused `bound` | duty from the 1 Hz mask; `bound` documented as ignored | – |
| E7 | file not closed; Xspacing unused | context manager; `read_gwosc` validates Xspacing | – |
| E8 | hard-coded OneDrive paths, string path joins | `$READLIGO_DATA_DIR` / `$READLIGO_OUT_DIR` or arguments; `Path` | – |
| E9 | 90 % of time in `np.exp` + first decimation, per band | one FFT per block for all bands | 32 bands × 5.8 days in 34 s |
| E10 | copyrighted PDF and local settings in git | untracked and ignored | – |
| G6 | ~6e7 output files for a full search | one file per detector per band | – |
| G7 | absolute phase unvalidated | MFDv5 (3e-8 rad) and hardware injections (phase offsets ≤ 0.12 rad for the strong four) | – |

## 5. Validation

Scripts are in `scripts/validation/`; their outputs are in `products/validation/` (not in git).

### 5.1 Channelizer on synthetic data
- **Tone sweeps and aliasing** (`tests/test_channelize.py`, and the study's 219-tone sweep):
  - in-band |G| − 1 < 2e-6 and phase < 1e-7 rad beyond the guard;
  - out of band, at most −120 dB from 0.1 mHz to 3 Hz from the edges;
  - a 1e10 tone at 7.93 Hz gives < 1e-12 in 56–57, 71–72 and 100–101 Hz.
- **MFDv5 equivalence** (`mfdv5_equivalence.py`). MFDv5 broadband frames through the channelizer, compared with MFDv5 narrowband frames:

  | Band | Interior relative rms | \|gain\| − 1 |
  |---|---|---|
  | 100.68–100.78 Hz | 2.2e-6 | −1.1e-7 |
  | 100.48–100.98 Hz | 1.4e-6 | +1.2e-8 |
  | 100.18–101.18 Hz | 1.4e-6 | −4.8e-9 |

  This is the same floor the study measured between MFDv5 and an ideal FFT reference.
- **Chunking.** Five arbitrary splits are bit-identical to the whole run.

### 5.2 Real H1 data (`real_data_checks.py`)
32 bands over the full 5.8 days.

| Check | Result |
|---|---|
| C1 high-pass ratio (11 bands) | 0.99999988 – 1.00000018 |
| v1 fake lines present | none |
| Rate | exactly 2B in all bands (1/5, 3/10, 37/50, 11/10, 2, … Hz); uniform grid |
| Samples > 30σ² (Sx·B) | 0 in every band above 50 Hz except one sample at 30.0σ² in 519–520 Hz; 3–5 in the 26–38 Hz bands (noisy segments, which the per-segment Sx down-weights) |
| Mean z near 44 internal file boundaries | 0.91–1.07 in bands without strong lines; 1.14 and 1.26 in the 519.9 and 37.8 Hz line bands, whose overall means are 1.04 and 1.30 |
| Variance / (Sx·B) | 0.95–0.99 in clean bands (expected 0.975 at 0.1 Hz, 0.998 at 1 Hz) |

### 5.3 Hardware injections (`hwinj_recovery.py`)
- **Statistic.** A known-template coherent JKS F-statistic on each product, over its masked samples, weighted by per-segment Sx.
- **Phase model.** Exact LAL SSB timing and spin-down from the GWOSC O3April1 parameters, plus a circular orbit for P16 and P17.
- **Expected 2F.** `4 + Aᵀ M A` from the product's own noise and mask.
- **Background.** 2F at 780–1090 frequency offsets ≥ 1 mHz from the injection, with the template inside the flat band. These trials are independent, since 1/T = 2 µHz.

| P | f (Hz) | 2F v2 | expected | amplitude | phase (rad) | background mean / max | 2F, v2 without gating (background mean) | raw-strain study, gated (expected / measured) |
|---|---|---|---|---|---|---|---|---|
| P04 | 1390.93 | **81.4** | 80.9 | 1.03 | −0.08 | 4.06 / 20.1 | 77.7 (4.2) | 79.2 / 70.6 |
| P01 | 848.93 | **75.1** | 77.4 | 1.01 | +0.03 | 4.10 / 18.7 | 61.2 (4.2) | 76.1 / 65.2 |
| P06 | 145.45 | **74.3** | 75.5 | 0.99 | −0.08 | 4.04 / 19.4 | 122 (16.0, peak 17/T off) | 71.0 / 74.8 |
| P14 | 1991.28 | **71.1** | 77.8 | 0.98* | −0.12 | 3.87 / 21.6 | 71.3 (3.9) | 79.0 / 73.8 |
| P16 (binary) | 234.57 | **50.8** | 37.9 | 1.22 | +0.23 | 4.01 / 17.0 | 33.8 (13.9, p = 0.03) | 36.9 / 46.2 |
| P05 | 52.81 | **39.9** | 52.0 | 0.82 | +0.17 | 4.04 / 17.0 | 25.4 (45.4, lost) | 33.0 / 26.3 |
| P00 | 265.57 | 25.4 | 14.3 | 1.55 | −0.07 | 4.14 / 18.5 | 18.2 (40.9) | 13.7 / 23.9 |
| P07, P09, P03, P11 | | 13–17 | 6–14 | | | | | |
| P02, P08, P10, P12, P17 | | 2–12 | 5–14 | | | | | |

\* After the GWOSC anti-alias gain g = 0.0071 at 1991 Hz, which scales signal and noise alike.

How to read the table:
- **Strong four** (P04, P01, P06, P14): recovered at expected strength, louder than every off-source trial. Amplitudes are within 3 % and phases within 0.12 rad. The 2F peak sits exactly at the injection frequency, so there is no frequency bias.
- **P16 (binary) and P05:** also louder than every off-source trial.
- **P00, P07, P09, P03:** weak evidence (p = 0.001–0.007), consistent with expected 2F of 10–14.
- **The rest:** consistent with their expected 2F < 15. P03 and P08, named in ISSUES V6, cannot be recovered from 5.8 days of H1.
- **Without gating,** the low-frequency bands are no longer Gaussian: the background mean is 14–80 instead of 4. P05 is lost, P16 becomes marginal, and P06 peaks at the wrong frequency.
- **The v1 baseline** was P6 2F ≈ 80 (V7). v2 matches it, and adds five more injections.

**Independent check with LAL** (`lal_fstat_crosscheck.py`):
- 1 Hz products around each injection, made by the same pipeline, are cut into 89 SFTs (1800 s, duty ≥ 98 %) with `MakeSFTsFromREAL8TimeSeries`.
- `lalpulsar_ComputeFstatistic_v2` scores them with its own running-median noise estimate.

| P | ComputeFstatistic_v2 | readLIGO cwtools, same segments |
|---|---|---|
| P04 | 68.7 | 70.1 |
| P14 | 67.9 | 73.1 |
| P01 | 61.5 | 59.0 |
| P06 | 60.8 (resampling: 59.6) | 58.4 |
| P05 | 40.7 | 39.4 |

The two agree to 2–7 %, as expected from their different noise estimators. This confirms the product's timing, heterodyne, amplitude and SFT compatibility independently of cwtools.

For the binary P16, ComputeFstatistic_v2 gives 2F ≈ 1 with every orbit-argument convention tried, while cwtools gives 42.5. The circular-orbit model cwtools uses (also used in the raw-strain study) recovers P16 at the expected strength with phase offset ≈ 0.2 rad, so the model matches the injection. The mismatch is in how the orbit options were passed to ComputeFstatistic_v2, which is still open.

## 6. How to run

```bash
pip install -e .            # once (also fixes V5: notebooks then import this checkout)
python -m readligo clean --data-dir /path/to/O3a_marvin --out-dir products/v2_H1 --ifo H1 \
       --band 100.68 0.1 --band 145.4 0.1          # or --bands-file scripts/o3a_h1_bands.txt
python -m readligo export products/v2_H1/H1_100.68-100.78Hz_*.h5 --out-dir ROOT_frames \
       --tstart 1238319104 --duration 345600 --tseg 86400
pytest                                              # 47 tests, ~15 s
python scripts/validation/hwinj_recovery.py products/v2_H1
python scripts/validation/real_data_checks.py products/v2_H1 --data-dir /path/to/O3a_marvin
```

From Python:

```python
from readligo import BandSpec, PipelineConfig, run_pipeline, read_product, band_around
cfg = PipelineConfig(data_dir=..., out_dir=..., ifo="H1",
                     bands=[BandSpec(100.68, 0.1), band_around(145.4484, 145.4514, 0.1)])
p = read_product(run_pipeline(cfg)[0])
p.data, p.window, p.mask, p.times, p.sqrt_sx, p.tables["noise_segments"]
```

The products of this report are in `products/v2_H1/`:
- 32 bands: the 16 hardware-injection bands, the default band, and the C1, C3 and C6 test bands;
- `products/v2_H1_nogate/` holds the no-gating comparison, and `products/v2_H1_hwinj_1Hz/` the 1 Hz bands for the LAL check;
- `products/rustyprune_H1_100.68/` holds an example export.

## 7. What is left

- **Search codes** (not touched, as agreed). Each item maps to what the product already provides:
  - **D1:** AND the mask (or use the window) in `valid_mask`/`segment_window`, and define a policy for empty segments. The product provides `mask` and `window`.
  - **D2:** pyloki should check that the rate, uniform sampling and start time match its config.
  - **D3:** rustyprune needs a real-data `prepare`, with a sidecar giving tstart, fs, f_het, IFO, mask and variance. `readligo_meta.json` has all of these.
  - **D4:** recalibrate thresholds on real duty and noise with off-source nulls. The `hwinj_recovery.py` background is a template for that.
  - **D5:** a variance per segment and per IFO. The product provides `noise_segments`.
- **L1** (no data downloaded yet). The pipeline is detector-agnostic for H1 and L1. Use the same epoch for both, so their samples align, and export them together.
- **Line handling (G1).** Lines are listed and excluded from Sx but not removed. Bands near very strong lines (violin modes at 500 and 1000 Hz, 60 Hz harmonics) carry their leakage. Gates add brief leakage, which the veto catches. Line subtraction before gating would be the next step.
- **Low-frequency non-stationarity (G2).** Below 40 Hz, 6–8 of the 104 segments have Sx above twice the median. The per-segment Sx is stored, so a noise-weighted search (D5) uses it. An unweighted search sees the live-time average.
- **P16 with ComputeFstatistic_v2:** the binary-orbit option convention is still open (§5.3).
- **V2.** The notebook `injection_recovery_real_data.ipynb` was not changed, because it has your uncommitted edits. `hwinj_recovery.py` is the correct real-data scoring: it zero-fills gaps, uses the exact template and applies the mask.
- **V5.** Run `pip install -e .` so notebooks stop importing the stale installed copy.
- **G4.** The local data are 0.68 % of the planned 180 days. The pipeline itself scales: one FFT per 2 h block, about 1 min per day per detector.

## 8. Files

| File | Role |
|---|---|
| `readligo/io.py` | `read_gwosc` (validated `GwoscFile`); v1 `loaddata` |
| `readligo/inventory.py` | `scan_directory`, `contiguous_runs` |
| `readligo/config.py` | `BandSpec` (exact grid), `band_around`, `PipelineConfig`, `ChannelizerParams`, `NoiseParams`, `VetoParams` |
| `readligo/conditioning.py` | `GateParams`, `RunData`, `condition_run`, `detect_gates_run`, `ConditionedStrain`, `output_window` |
| `readligo/channelize.py` | `FFTChannelizer` |
| `readligo/noise.py` | `estimate_noise`, `robust_scale`, `narrowband_veto`, `gwosc_antialias_gain` |
| `readligo/products.py` | `BandProduct`, `write_product`, `read_product`, `export_rustyprune` |
| `readligo/pipeline.py`, `cli.py` | `run_pipeline`; `python -m readligo clean|export` |
| `readligo/preprocessing.py`, `process_run.py` | v1, kept for comparison |
| `tests/` | pytest suite and the synthetic GWOSC file writer |
| `scripts/o3a_h1_bands.txt` | the band list used here |
| `scripts/validation/` | `hwinj_recovery.py`, `cwtools.py`, `hwinj_o3a.json` (GWOSC parameters), `real_data_checks.py`, `mfdv5_equivalence.py`, `lal_fstat_crosscheck.py` |

# readLIGO issue tracker

Issues found in the November 2026 review of the cleaning pipeline and how its
output feeds the pyloki (`pruning_copy`) and rustyprune CW searches.
Tick each box when its issue is fixed and its **Done when** check passes.

**Severity:** 🔴 critical · 🟠 high · 🟡 medium · ⚪ low

**Abbreviations:**
- B: bandwidth, `f_max − f_min`
- fs: output sample rate
- f_h: heterodyne frequency
- Pn: median noise power
- DQ: data quality

Evidence scripts are in the review scratchpad. This directory is temporary and
may be deleted, so copy anything you want to keep:
`/private/tmp/claude-502/-Users-saifa-Desktop-readLIGO/f21890f3-b9cd-4aa5-9bc9-9eb4f89349f9/scratchpad/`

---

## Suggested order

| Step | Issues | Why first |
|---|---|---|
| 1. Quick fixes | E1, O1, C5, E5, C3 (assert only), O2 | Small changes. Stop crashes, overwrites and silently wrong masks |
| 2. Redesign the chain | C6, C1, C4, C2, C3, C7, C8, O3, O4 | The main problems in the cleaning. **Done in v2, 2026-11-15** (`docs/cleaning_report.md`) |
| 3. Search codes | D1–D5 | Needed before any real-data search |
| 4. Validation | V1–V7 | Prove the new chain works |
| 5. Scale and science | G1–G8 | Needed for a real 180-day search |
| Anytime | E2–E10, C9, O5 | Housekeeping |
| 6. Attack review of v2 | A1–A15 | Found 2026-11-15 by attacking the v2 pipeline. **Fixed 2026-11-21** (A9 partly); `docs/cleaning_pipeline.md` |

---

## C. Cleaning pipeline (`readligo/preprocessing.py`)

### [x] C1 🔴 Content below 10 Hz aliases into bands and creates fake lines
- **Where:** `preprocessing.py:89-104` (`heterodyne_downsample`)
- **Problem:**
  - `signal.decimate(..., ftype="fir")` gives only about 53–66 dB of
    stopband per alias path.
  - GWOSC strain is not high-passed. Its 6.5–10 Hz content, including a
    7.93 Hz line, is about 100 dB above the in-band floor.
  - Bands with f_h ≈ 64k ± (6.5–10) Hz or 512k ± (6.5–10) Hz therefore get
    flooded.
- **Evidence:** ratio of median |y|² without vs with a 12 Hz high-pass, on
  GPS 1238319104:

  | Band (Hz) | Ratio |
  |---|---|
  | 71–72 | 789× |
  | 56–57 | 493× |
  | 519–520 | 901× |
  | 100–101 | 1.0× |

  - Fake narrow lines at 56.07, 71.93, 120.07, 135.93, 199.93, 519.93 and
    1016.07 Hz.
  - About 25 of the 1 Hz bands are inflated more than 2× (14 more than 10×,
    4 more than 100×).
  - About 129 of the 0.1 Hz bands in 12–400 Hz are inflated more than 2×.
- **Fix:**
  - Zero-phase high-pass at 10–12 Hz before heterodyning.
  - Decimation filters with at least 120–140 dB stopband.
- **Done when:** the bands above show a ratio of about 1, and none of the
  fake lines appear.
- **Scripts:** `orchestrator/check.py`, `dsp-heterodyne-decimation/e3d_map.py`
- **Resolved 2026-11-15 (v2):** Zero-phase FIR high-pass (≤10 Hz at 120 dB, flat ≥15 Hz) and the FFT channelizer (≤ −123.6 dB aliasing; −260 dB at 50 Hz offsets). On real H1 the 11 test bands give var(no high-pass)/var(high-pass) = 1.0000000 ± 2e-7, and no v1 fake line appears in any product.

### [x] C2 🟠 Output is critically sampled (fs = B), so band edges are distorted
- **Where:** `preprocessing.py:97-104`
- **Problem:** the filter's −6 dB point sits on the band edge.
  - The response is flat only for |Δf| ≤ 0.42B.
  - At the edge a signal keeps 0.50 of its amplitude, i.e. 25% of its power.
  - A tone 0.04B outside the band aliases in at gain 0.133 at the opposite
    edge.
  - Lines produce ghost copies ±B away.
- **Fix:** output at 2B (pyloki requires this anyway), with guard bands and
  overlapping bands.
- **Done when:** a tone swept across the band has flat gain to the edges, and
  out-of-band tones are suppressed by at least 120 dB.
- **Resolved 2026-11-15 (v2):** Output at 2B with the filter transition inside the band: flat to 7e-7 beyond g = 2.18 mHz from each edge, |H| ≤ 1e-6 everywhere outside the band. Edges roll off within g, so adjacent search bands should overlap by ≥ 2g.

### [x] C3 🟠 Sample-rate logic is fragile and sometimes aliases
- **Where:** `preprocessing.py:97-104` (the `while current_fs / bandwidth > 13`
  loop and `int(np.round(current_fs / bandwidth))`)
- **Problem:**
  - B = 0.6 gives fs = 0.5 Hz, and B = 0.9 gives fs = 0.889 Hz. Both are
    below B, so the band aliases.
  - B = 0.4 gives fs = 1/3 Hz for 35–61% of fmin values on the 1/1800 Hz
    grid, depending on the fmin range (0.5 Hz otherwise). The final factor
    is round(1 Hz / B) = round(2.5): float error in `f_max - f_min` tips it
    to 3, and numpy's round-half-to-even sends an exact 2.5 to 2.
  - No B ever produces fs = 2B.
  - Sweep of B = 0.1–1.0 Hz in 0.05 steps, fmin = 20–500 Hz:

    | Result | B (Hz) |
    |---|---|
    | fs_out = B exactly | 0.1, 0.2, 0.25, 0.5, 0.8, 1.0 |
    | fs_out > B (safe, not exact) | 0.3, 0.45, 0.65, 0.7, 0.85, 0.95 |
    | fs_out < B (aliases) | 0.15, 0.35, 0.55, 0.6, 0.75, 0.9 (always); 0.4 (60% of fmin) |

- **Requirement:** for **any** user-requested B in 0.1–1 Hz (most bands
  will be 0.1–0.5 Hz), the output covers exactly [f_min, f_max], with no
  rounding of B and no dependence on fmin. This needs C2 (output at 2B, so
  the band is flat to its edges) as well as an exact fs.
- **Fix:** choose the decimation factors from an explicit target fs and
  assert `fs_out >= 2B` (the quick fix is `assert fs_out >= B`). For B where
  4096 / 2B is not an integer (e.g. 0.15, 0.35), use rational resampling
  (`scipy.signal.resample_poly`) rather than rounding the factor.
- **Done when:** for a sweep of B over 0.1–1 Hz (including non-round values
  such as 0.15, 0.37, 0.55) and fmin over 20–2000 Hz, fs_out equals the
  target exactly and a swept tone has flat gain from f_min to f_max.
- **Quick fix done 2026-11-13:** `heterodyne_downsample` raises `ValueError`
  if `fs_out < B` (relative tolerance 1e-9, since e.g. 20.1 − 20.0 is
  slightly above 0.1). B = 0.6, 0.9 and some B = 0.4 bands now stop with an
  error instead of aliasing. The target-fs redesign is still open.
- **Resolved 2026-11-15 (v2):** Band edges are exact fractions on the 1/1800 Hz grid, and the rate is 2B by construction (no decimation factors, no resampling). Tested at B = 0.1, 0.15, 0.37, 0.55, 1.0 for fmin 20–2000 Hz (exact rate, flat gain), and on real data at 0.15, 0.37, 0.55 Hz. Off-grid values are rejected with the nearest valid ones.

### [x] C4 🟠 Glitch cleaner fails at 0.1 Hz bandwidth
- **Where:** `preprocessing.py:138-270` (`clean_narrowband`, `nperseg=64`)
- **Problem:** `nperseg` is counted in samples.
  - At 0.1 Hz each window is 640 s, so a 4096 s file has only about 14
    columns.
  - The per-row percentile noise estimate becomes unreliable.
- **Evidence:**
  - The usable fraction of DQ-good data is 0.615 at 0.1 Hz, against 0.923
    at 1 Hz.
  - In the default band (100.68, 100.78), 14 of 32 fully-good files keep
    samples above 30σ² marked good.
  - 5 files keep start spikes of 37–347 Pn marked good.
- **Fix:** detect transients once, on whitened broadband strain or a band of
  at least 1 Hz. Apply one GPS gate list to every sub-band, using
  inverse-Tukey gating at 4 kHz in the style of gatestrain.
- **Done when:** the 0.1 Hz usable fraction is at least the 1 Hz value, and
  no samples above 30σ² remain marked good in the default band.
- **Script:** `critic/c5_default_band.py`
- **Resolved 2026-11-15 (v2):** Broadband gating once at 4 kHz (274 gates, 0.24 % lost), plus a narrowband outlier veto per band. Default band: 0 samples above 30σ² (max 18.4). The duty factor is identical at 0.1 and 1 Hz (0.3739); the mask fraction differs only by slot length (0.3710 vs 0.3735).

### [x] C5 🟠 Nearly-empty files are marked all-good, or crash the batch run
- **Where:** `preprocessing.py:203-205`
  (`if n_good < 2: return data.copy(), np.ones(n, dtype=bool), {}`)
- **Problem:**
  - The mask comes back True everywhere, so DQ-bad zeros and gate
    transients up to about 8000× the median are marked good.
  - The empty `diag` makes `data_loss_stats` raise `KeyError: 'good_cols'`.
    `process_run` always passes `loss_stats=True`, so batch runs crash.
- **Evidence:** this fires on 1238241280 and 1238380544 for B ≤ 0.2 Hz.
- **Fix:** return an all-False mask, or the DQ mask, and a `diag` that has
  every expected key.
- **Done when:** both files process at B = 0.1 without error and produce
  masks with no DQ-bad samples marked good.
- **Resolved 2026-11-11:**
  - When fewer than 2 DQ-good columns exist, the series is zeroed and fully
    masked, and `diag` now has every key (`sxx_scaled` is NaN because no
    noise estimate is possible).
  - `plot_cleaning` handles that NaN case.
  - Both files now process at 0.1 and 0.2 Hz with 0 samples marked good and
    no crash.
  - A normal file's output is bit-identical to before the fix.
  - A lone DQ-good column (n_good = 1) is reported as artifact loss in
    `data_loss_stats`.

### [x] C6 🟠 Each 4096 s file is processed on its own, leaving start spikes
- **Where:** `process_run.py` loop and `process_band`
- **Problem:** every file start acts as a hard gate on the raw strain and
  leaves a transient at sample 0.
  - At 100–101 Hz the χ² test always flags it.
  - In narrow or low bands it often doesn't: in 30.0–30.1 Hz, 9 of 40 file
    starts are unflagged, one carrying 884 Pn.
  - Adjacent GWOSC files are sample-continuous.
  - A high-pass does **not** fix this. In the critic's test it let more
    start spikes through as good.
- **Fix:** concatenate contiguous files and process each contiguous segment
  as one series.
- **Done when:** no transient appears at internal file boundaries in any
  tested band.
- **Resolved 2026-11-15 (v2):** Runs of back-to-back files are processed as one series. The same data as 3 files or as 1 file give identical output to 1e-12 (unit test). On real H1 the mean z near the 44 internal boundaries is 0.91–1.07.

### [x] C7 🟡 Heterodyne phase restarts at every file
- **Where:** `preprocessing.py:89-90` (`t = np.arange(len(data)) / fs`
  restarts at 0 in each file)
- **Problem:** this produces a phase staircase with step c = frac(4096·f_h).
  - Frequency bias of c/4096 Hz, up to 1.22e-4 Hz.
  - SNR² factor of sinc²(c), worst case 0.405.
  - Ghost peaks at ±1/4096 Hz.
  - 100–101 Hz is unaffected only because 4096·f_h is an integer there.
  - The default band (100.68, 100.78) has c = 0.08: a +1.95e-5 Hz bias and
    about 2% SNR² loss.
- **Fix:** heterodyne against one global GPS epoch, and store f_h and the
  epoch in the output.
- **Done when:** an injected CW in the default band is recovered with no
  frequency bias across files.
- **Resolved 2026-11-15 (v2):** One global epoch (default 1238319104) with exact phase arithmetic. Phase error against MFDv5 ≤ 3e-8 rad, and every recovered hardware injection peaks exactly at its frequency.

### [x] C8 🟡 Time grid is not uniform across files at B = 0.1
- **Problem:** a 4096 s file at 0.1 Hz gives 410 samples, so a 6 s junction
  appears between consecutive files. Concatenated series are not uniformly
  sampled, but pyloki and rustyprune assume they are.
- **Fix:** solved by C6 (contiguous processing) plus C3 (an exact fs).
- **Done when:** `np.diff(gps_times)` is constant across file boundaries.
- **Resolved 2026-11-15 (v2):** Single grid t_n = epoch + n/(2B) for the whole product.

### [x] C9 ⚪ BURST_CAT2 veto not applied
- **Where:** `preprocessing.py:26-53` (`apply_dq_mask` uses
  `DATA & CBC_CAT1`, which equals `DATA` in O3)
- **Problem:** this is optional. Without the BURST_CAT2 veto there is about
  0.7% excess noise at 100 Hz. Using `DATA` is the correct CW baseline, but
  the `CBC_CAT1` name is misleading.
- **Fix:** decide whether to use the veto; rename or comment the mask
  either way.
- **Resolved 2026-11-15 (v2):** Default `dq_flags = ("DATA",)`; `CBC_CAT1` is dropped (equal to DATA in O3). `BURST_CAT2` can be added in the config. The choice is recorded in every product.

---

## O. Output format (`readligo/process_run.py`)

### [x] O1 🟠 Output filenames collide for sub-Hz bands
- **Where:** `process_run.py:23`
  (`f"_f_band_{int(f_band[0])}-{int(f_band[1])}Hz.hdf5"`)
- **Problem:** every sub-Hz band within the same integer Hz gets the same
  filename, so each run silently overwrites the last.
- **Fix:** put the exact band edges in the name, e.g. `f"{f_min:.6f}"`, and
  refuse to overwrite an existing file without a flag.
- **Done when:** (100.68, 100.78) and (100.78, 100.88) write to different
  files.
- **Resolved 2026-11-11:**
  - New `output_path()` writes band edges in their shortest exact form, so
    distinct floats always give distinct names. Example:
    `..._f_band_100.68-100.78Hz.hdf5`. Existing 1 Hz names such as
    `_100-101Hz` are unchanged.
  - `save_hdf5_file` and `process_ligo_files` take
    `overwrite: bool = False` and raise `FileExistsError` instead of
    overwriting.
  - The batch run checks every output path before processing anything.
  - Tested in a scratch directory.

### [x] O2 🟠 Output carries almost no metadata
- **Problem:** the only attrs written are `source_file`, `f_band` and
  `fs_new`. Missing:
  - IFO
  - f_h
  - heterodyne epoch
  - processing parameters (`nperseg`, `fap`, `bound`, filter design)
  - package and library versions
  - noise level
- **Fix:** write all of these as HDF5 attrs.
- **Done when:** every product can be regenerated exactly from its own
  attrs.
- **Resolved 2026-11-13:** `process_ligo_files` now writes, as root attrs:
  - source: `detector` (must be H1 or L1), `source_gps_start`,
    `source_duration`, `strain_channel`;
  - processing (`processing_attrs`): `dq_flags`, `fs_in`, `f_h`,
    `heterodyne_epoch`, `heterodyne_convention`, `decimation_factors`,
    `decimation_ftype`, `nperseg`, `noverlap`, `stft_window`,
    `noise_quantile`, `fap`, `bound`;
  - provenance: `readligo_git_commit`, `readligo_git_dirty`, Python, numpy,
    scipy and h5py versions, `created_utc`.

  The fixed choices (DQ flags, window, quantile, filter type) are now
  module constants in `preprocessing.py`, used by both the code and the
  record. Output data are bit-identical to before. Noise level is left to
  O4. Products made before this fix (V1) still have no record.

### [x] O3 🟠 Output conventions don't match pyloki and rustyprune

| | readLIGO now | Searches expect |
|---|---|---|
| Data type | complex | real (rustyprune); pyloki also accepts positive-frequency complex |
| Heterodyne frequency | band centre | fmin (`pyloki/config.py:437`) |
| Phase origin | each file | one global epoch |
| Sample rate | ≈ B, varies with B | exactly 2B (`pyloki/config.py:420-425`) |
| Layout | one file per 4096 s | one contiguous, uniformly sampled series |
| Gaps | `sample_mask` | no mask input (see D1) |

- **Fix:** export one series per IFO per band: real, at 2B, heterodyned at
  fmin on a global epoch, with a mask and noise estimate.
- **Reference:** an adapter prototype exists and has been verified (phase
  error ≤ 1.7e-5 rad against MFDv5, per-segment 2F within 0.3%):
  `pyloki-interface/readligo_pyloki_adapter.py`
- **Done when:** the output loads into pyloki and rustyprune without an
  adapter, and an injection test passes.
- **Resolved 2026-11-15 (v2):** Products are real, at 2B, heterodyned at fmin on one global epoch, one series per band, with mask, window and noise. `export_rustyprune` writes `data.npy` plus a `band.json` with only accepted keys, so no adapter is needed. Running the searches on it is left for the search-code work (D1–D3).

### [x] O4 🟠 No noise-level estimate, and a naive estimate is biased
- **Problem:** the searches normalise by one scalar variance, Sx·B.
  - A plain sample variance of the output is 3.4% low because of the edge
    roll-off. That gives about 4× more false alarms at the final threshold.
  - In bands with lines, a data-derived variance is off by 5× to 1500×.
- **Fix:** estimate Sx from the flat centre of the band, excluding lines,
  and store it per segment.
- **Resolved 2026-11-15 (v2):** Sx per 1800 s segment from the flat part of the band (median/ln 2), with lines excluded. `sqrt_sx` is the live-time mean, with the harmonic mean, median and per-segment table stored. Var/(Sx·B) is 0.95–0.99 in clean bands, as expected from the edge roll-off.

### [x] O5 ⚪ Amplitude convention is undocumented
- **Problem:** the complex output carries A/2 of the signal amplitude. The
  equivalent real series is 2·Re(y).
- **Fix:** document this in `docs/preprocessing.md` and in the output
  attrs.
- **Resolved 2026-11-15 (v2):** `amplitude_convention` and `heterodyne_convention` attrs in every product; documented in the report.

---

## D. Search codes (pyloki and rustyprune; separate repos)

### [ ] D1 🔴 No mask input, so zero-filled gaps are counted as data
- **Where:**
  - pyloki: `/Users/saifa/Downloads/pruning_copy/pyloki/fft.py:393-431`
    (`valid_mask_segments`)
  - rustyprune: `/Users/saifa/Downloads/rustyprune/src/fft.rs:158-167`
- **Problem:** with duty factor η:
  - noise 2F comes out ≈ 4η;
  - signal 2F comes out ≈ η²ρ² instead of ηρ².
  - At η = 0.82 and threshold 89.55, detection probability drops from 0.853
    to 0.519.
- **Fix:** AND a dilated readLIGO mask into the existing `valid_mask` hook.
  Define a policy for fully masked segments, whose covariance is singular.
- **Done when:** on gapped data, noise 2F matches χ²₄ statistics and an
  injection's 2F scales as ηρ².
- **Status 2026-11-15:** readLIGO side done: products and export carry `mask`/`window` for the `valid_mask`/`segment_window` hook.

### [ ] D2 🟠 pyloki doesn't validate its input time series
- **Where:**
  - `pyloki/config.py:420-425`
  - `pyloki/core/resampling.py:167-204` (index arithmetic assumes uniform
    time)
- **Problem:**
  - `ts.fs` is never compared with `cfg.fs`.
  - With gapped `absolute_times`, every sample after the first gap is read
    at the wrong time.
- **Fix:** assert that fs matches, timestamps are uniform and the start
  time matches.
- **Status 2026-11-15:** readLIGO side done: the grid is exactly uniform from `tstart`, and `readligo_meta.json` states fs and tstart for pyloki to check against.

### [ ] D3 🟠 rustyprune has no real-data input path
- **Where:**
  - `rustyprune/src/allsky/search.rs:357-381`
    (`tstart: DEFAULT_TSTART = 1238319104`)
  - `Root::open` checks only the dtype and shape of `data.npy`
  - `TODO.md` item 4b ("prepare from frames")
- **Fix:** add a real-data `prepare` step with a metadata sidecar (tstart,
  fs, f_het, IFO, mask, variance), and validate the sidecar on open.
- **Status 2026-11-15:** readLIGO side done: `readligo_meta.json` is the sidecar (tstart, fs, f_het, epoch, IFOs, duty, noise).

### [ ] D4 🟡 Threshold schedules assume simulated, full-duty data
- **Where:** `pyloki/allsky/search.py:65-75` (hard-coded for "0.1 Hz band,
  90 × 2 d")
- **Problem:** the thresholds assume χ²₄ statistics at full duty.
- **Fix:** recalibrate on real duty and noise using off-source nulls (see
  V6).
- **Status 2026-11-15:** readLIGO side: duty and per-segment noise are stored; `hwinj_recovery.py`'s off-source background shows how to build nulls.

### [ ] D5 🟡 Single scalar variance for all data
- **Problem:** one `sqrt_sx` comes from the config. Real data needs
  per-detector, per-segment variance.
- **Fix:** accept a variance per segment and per IFO, taken from O4.
- **Status 2026-11-15:** readLIGO side done: `noise_segments` table (Sx per 1800 s) per detector.

---

## V. Validation and notebooks

### [x] V1 🟠 Stored products were made with `bound=15`, which isn't recorded
- **Problem:** the products on disk were made with `bound=15`, set in a
  notebook cell; the default is 10.
  - All 56 files reproduce bit-for-bit at `bound=15`.
  - At the default, 11 files differ, and a default re-run silently
    overwrites them.
- **Fix:** decide on one value, make it the default, and record it in the
  attrs (see O2).
- **Resolved 2026-11-15 (v2):** v1 default `bound` is now 15 (`DEFAULT_BOUND`), matching the stored products; recorded since O2.

### [ ] V2 🟠 Real-data injection score of 2.84 is meaningless
- **Where:** `notebooks/injection_recovery_real_data.ipynb`
- **Problem:**
  - It FFTs `data[sample_mask]`, which splices across gaps.
  - It reads bin 819, which is 0.2245 Hz, not 0.2 Hz.
  - Its covariance ignores the mask.
- **Evidence:** scored correctly, the result is 125 against an optimal 139,
  ranking 1st of 4096.
- **Fix:** zero-fill instead of splicing, use the correct bin, and apply
  the mask to the covariance.
- **Status 2026-11-15:** Superseded by `scripts/validation/hwinj_recovery.py`, which zero-fills gaps, uses the exact template and applies the mask. The notebook is unchanged because it has uncommitted edits.

### [x] V3 🟡 `gating.ipynb` doesn't justify STFT cleaning over gatestrain
- **Problem:**
  - "Data removed" counts DQ-bad time on the STFT side. Like-for-like,
    loss is 7.75% for STFT cleaning against 0.48% for gatestrain.
  - The gatestrain port whitens without a window, which creates gates at
    both file edges.
  - The port uses O1/O2 parameters, not O3a.
  - The confusion-matrix comparison stopped on a KeyboardInterrupt, and the
    complementarity test never ran.
  - On in-band SNR², 4 kHz gating wins by about 3–15%.
- **Fix:** redo the comparison with a windowed port and O3a parameters, and
  finish both tests (this feeds the C4 decision).
- **Resolved 2026-11-15 (v2):** Redone on O3a by the gating study: broadband gating loses 0.24 % of data against 36 % for the v1 cleaner at 0.1 Hz. After gating, the STFT cleaner flags 0 columns at 100–101 Hz (248 before).

### [x] V4 🟡 Only one band and width was ever validated
- **Problem:**
  - Only (100, 101) at 1 Hz has been tested, and band edges never were.
  - The default search band (100.68, 100.78) has never been characterised.
- **Fix:** make (100.68, 100.78) the reference configuration, and add
  band-edge tests.
- **Resolved 2026-11-15 (v2):** Default band 100.68–100.78 characterised on real data (noise, lines, outliers, injections nearby). Band-edge behaviour is tested in `tests/test_channelize.py`.

### [ ] V5 🟡 Notebooks run against a stale installed copy
- **Problem:**
  - The notebooks import a non-editable `pip install .` copy, which
    contains `projection.py`.
  - `projection.py` was never committed; it exists only in `build/lib` and
    site-packages.
  - Its `w_eff` uses a symmetric Hann window; it should be periodic, to
    match scipy.
- **Fix:** reinstall with `pip install -e .`. `projection.py` was dropped
  in E1, so the window bug no longer applies.
- **Status 2026-11-15:** Run `pip install -e .`; not done here because it changes your Python environment.

### [ ] V6 🟡 No end-to-end search on real data yet
- **Fix:**
  - Use hardware injections as regression targets: P6 145.445 Hz, P5
    52.808 Hz, P3 108.857 Hz, P8 ~190 Hz, in 0.1 Hz bands.
  - Inject software CWs into the raw 4 kHz strain.
  - Calibrate thresholds against an off-source null.
- **Status 2026-11-15:** Hardware injections are now regression targets (`hwinj_recovery.py`, with an off-source background). P03 and P08 are not recoverable from 5.8 days of H1, so the targets are P04, P01, P06, P14, P16 and P05. An end-to-end search waits for D1–D3.

### [x] V7 ⚪ Hardware-injection baseline
- **Current state:**
  - P6 is recovered from the current output: 2F ≈ 80, and a 5-harmonic
    statistic of 53.6 against an off-source maximum of 25.9 over 6,822
    trials. Amplitude is within 7%.
  - P5 and P3 are recovered more weakly. P8 is untested.
- **Use:** keep these numbers as the before/after benchmark for every fix.
- **Resolved 2026-11-15 (v2):** New baseline (report §5.3): P04 81.4, P01 75.1, P06 74.3, P14 71.1, P16 50.8, P05 39.9, each louder than every off-source trial.

---

## E. Engineering

### [x] E1 🟠 `import readligo` is broken
- **Where:** `readligo/__init__.py:13` (`from readligo.projection import ...`)
- **Problem:** `projection.py` is not in the repo.
- **Fix:** commit it or remove the import (see V5).
- **Resolved 2026-11-11:** dropped the projection import and its six
  `__all__` entries. Nothing in the repo used them, and pyloki does its own
  antenna projection. `import readligo` now works from a fresh checkout.

### [x] E2 ⚪ Dependencies are wrong
- **Where:** `pyproject.toml`
- **Problem:** `rich` is used (`process_run.py:4`) but undeclared; `pycbc`
  is declared but unused.
- **Resolved 2026-11-15 (v2):** Core deps numpy/scipy/h5py with tested minimums; `legacy` (matplotlib, rich), `validation` (lalsuite) and `dev` extras; `readligo` console script.

### [x] E3 ⚪ Docs have drifted from the code
- `process_band` returns 5 values; the docs say 4.
- `plot_cleaning` draws 3 panels; the docs say 4.
- The README quick start uses `time['start']` with `tvec=True`, which fails.
- `docs/preprocessing.md:43` says the 39.4th percentile equals the mean. For
  an exponential it is half the mean. The code is correct; only the
  explanation is wrong.
- **Resolved 2026-11-15 (v2):** Docs rewritten for v2 (`docs/README.md`, `project_context.md`, `pipeline_architecture.md`, `README.md`); `docs/preprocessing.md` marked as v1 and corrected.

### [x] E4 🟡 No tests
- **Fix:** add regression tests for C1, C3, C5, O1 and the hardware
  injections.
- **Resolved 2026-11-15 (v2):** 47 pytest tests: channelizer (rate, gain, aliasing, chunking, C1), conditioning, noise and veto, products and export, and end to end on synthetic GWOSC files.

### [x] E5 🟡 No NaN / finite guard
- **Problem:** NaNs currently all fall in DATA = 0 seconds, but nothing
  enforces this.
- **Fix:** assert `np.isfinite` on all samples that the mask marks good.
- **Resolved 2026-11-13:** `apply_dq_mask` raises `ValueError` (with the
  count and first offending second) if any DQ-good sample is non-finite.
  Without it, one NaN in a good second made the whole file 100 % "good"
  and passed 96 NaN samples downstream. All 56 raw files pass the check,
  and output on a normal file is bit-identical to before.

### [x] E6 ⚪ `data_loss_stats` hard-codes 4096 and ignores `bound`
- **Where:** `preprocessing.py:357` (`np.repeat(dq_mask_1hz, 4096)`), and
  the `bound` parameter at `:278`
- **Fix:** pass `fs` through, and use `bound` or drop it.
- **Resolved 2026-11-15 (v2):** Duty from the 1 Hz mask (no 4096); `bound` documented as ignored.

### [x] E7 ⚪ `io.py` issues
- **Where:**
  - `io.py:41`: `h5py.File` is opened without a context manager.
  - `io.py:51`: `Xspacing` is read but never used, so 4096 Hz is assumed
    elsewhere.
- **Resolved 2026-11-15 (v2):** Context manager in `read_hdf5`; `read_gwosc` reads and validates Xspacing.

### [x] E8 ⚪ `process_run.py` portability
- Hard-coded OneDrive paths at lines 9-10.
- `data_dirname + files.name` at line 60 breaks without a trailing slash;
  use `files` directly, since it is already a `Path`.
- `loss_stats=True` is always passed (line 72).
- **Resolved 2026-11-15 (v2):** Directories come from arguments or `$READLIGO_DATA_DIR` / `$READLIGO_OUT_DIR`; `Path` joins; `loss_stats` is a parameter.

### [x] E9 ⚪ Performance
- **Problem:** about 90% of runtime is `np.exp` plus the first decimation
  stage, repeated for every band.
- **Fix:** a one-FFT-per-file channeliser would be about 1000× faster.
  Needed for scale (see G6).
- **Resolved 2026-11-15 (v2):** One FFT per 2 h block serves every band: 32 bands × 5.8 days in 34 s; the whole pipeline takes 2.2 min.

### [x] E10 ⚪ Files that shouldn't be committed
- `docs/gatestrain.pdf`: copyrighted APS paper (Steltner, Papa & Eggenstein
  2022).
- `.claude/settings.local.json`: local settings.
- **Fix:** remove them from git and add them to `.gitignore`.
- **Resolved 2026-11-15 (v2):** Both files untracked (`git rm --cached`) and added to `.gitignore`; they stay on disk.

---

## G. Gaps for a real search

- [ ] **G1 🟠 No line handling anywhere in the chain.** Data-derived
  variance is off by 5× to 1500× in line bands. The 100.30000 Hz feature is
  real h(t) content. Add line lists and a line veto.
  **2026-11-15:** Partly: a line list per band, lines excluded from Sx, and the veto removes gate-induced line leakage. Lines are not removed (no line subtraction).
  **2026-11-15 (attack):** The stored line list is enough for a line veto: in 28–28.5 Hz, 178 of 61,275 off-source templates have 2F > 30; away from flagged lines 1 remains (0.28 expected). See A7 for its limits.
- [ ] **G2 🟡 Non-stationarity not examined.**
  **2026-11-15:** Partly: Sx per 1800 s segment is stored; below 40 Hz, 6–8 of 104 segments are above twice the median.
- [ ] **G3 🟡 Mask propagation through pyloki's SSB resampling is
  untested.**
- [ ] **G4 🟠 Data coverage is far too small.** Local data are H1 only and
  cover 0.68% of the planned 180 days. Get at least 30–60 contiguous days of
  H1+L1.
  **2026-11-21:** H1 is covered: the whole O4a H1 run (160.6 DQ-good days) was cleaned and validated on marvin. L1 is still missing. The cleaning always runs over a whole observing run, so effects seen only in the 5.8 local days are not issues.
- [ ] **G5 🟡 No duty-factor planning.**
  **2026-11-15:** Duty per product and per export is recorded (0.374 over the 5.8 days; 0.297 for 4 days from 1238319104).
- [x] **G6 🟡 Output layout won't scale.** The current per-file, per-band
  layout gives about 6e7 output files.
  **2026-11-15:** One file per detector per band.
- [x] **G7 🟡 Absolute phase not validated against an external
  reference.**
  **2026-11-15:** Validated against MFDv5 (3e-8 rad) and the hardware injections (phase offsets ≤ 0.12 rad for P04, P01, P06, P14).
- [x] **G8 🟡 None of the proposed fixes are validated yet.** Each fix
  above needs its **Done when** check run.
  **2026-11-15:** Every Done-when check above was run; results are in `docs/cleaning_report.md` §5.

---

## A. Attack review of the v2 pipeline (2026-11-15)

The v2 pipeline was attacked from four sides: two independent code reviews
with reproductions, experiments on the real H1 data (75 extra bands next to
strong lines and at 20–40 Hz), search-level tests (2F over 40–110 thousand
off-source templates per band), and scaling tests to 180 days.
Scripts and outputs: `attack/main/`, `attack/agent1/`, `attack/agent2/` in the
scratchpad named at the top.

### [x] A1 🔴 The work folder can delete user data, and parallel jobs corrupt each other
- **Where:** `pipeline.py:134, 144, 288-294`
- **Problem:** the scratch folder is `<work_dir>/<ifo>` (default
  `out_dir/_work/<ifo>`). An existing folder of that name is reused, and it is
  removed with `rmtree` when the run ends, on success or error.
  - Raw data kept in `<root>/H1` are erased by a run with `work_dir=<root>`.
  - Two jobs for the same detector and `out_dir` write and delete the same
    files.
- **Evidence (synthetic files only):**
  - A raw folder `<root>/H1` with one strain file and a text file is gone
    after the run.
  - Two jobs started together both finish without error, and both outputs
    differ from a solo run (8e-4 rms).
  - Started 2–6 s apart, the second dies with `FileNotFoundError`. Two CLI
    jobs: one died in 6 of 6 trials (SIGBUS or mmap error).
- **Fix:** make a fresh, uniquely named scratch folder per run and delete
  only that.
- **Scripts:** `attack/main/workdir_verify.py`, `concurrent.py`;
  `attack/agent1/t05_workdir.py`, `t05b_parallel_jobs.py`
- **2026-11-21 fixed:** each run makes its own folder `readligo-<ifo>-XXXXXXXX`
  inside `work_dir` (`tempfile.mkdtemp`) and deletes only that; the parent is
  removed only if the run created it and it is empty. Products are written
  under a unique temporary name and linked into place, so a parallel job's
  product is never silently replaced and a crash leaves no `.part`. Test
  `test_work_dir_is_private_and_parallel_matches_serial`: a raw folder
  `<root>/H1` with a text file is untouched after a run with
  `work_dir=<root>`. A job stopped with SIGTERM also removes its folder
  (`test_sigterm_removes_scratch`); only SIGKILL leaves it.

### [x] A2 🔴 Bands are not planned for the Doppler shift of an all-sky search
- **Where:** `config.py` (`band_around`, no planner),
  `docs/cleaning_report.md:151-152`, the C2 note above ("overlap by ≥ 2g");
  rustyprune `src/config.rs:136, 304-308`, `src/allsky/search.rs:376-390`
- **Problem:** rustyprune searches every frequency in `[fmin, fmin + B]` and
  requires that range to equal the data band. The searched frequency is the
  source frequency; at the detector it moves by ±1.06e-4·f (plus ḟ·T). The
  product holds a signal only while it is inside `[fmin + g, fmax − g]`.
  - Usable source-frequency width per band is `B − 2(1.06e-4·f + g)`:

    | f (Hz) | B = 0.1 | B = 0.2 | B = 0.3 | B = 0.5 | B = 1.0 |
    |---|---|---|---|---|---|
    | 20 | 0.091 | 0.191 | 0.291 | 0.491 | 0.991 |
    | 100 | 0.074 | 0.174 | 0.274 | 0.474 | 0.974 |
    | 300 | 0.032 | 0.132 | 0.232 | 0.432 | 0.932 |
    | 450 | 0 | 0.100 | 0.200 | 0.400 | 0.900 |
    | 700 | 0 | 0.047 | 0.147 | 0.347 | 0.847 |
    | 1000 | 0 | 0 | 0.084 | 0.284 | 0.784 |

  - The 5.8-day hardware-injection test cannot show this: each injection's
    band was centred on its track.
- **Evidence:** a Doppler-modulated tone through the real channelizer, 180
  days: 5 mHz inside a 100.0–100.1 Hz band keeps 26–63 % of its power; at
  the centre of a 1000.0–1000.1 Hz band it keeps 33 %.
- **Fix:** add a band planner: given the search range, T and the ḟ range, it
  returns overlapping bands with `B(f) ≥ search width + 2(1.06e-4·f +
  |ḟ|T/2 + g)` and the usable source range of each. Store that range in the
  product and in the export, and correct the two doc lines. The search side
  must restrict its frequency range to it (with D3).
- **Scripts:** `attack/agent2/a7_doppler_margin.py`, `attack/main/tails.py`
- **2026-11-21 fixed (cleaning side):** `readligo/planning.py`
  (`usable_range`, `plan_bands`, `python -m readligo plan`). Bands overlap so
  their usable source ranges meet exactly; `bandwidth="auto"` takes the
  narrowest k/10 Hz width keeping ≥ 50 % usable. 20–2000 Hz, 180 d, fdot in
  [−1e-9, 0]: 9,448 bands, B from 0.1 Hz (20 Hz) to 0.9 Hz (2 kHz), 3,547 Hz
  in total; a fixed 0.1 Hz fails above 375 Hz. `BandSpec.usable` is stored as
  `usable_source_band` in each product and in `readligo_meta.json` (with
  `usable_doppler_only`). Doc lines corrected (`cleaning_report.md` §3.5,
  `cleaning_pipeline.md` §3). The search must still restrict itself to the
  usable range (D3).

### [x] A3 🟠 A DQ drop 62–63 s before a file boundary can gate away the rest of that file
- **Where:** `conditioning.py:279, 299` (`detect_gates`)
- **Problem:** each file's glitch search starts 64 s before the file. A
  stretch that ends less than 3 s into that block makes `b - guard`
  negative. The slices then wrap around and take in the whole block, gaps
  included, and the zeros pull the PSD and sigma down.
- **Evidence (Gaussian noise, no glitch):**

  | DATA off before the boundary | Gap | Result |
  |---|---|---|
  | 60, 61 or 64 s | 1800 s | correct, 0 gates |
  | 62 or 63 s | 1800 s | 1 gate, 2357.5 of 2358 live seconds zeroed, sigma 0.30 of normal |
  | 62 s | 600 s | sigma 14 % low, no false gate |

  Same on real strain with the gap set in memory: 100 % of 2358 s zeroed.
- **How often:** 2 of 4096 positions per stretch end, so about 0.3 times per
  detector in 180 days.
- **Fix:** skip a stretch when `b - a <= 2 * guard`, and the same for
  `interior`.
- **Scripts:** `attack/main/negslice_verify.py`;
  `attack/agent1/t02_negative_slice.py`, `t02c_…real.py`, `t02d_offsets.py`
- **2026-11-21 fixed:** `block_psd` skips a stretch unless one PSD segment
  fits after the guards; the interior bounds are checked. Test
  `test_detect_gates_short_stretch_near_block_start` (sigma within 5 % of
  the clean value, no gate).

### [x] A4 🟠 The pipeline does not scale to 180 days and thousands of bands
- **Where:** `channelize.py:363-389`, `pipeline.py:174, 222-281`,
  `conditioning.py:610-623`, `noise.py:187-190`
- **Problem and evidence:**

  | Limit | Now | At 180 days |
  |---|---|---|
  | Every band's output is held in RAM | fine for 75 bands | 25 MB per 0.1 Hz band, 245 GB for 9,800 bands |
  | `output_window` loops over all gates for all touched slots | negligible (274 gates) | 2.1 s per band at 4,000 gates, ×3 per doubling; and it is recomputed for every band although it depends only on B |
  | `robust_scale` selects each segment with a full-length mask | negligible | 15.4 s per 0.1 Hz band |
  | Work files are float64 at 4 kHz for each whole run | 7.5 GB | about 360 GB |
  | No resume, no append | – | any error deletes the work folder; adding data means redoing everything |

  The two per-band steps alone come to about 60 h for 9,800 bands (taking
  8,500 gates, the local rate over 180 days). The channelizer and
  conditioning scale fine (75 bands in 150 s here).
- **Fix:** write each band's samples to disk block by block; compute the
  window once per bandwidth with a sorted sweep; slice segments by index in
  `robust_scale`; store conditioned strain as float32 or condition block by
  block; let a run reuse a kept work folder and append new time.
- **Script:** `attack/main/bench_scale.py`
- **2026-11-21 fixed:**
  - bands are channelized in chunks of at most `chunk_gb`, each block written
    straight into a memory-mapped scratch array per bandwidth;
  - `workers` processes: strain check per file, conditioning and gating per
    run, channelizing per block, window/noise/vetoes/product per band
    (bit-identical to serial, tested);
  - the window is computed once per bandwidth and linearly in the gates
    (8,500 gates over 180 days: 0.17 s, identical to the old result in 60
    random cases); `robust_scale` and `estimate_noise` are vectorised;
  - `work_dtype="float32"` halves the scratch (products change by ≤ 8e-6 of
    the rms); `skip_existing` resumes an interrupted job.
  - Local: 5.8 days, 21 bands in 47 s with 6 workers (was ~190 s serial).
  - Full scale on marvin, O4a H1 (3,901 files, 160.6 live days), 379 bands
    from 20 to 2000 Hz, 64 workers: 26.7 min in total (conditioning 12–19
    min, channelizing 287 s in one pass, products 426 s); all six hardware
    injections recovered at amplitude 0.96–1.00 and phase within 0.05 rad
    (`docs/cleaning_pipeline.md` §6.4, `scripts/validation/hwinj_o4a_check.py`).
  - Not done: appending new data to an earlier run (a resume repeats the
    conditioning); conditioning is parallel per run, so the longest run sets
    its wall time.

### [x] A5 🟠 Gates and gap edges leak strong lines into nearby bands
- **Where:** `conditioning.py` (`gate_window`, edge tapers); the veto in
  `noise.py` only mops up
- **Problem:** multiplying the strain by a gate spreads a strong line over
  about ±1 Hz for the length of the gate. Bands inside the violin-mode
  regions get a burst at every gate and every stretch start.
- **Evidence (real H1, mean z = x²/s² in fully good samples within 3 samples
  of a gate; 1 is normal):**

  | Band (0.1 Hz wide) | Veto off | Veto on | Vetoes (274 gates) | 4σ outliers vs Gaussian, veto on |
  |---|---|---|---|---|
  | 498, 522, 990, 1005 Hz | 1.0–1.1 | 1.0–1.1 | 0 | 0.9–1.7× |
  | 502 | 12 | 3.3 | 210 | 18× |
  | 508 | 21 | 4.3 | 272 | 32× |
  | 510 | 11 | 3.2 | 207 | 18× |
  | 515 | 10 | 2.4 | 109 | 18× |
  | 520 | 3.6 | 1.1 | 26 | 4.7× |
  | 1000 | 44 | 7.3 | 381 | 63× |

  - Synthetic proof: Gaussian noise plus one stationary line 1e6 times the
    floor (1/1800 Hz bins). A band 0.55 Hz away has z = 3.65 next to gates
    (1.17 without the line); a band 1.55 Hz away is unaffected.
  - The 60 Hz and calibration lines are too weak to show it.
- **Affected:** about 500–521 Hz and around 995–1000 Hz, sampled every
  2–5 Hz (roughly 3 % of 20–1000 Hz).
- **Fix:** keep strong lines through the gate: estimate the narrow line
  component around each strong PSD peak, gate only the remainder, and add
  the line back. Until then, flag these bands in the product.
- **Scripts:** `attack/main/run_lines.py`, `leak.py`, `stats.py`,
  `synth_line_leak.py`
- **2026-11-21 fixed:** gate taper 0.25 → 4 s, edge taper 1 → 8 s. Real H1,
  veto off, mean z next to gates / next to stretch edges / 4σ outliers vs
  Gaussian:

  | Band | taper 0.25 s, edge 1 s | 2 s, 4 s | 4 s, 8 s (default) |
  |---|---|---|---|
  | 502 Hz | 13.8 / 4.8 / 92× | 1.8 / 1.0 / 9× | 1.01 / 1.06 / 3.4× |
  | 508 Hz | 23.0 / 10.4 / 148× | 1.7 / 0.9 / 4× | 0.97 / 0.92 / 2.2× |
  | 1000 Hz | 48.0 / 14.3 / 272× | 9.2 / 1.1 / 82× | 1.36 / 0.85 / 3.9× |
  | 498, 990 Hz (controls) | 1.0 / 0.9–1.0 / 1.3–1.7× | same | same |

  The duty factor drops from 0.3739 to 0.3711 (0.75 %). Each product now
  stores `near_gate_mean_z`, `near_edge_mean_z`, `far_mean_z`. Line
  subtraction was not needed.

### [x] A6 🟡 `export_rustyprune` writes wrong or unusable settings without complaint
- **Where:** `products.py:284-334`
- **Problem and evidence:**
  - **Noise level of a sub-span.** `sqrt_sx` is the whole product's, not the
    exported span's. Product with Sx = 1 then 9 over two 10-day halves: both
    exports write 5.0, so rustyprune's normalised variance is 0.20 and 1.79.
  - **tstart.** Any on-grid tstart is accepted, but `band.json` has no tstart
    and rustyprune hard-codes 1238319104 (`cw_mfdv5.rs:34`). An export at
    +5 s would be read 5 s late.
  - **Silent cases.** A NaN `sqrt_sx`, zero overlap with the product (all
    zeros, `sqrt_sx` 0 in zero-filled mode) and a band that is not k/10 Hz
    are all written without a warning.
  - **Overwrite.** Only `data.npy` is guarded; a hand-edited `band.json` is
    replaced.
- **Fix:** recompute `sqrt_sx` from `noise_segments` over the span; raise on
  tstart ≠ 1238319104 (until rustyprune takes one), on non-finite or zero
  `sqrt_sx`, on zero overlap and on band ≠ k/10; guard every file.
- **Scripts:** `attack/agent2/a6_export.py`, `a9_misc.py`
- **2026-11-21 fixed:** `sqrt_sx` is computed over the exported span from
  `noise_segments` (`span_sqrt_sx`); `tstart ≠ 1238319104` raises unless
  `allow_any_tstart`; a band that is not k/10 Hz, a span without live data
  or without a noise estimate raise; every output file is guarded. Test
  `test_export_checks` (1 then 9: both halves give the right level).

### [x] A7 🟡 The line list and Sx have blind spots
- **Where:** `noise.py:93-124`
- **Problem and evidence:**
  - **Weak lines are never listed on long runs.** The threshold is
    `max(1 + 6/√n, 1.5)`. Beyond 144 segments (3 days) it stays at 1.5. With
    8,640 segments a line of ratio 1.37 is 34σ and is not flagged.
  - **A strong line off a bin centre inflates `sqrt_sx` at 0.1 Hz.** Its Hann
    leakage reaches tens of bins, but only ±1 bin is excluded.
    `sqrt_sx²`/true is 1.04, 1.14 and 2.78 for line amplitudes 1e3, 1e4 and
    1e5 (units of √(Sx·0.1 Hz)); at 1 Hz it stays below 1.012.
  - **False lines with few segments.** With 1–4 segments, 3–18 % of clean
    0.1 Hz products and 16–77 % of 1 Hz products list a false line.
  - **`min_duty` uses the Hann²-weighted duty.** A segment with 78 % live
    data but a 400 s gap in its middle is dropped (duty 0.45); one with only
    its central 420 s live is kept.
- **Fix:** threshold on significance with the exact Gamma tail; iterate
  flag-and-re-estimate with an exclusion width that grows with line
  strength; test `min_duty` on the plain live fraction.
- **Scripts:** `attack/agent2/a13_weak_lines.py`, `a17_line_bias_widths.py`,
  `a2_noise_bias.py`, `a19_hann_duty.py`
- **2026-11-21 fixed:** Kaiser β = 16 periodogram taper; lines against a
  running-median baseline with the exact Gamma tail (1e-3 false alarm per
  product) and hysteresis; line skirts excluded and Sx re-estimated three
  times; Sx per segment = mean of the non-line bins (the median/ln 2
  small-sample bias of +0.5 % is gone); `min_duty` on the plain live
  fraction. Synthetic: false lines in 0–0.3 % of products with 1–16
  segments (was 3–77 %); a line of ratio 1.18 is found with 2,000 segments;
  `sqrt_sx²`/true is 1.01 for amplitudes 1e4 and 1e5 (was 1.14 and 2.78).

### [x] A8 🟡 The noise is not white inside a band below about 35 Hz
- **Where:** product convention (one `sqrt_sx` per band); `noise.py`
- **Problem:** the detector PSD is steep at low frequency, so one variance
  per band is wrong at both band ends.
- **Evidence (real H1):**

  | Band | PSD change across the band | Noise-only 2F, low edge → high edge |
  |---|---|---|
  | 24–24.5 Hz | −36 % | 4.56 → 3.50 |
  | 28–28.5 Hz | −23 % | – |
  | 25–26 Hz | −75 % | 5.82 → 3.12 |
  | 30–31 Hz | −29 % | 4.71 → 3.68 |
  | 56–57 Hz | −7 % | – |
  | 100–100.5 Hz | under 2 % | 3.95–4.09 (flat) |

- **Fix:** store a smooth in-band PSD shape with each product (the
  `spectrum` table already has the data), or whiten each band in the
  channelizer and record the shape; keep B ≤ 0.1 Hz below about 35 Hz until
  then.
- **Scripts:** `attack/main/stats.py`, `tilt2f.py`
- **2026-11-21 fixed:** in-band flattening. A mean Welch PSD (128 s, gate-free
  segments) of the conditioned strain is fitted per band (polynomial in
  log S over `fc ± max(0.75 B, 0.25 Hz)`, lines dropped), and the channelizer
  multiplies the band by `g(f) = sqrt(S(fc)/S(f))` when that changes S by
  > 1 % and is significant. Stored as `inband_shape`, `inband_gain_fmin/fmax`.
  Quarter-band noise ratio high/low on real H1: 25–26 Hz 0.55 → 0.96,
  24–24.5 Hz 0.78 → 0.99, 30–31 Hz 0.78 → 0.97, 300–300.5 Hz (60 Hz-harmonic
  skirt) 0.86 → 0.98, 1900–1901 Hz (anti-alias roll-off) 0.96 → 0.99; clean
  bands unchanged. `hwinj_recovery.py` includes g in the expectation.

### [x] A9 🟡 Below 40 Hz the cleaned data are still non-Gaussian and non-stationary
- **Where:** gating design (`conditioning.py`): one broadband statistic over
  20–1700 Hz
- **Problem:** a transient confined to a 30 Hz-wide low band is diluted in
  the broadband statistic (by about √(1680/30) ≈ 7, an estimate), so only
  the per-band veto sees it.
- **Evidence (real H1, after gating and veto):**

  | Bands | Kurtosis | 4σ outliers vs Gaussian | Sx max / median | Mean / harmonic-mean Sx |
  |---|---|---|---|---|
  | 24–31 Hz | 0.24–0.46 | 9–14× | 3.0–8.1 | 1.03–1.19 |
  | 36.65 Hz (calibration line) | 0.65 | 18× | 14 | 1.64 |
  | clean bands above 70 Hz | ≤ 0.12 | 0.4–3.4× | 1.1–1.8 | ≤ 1.03 |

  - The gate threshold is not the cause: 20 → 12 → 8 sigma gives 274 → 306
    → 350 gates and the same statistics in every band.
  - 2F tail at 30 Hz: 69 templates above 20 (44 expected), 3 above 30 (0.43
    expected).
- **Fix:** add a second gate statistic on a low band (for example whitened
  20–60 Hz), and use the per-segment noise as weights in the search (D5).
- **Scripts:** `attack/main/run_thr.py`, `stats.py`, `tails.py`
- **2026-11-21 partly fixed:** the excess is variance episodes of 10 s –
  5 min at 2–4× (single samples are Gaussian to 1.1×; 5-min mean z > 2 in
  0.7 % of windows at 28 Hz against 7e-6). New **excess-power veto** per band
  (windows 10, 30, 100, 300 s; χ² tail at 1e-3 per band and length; shortest
  first). Real H1: 4σ outliers 25–26 Hz 12.9 → 6.9×, 30–31 Hz 8.9 → 4.6×,
  30.0–30.1 Hz 12.5 → 7.8×, for 0.7–1.5 % of their data; 0 % lost in clean
  bands. A second gate statistic on whitened 20–60 Hz was tried and
  rejected: at 6σ it costs 1.9 % of all data (779 gates) and leaves 25–26 Hz
  at 6.7×. **Left:** 28 Hz stays at ~10× (episodes under ~10 s cannot be
  resolved at 5 s per sample in a 0.1 Hz band); per-segment weights in the
  search (D5) are the next step.
- **2026-11-21 follow-up (found on O4a):** a burst filling most of a noise
  segment (a line ringing up at lock loss) inflates that segment's robust
  variance, so one veto pass removed only its core: in 1001.69–1002.19 Hz,
  samples at z ≈ 12,000 survived. Both vetoes now repeat with the variance
  recomputed without the vetoed samples until nothing new is found
  (`noise.veto_band`, `test_veto_band_iterates_through_a_long_burst`).

### [x] A10 🟡 Hardware-injection flags other than CW cannot be excluded
- **Where:** `conditioning.py:345-347` (`RunData`), `config.py` (`dq_flags`)
- **Problem:** `dq_flags` only looks at `quality/simple`. The injection bits
  (`NO_CBC_HW_INJ`, `NO_BURST_HW_INJ`, `NO_DETCHAR_HW_INJ`,
  `NO_STOCH_HW_INJ`) cannot be required; naming one raises "DQ bits not in
  file". Stochastic and detector-characterisation injections add broadband
  noise or sinusoids for minutes.
- **Evidence:** the 56 local files have 0 s of such injections inside
  DATA-good time, so no effect was measured. 180 days will have some.
- **Fix:** accept injection bits in `dq_flags`; require `NO_STOCH_HW_INJ` and
  `NO_DETCHAR_HW_INJ` by default, never `NO_CW_HW_INJ`.
- **Script:** `attack/main/dq_flags.py`
- **2026-11-21 fixed:** `dq_flags` accepts DQ and injection bits; default
  `DATA, NO_STOCH_HW_INJ, NO_DETCHAR_HW_INJ`; `NO_CW_HW_INJ` raises. O4a H1
  (3,901 files): inside DATA, 0 s of stochastic, detchar, burst or CBC
  injections; the O4a files also carry `CW_CAT1` (0 s removed) and
  `STOCH_CAT1` (119 s).

### [x] A11 🟡 `hop = 900` or `2700` with an odd-grid `fmin` gives garbage silently
- **Where:** `channelize.py:302` (`int(fm * self.T)`)
- **Problem:** the block length T is then an odd multiple of 900 s, `fmin·T`
  is a half-integer, and it is truncated. Default `hop = 3600` is safe.
- **Evidence:** tone test, `fmin = 18001/1800` Hz: error 141 % of the signal
  for hop 900 and 2700; 5e-8 for hop 3600.
- **Fix:** raise if `fmin * T` is not an integer.
- **Script:** `attack/main/hop_odd.py`
- **2026-11-21 fixed:** the channelizer and `PipelineConfig` raise when
  `fmin · T` is not an integer. Test `test_rejects_half_cycle_blocks`.

### [x] A12 ⚪ Gating edge cases
- **Long transient across a file boundary** (`conditioning.py:476-487`):
  gated only up to 64 s past the boundary. Noise ×3 for 230 s across it: the
  last 134 s keep mask = True in 72 % of samples with 4× variance.
- **Gate ownership:** two equal glitches on either side of a file boundary
  were missed in 1 of 40 trials and gated twice in 1 of 40.
- **Glitch storms:** when more than half of the 4 s PSD segments in a block
  hold a transient, the median PSD absorbs them and gating stops (3,000
  glitches in 8,192 s: 7 gated).
- **Dead channel:** all-zero strain with DATA = 1 passes as good data
  (`sqrt_sx` 0, or 7.2e-24 against 1e-23 when one file of three is zero).
- **Fix:** let a cluster that touches the block edge extend the block; merge
  overlapping gates instead of assigning owners; re-estimate the PSD after a
  first pass; raise on a zero-variance stretch.
- **Scripts:** `attack/agent1/t03_…`, `t12_…`, `t07_…`;
  `attack/main/gate_ownership.py`
- **2026-11-21 fixed:** per-file detections are merged into one interval list,
  with the 4 s next to a cut block end excluded (a 200 s burst across a file
  boundary is gated whole and once: `test_long_transient_across_file_boundary_is_gated_whole`);
  each file is whitened by the lower of its own and the run's median PSD
  (glitches every 3 s through two files: 680 of 680 gated, 37 without:
  `test_glitch_storm_is_gated`); DQ-good seconds that are constant or
  non-finite raise before processing, or are dropped with
  `bad_strain="drop"` (`test_strain_check`); `python -m readligo check`.

### [x] A13 ⚪ Settings are validated late or not at all
- Rejected only after all conditioning and gating: `hop=1000`,
  `kernel_length=900`, `edge="middle"`, a band too close to Nyquist
  (2046.6–2047.6 Hz), a noise segment that is not a whole number of samples.
- Accepted and wrong: `min_stretch < 2·edge_taper` (reported window 0.33,
  true 0.83), `gate_context=0` (2 false gates on a real file boundary),
  `gate_pad < 0`, `gate_context=64.0` from JSON (`TypeError`).
- CLI: `--config` values of `overwrite` and `keep_work` are always replaced
  by the flags; a bands-file line `60,1` gives a `TypeError`.
- `band_around(101.0, 100.0, 0.1)` returns a band that holds neither
  frequency, and a range ending inside the guard is accepted.
- **Fix:** validate everything in `PipelineConfig.__post_init__`.
- **Scripts:** `attack/agent2/a12_late_validation.py`, `a8_config.py`;
  `attack/agent1/t01b_…`, `t06_cli.py`, `t10b_…`, `t15_…`
- **2026-11-21 fixed:** `__post_init__` checks in `GateParams`,
  `ChannelizerParams`, `NoiseParams`, `VetoParams` and `PipelineConfig`
  (min_stretch ≥ 2 edge_taper, gate_context whole and ≥ 16 s, gate_pad ≥ 0,
  noise segment whole at 2B, fmin·T integer, ...); the channelizer is built
  before any processing (Nyquist, narrow bands). CLI: `--config` values are
  replaced only by flags actually given; bands files accept commas and
  usable columns. `band_around` raises for `f_hi < f_lo` and checks the
  guard.

### [x] A14 ⚪ `mask = True` is "slot untouched", not "sample exact"; two doc claims are too strong
- **Kernel reach:** each sample is a band-limited average, so a gate or gap
  changes its neighbours. Noise-free tone: fully good samples next to a gate
  are off by up to 11 %, against 0.05–0.2 % far away. Negligible for weak
  signals; it adds to A5 for strong lines.
- **"Any split gives bit-identical output"**
  (`pipeline_architecture.md:38-41`, `cleaning_report.md:154`): true for
  reader and channelizer calls (120 random splits, exact). Not true for
  `gps_start`/`gps_end` splits: the cut is a tapered edge, and the halves
  differ by 22 % next to it, 1.4 % at 100–300 s and 1e-4 at 1000–1800 s.
- **"3 files and 1 file give identical output (1e-12)":** only without
  gates. With gates, edges move by up to 0.004 s and data by 2.5e-3 of the
  rms.
- **Fix:** reword the three statements; optionally widen `mask = False` by
  two slots around gates and edges.
- **Scripts:** `attack/agent1/t09_…`, `t04_…`, `t08_…`, `t14_…`
- **2026-11-21 fixed (docs):** the three statements are reworded
  (`pipeline_architecture.md`, `cleaning_report.md`, `cleaning_pipeline.md`
  §4.7–4.8). The longer tapers reduce the effect next to gates and edges
  (mean noise power there 0.85–1.0 of normal); the mask was not widened.

### [x] A15 ⚪ Minor
- The product stops up to half a slot before the data end (`pipeline.py:225`).
- The veto's false rate is 190× the documented 5.7e-7 per sample in segments
  with only 32 live samples (2.75× at full duty, B = 0.1 Hz). The loss is a
  few samples per 180 days.
- Veto runs one sample apart are returned as overlapping ranges, so
  `vetoed_seconds` double counts.
- The channelizer does not check the length the reader returns.
- `gwosc_antialias_gain` returns 0.989–0.998 below 1.5 kHz (filter ripple);
  the docstring says 1.
- A failed `write_product` leaves a `.part` file.
- `{ifo}_pipeline_summary.json` is overwritten by each run into the same
  `out_dir`.
- Non-finite strain in a DQ-good second is found only when its run is
  conditioned, possibly hours into a job. Add a check-only mode.
- **2026-11-21 fixed:** product range = slots overlapping the live span
  (`product_range`); veto ranges merged; `robust_scale` takes a segment with
  < 25 % live samples from its nearest neighbour; the channelizer checks the
  reader's length; docstring corrected; `.part` files are unique and removed
  on failure; one summary file per run; `python -m readligo check`.

### Open after the A fixes (2026-11-21)
- [ ] **A16 ⚪ Conditioning load balance.** Split long runs into per-stretch
  high-pass and per-file gating tasks; would cut O4a conditioning from 12–19
  to ~8 min on 64 workers.
- [ ] **A17 🟡 Low-frequency residual (from A9).** 20–31 Hz bands at the noise
  floor keep 4σ outliers at 6–10× Gaussian on O4a too. Next: per-segment
  weights in the search (D5) and off-source threshold calibration (D4).
- [ ] **A18 ⚪ O4 injection table.** `hwinj_o3a.json` holds O3 amplitudes; the
  O4 values are only in `hwinj_o4a_check.py` (from arXiv:2603.14168 Table I).
  Fetch the official GWOSC O4 table (gwosc.org was unreachable on
  2026-11-21) and add the O4 no-injection timelines.

### Note on the attack scripts (2026-11-21)
The `attack/` folders named above were in the temporary scratchpad and have
since been removed by macOS. The checks that matter are now unit tests
(`tests/`) or kept scripts: `scripts/validation/leakage_check.py` (A5, A9)
and `scripts/validation/inband_flatness.py` (A8).

### Held under attack
- **Clean bands look Gaussian to the search.** 2F over 40–80 thousand
  off-source templates: mean 3.97–4.12 and a χ²₄ tail at 71.9, 100.68,
  120, 135.9, 100–100.5, 300–300.5 and 800–800.5 Hz.
- **Absolute noise level.** Product Sx against a Welch PSD of the raw strain:
  1.007 ± 0.004 at 100–101 Hz.
- **Channelizer.** 675 non-default configurations (L, hop, attenuation,
  widths, fmin, spans before the epoch): gain error ≤ 7e-7, exact rate,
  bit-identical chunking. Every width k/1800 from 0.1 to 1 Hz works.
- **Window and mask.** Brute force over 50 cases: no sample with
  mask = True and W ≠ 1; window error ≤ 0.0074.
- **Gate threshold.** Lowering it from 20 to 8 sigma changes nothing.
- **Veto.** Never fires on stationary or amplitude-modulated tones up to
  100σ. It removes the bursts of the calibration lines at lock loss (mean
  z of 2,100 just before stretch ends in the 36.7 Hz band; 0.96 after the
  veto).
- **Pathological input.** Empty files, 16 s stretches, mixed file lengths,
  fs = 16384, DQ flapping and glitches of 2e11 σ are all handled.
- **Export.** Shape, dtype, row order and alignment match what rustyprune's
  source reads, for one and two detectors (rustyprune was read, not run).

---

## Already correct (keep these when changing things)

- Deterministic and bit-reproducible.
- NaN handling is exact: all 1.7e8 NaN samples fall in DATA = 0 seconds.
- `DATA` is the correct DQ baseline for CW.
- Zero-phase FIR with correct timestamps (≤ 2.3e-11 s).
- Heterodyne sign matches pyloki.
- χ² normalisation is correct.
- All 217 of 217 flagged columns are real glitches. RMS/σ drops from 4.95
  to 1.006.
- Strong and Doppler-shifted CWs are never flagged.
- The DQ-edge mask margin covers filter ringing (32 s against 11.4 s needed
  at 1 Hz).

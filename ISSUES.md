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
| 2. Redesign the chain | C6, C1, C4, C2, C3, C7, C8, O3, O4 | The main problems in the cleaning |
| 3. Search codes | D1–D5 | Needed before any real-data search |
| 4. Validation | V1–V7 | Prove the new chain works |
| 5. Scale and science | G1–G8 | Needed for a real 180-day search |
| Anytime | E2–E10, C9, O5 | Housekeeping |

---

## C. Cleaning pipeline (`readligo/preprocessing.py`)

### [ ] C1 🔴 Content below 10 Hz aliases into bands and creates fake lines
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

### [ ] C2 🟠 Output is critically sampled (fs = B), so band edges are distorted
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

### [ ] C3 🟠 Sample-rate logic is fragile and sometimes aliases
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

### [ ] C4 🟠 Glitch cleaner fails at 0.1 Hz bandwidth
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

### [ ] C6 🟠 Each 4096 s file is processed on its own, leaving start spikes
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

### [ ] C7 🟡 Heterodyne phase restarts at every file
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

### [ ] C8 🟡 Time grid is not uniform across files at B = 0.1
- **Problem:** a 4096 s file at 0.1 Hz gives 410 samples, so a 6 s junction
  appears between consecutive files. Concatenated series are not uniformly
  sampled, but pyloki and rustyprune assume they are.
- **Fix:** solved by C6 (contiguous processing) plus C3 (an exact fs).
- **Done when:** `np.diff(gps_times)` is constant across file boundaries.

### [ ] C9 ⚪ BURST_CAT2 veto not applied
- **Where:** `preprocessing.py:26-53` (`apply_dq_mask` uses
  `DATA & CBC_CAT1`, which equals `DATA` in O3)
- **Problem:** this is optional. Without the BURST_CAT2 veto there is about
  0.7% excess noise at 100 Hz. Using `DATA` is the correct CW baseline, but
  the `CBC_CAT1` name is misleading.
- **Fix:** decide whether to use the veto; rename or comment the mask
  either way.

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

### [ ] O2 🟠 Output carries almost no metadata
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

### [ ] O3 🟠 Output conventions don't match pyloki and rustyprune

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

### [ ] O4 🟠 No noise-level estimate, and a naive estimate is biased
- **Problem:** the searches normalise by one scalar variance, Sx·B.
  - A plain sample variance of the output is 3.4% low because of the edge
    roll-off. That gives about 4× more false alarms at the final threshold.
  - In bands with lines, a data-derived variance is off by 5× to 1500×.
- **Fix:** estimate Sx from the flat centre of the band, excluding lines,
  and store it per segment.

### [ ] O5 ⚪ Amplitude convention is undocumented
- **Problem:** the complex output carries A/2 of the signal amplitude. The
  equivalent real series is 2·Re(y).
- **Fix:** document this in `docs/preprocessing.md` and in the output
  attrs.

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

### [ ] D3 🟠 rustyprune has no real-data input path
- **Where:**
  - `rustyprune/src/allsky/search.rs:357-381`
    (`tstart: DEFAULT_TSTART = 1238319104`)
  - `Root::open` checks only the dtype and shape of `data.npy`
  - `TODO.md` item 4b ("prepare from frames")
- **Fix:** add a real-data `prepare` step with a metadata sidecar (tstart,
  fs, f_het, IFO, mask, variance), and validate the sidecar on open.

### [ ] D4 🟡 Threshold schedules assume simulated, full-duty data
- **Where:** `pyloki/allsky/search.py:65-75` (hard-coded for "0.1 Hz band,
  90 × 2 d")
- **Problem:** the thresholds assume χ²₄ statistics at full duty.
- **Fix:** recalibrate on real duty and noise using off-source nulls (see
  V6).

### [ ] D5 🟡 Single scalar variance for all data
- **Problem:** one `sqrt_sx` comes from the config. Real data needs
  per-detector, per-segment variance.
- **Fix:** accept a variance per segment and per IFO, taken from O4.

---

## V. Validation and notebooks

### [ ] V1 🟠 Stored products were made with `bound=15`, which isn't recorded
- **Problem:** the products on disk were made with `bound=15`, set in a
  notebook cell; the default is 10.
  - All 56 files reproduce bit-for-bit at `bound=15`.
  - At the default, 11 files differ, and a default re-run silently
    overwrites them.
- **Fix:** decide on one value, make it the default, and record it in the
  attrs (see O2).

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

### [ ] V3 🟡 `gating.ipynb` doesn't justify STFT cleaning over gatestrain
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

### [ ] V4 🟡 Only one band and width was ever validated
- **Problem:**
  - Only (100, 101) at 1 Hz has been tested, and band edges never were.
  - The default search band (100.68, 100.78) has never been characterised.
- **Fix:** make (100.68, 100.78) the reference configuration, and add
  band-edge tests.

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

### [ ] V6 🟡 No end-to-end search on real data yet
- **Fix:**
  - Use hardware injections as regression targets: P6 145.445 Hz, P5
    52.808 Hz, P3 108.857 Hz, P8 ~190 Hz, in 0.1 Hz bands.
  - Inject software CWs into the raw 4 kHz strain.
  - Calibrate thresholds against an off-source null.

### [ ] V7 ⚪ Hardware-injection baseline
- **Current state:**
  - P6 is recovered from the current output: 2F ≈ 80, and a 5-harmonic
    statistic of 53.6 against an off-source maximum of 25.9 over 6,822
    trials. Amplitude is within 7%.
  - P5 and P3 are recovered more weakly. P8 is untested.
- **Use:** keep these numbers as the before/after benchmark for every fix.

---

## E. Engineering

### [x] E1 🟠 `import readligo` is broken
- **Where:** `readligo/__init__.py:13` (`from readligo.projection import ...`)
- **Problem:** `projection.py` is not in the repo.
- **Fix:** commit it or remove the import (see V5).
- **Resolved 2026-11-11:** dropped the projection import and its six
  `__all__` entries. Nothing in the repo used them, and pyloki does its own
  antenna projection. `import readligo` now works from a fresh checkout.

### [ ] E2 ⚪ Dependencies are wrong
- **Where:** `pyproject.toml`
- **Problem:** `rich` is used (`process_run.py:4`) but undeclared; `pycbc`
  is declared but unused.

### [ ] E3 ⚪ Docs have drifted from the code
- `process_band` returns 5 values; the docs say 4.
- `plot_cleaning` draws 3 panels; the docs say 4.
- The README quick start uses `time['start']` with `tvec=True`, which fails.
- `docs/preprocessing.md:43` says the 39.4th percentile equals the mean. For
  an exponential it is half the mean. The code is correct; only the
  explanation is wrong.

### [ ] E4 🟡 No tests
- **Fix:** add regression tests for C1, C3, C5, O1 and the hardware
  injections.

### [x] E5 🟡 No NaN / finite guard
- **Problem:** NaNs currently all fall in DATA = 0 seconds, but nothing
  enforces this.
- **Fix:** assert `np.isfinite` on all samples that the mask marks good.
- **Resolved 2026-11-13:** `apply_dq_mask` raises `ValueError` (with the
  count and first offending second) if any DQ-good sample is non-finite.
  Without it, one NaN in a good second made the whole file 100 % "good"
  and passed 96 NaN samples downstream. All 56 raw files pass the check,
  and output on a normal file is bit-identical to before.

### [ ] E6 ⚪ `data_loss_stats` hard-codes 4096 and ignores `bound`
- **Where:** `preprocessing.py:357` (`np.repeat(dq_mask_1hz, 4096)`), and
  the `bound` parameter at `:278`
- **Fix:** pass `fs` through, and use `bound` or drop it.

### [ ] E7 ⚪ `io.py` issues
- **Where:**
  - `io.py:41`: `h5py.File` is opened without a context manager.
  - `io.py:51`: `Xspacing` is read but never used, so 4096 Hz is assumed
    elsewhere.

### [ ] E8 ⚪ `process_run.py` portability
- Hard-coded OneDrive paths at lines 9-10.
- `data_dirname + files.name` at line 60 breaks without a trailing slash;
  use `files` directly, since it is already a `Path`.
- `loss_stats=True` is always passed (line 72).

### [ ] E9 ⚪ Performance
- **Problem:** about 90% of runtime is `np.exp` plus the first decimation
  stage, repeated for every band.
- **Fix:** a one-FFT-per-file channeliser would be about 1000× faster.
  Needed for scale (see G6).

### [ ] E10 ⚪ Files that shouldn't be committed
- `docs/gatestrain.pdf`: copyrighted APS paper (Steltner, Papa & Eggenstein
  2022).
- `.claude/settings.local.json`: local settings.
- **Fix:** remove them from git and add them to `.gitignore`.

---

## G. Gaps for a real search

- [ ] **G1 🟠 No line handling anywhere in the chain.** Data-derived
  variance is off by 5× to 1500× in line bands. The 100.30000 Hz feature is
  real h(t) content. Add line lists and a line veto.
- [ ] **G2 🟡 Non-stationarity not examined.**
- [ ] **G3 🟡 Mask propagation through pyloki's SSB resampling is
  untested.**
- [ ] **G4 🟠 Data coverage is far too small.** Local data are H1 only and
  cover 0.68% of the planned 180 days. Get at least 30–60 contiguous days of
  H1+L1.
- [ ] **G5 🟡 No duty-factor planning.**
- [ ] **G6 🟡 Output layout won't scale.** The current per-file, per-band
  layout gives about 6e7 output files.
- [ ] **G7 🟡 Absolute phase not validated against an external
  reference.**
- [ ] **G8 🟡 None of the proposed fixes are validated yet.** Each fix
  above needs its **Done when** check run.

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

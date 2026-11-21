# How the readLIGO cleaning pipeline works

This is the reference for the cleaning that feeds the fully coherent all-sky
CW searches (rustyprune, pyloki) over 20–2000 Hz. It describes the pipeline
as of 2026-11-21 (v2 plus the fixes of ISSUES.md section A). The history of
the v1 → v2 redesign and its first validation is in `cleaning_report.md`.

Contents:
1. What the pipeline delivers
2. The chain in one picture
3. Planning the bands (Doppler)
4. The steps, one by one
5. Running it (laptop and marvin)
6. Validation
7. What the search must still do, and known limits
8. Settings reference

---

## 1. What the pipeline delivers

One HDF5 **band product** per detector and band `[fmin, fmin + B]`:

```
data[n]    real float64 at exactly fs = 2B, sample n at t_n = epoch + n/(2B)
           data = 2 Re[ z(t_n) exp(-2 pi i fmin (t_n - epoch)) ],  z = positive-frequency band-pass
window[n]  RMS of the strain window W(t) over sample n's slot (0 = no data)
mask[n]    True if W = 1 over the whole slot
sqrt_sx    one-sided noise ASD of the live data (one variance Sx*B per band)
+ noise per 1800 s, line list, gate and veto lists, usable source band, all settings, git commit
```

This is the Makefakedata_v5 convention rustyprune and pyloki are built on:
real, heterodyned at **fmin**, sampled at **2B**, one global **epoch**
(default 1238319104, rustyprune's `tstart`). A strain term
`A cos(2 pi f t + phi)` in the band appears as
`A g(f) cos(2 pi (f - fmin)(t - epoch) + phi + 2 pi f epoch)`, where `g(f)`
is the in-band flattening gain (§4.6, 1 almost everywhere above 40 Hz).
Gaps are exactly zero. `export_rustyprune` turns products into rustyprune's
`data.npy` + `band.json` (§4.11).

## 2. The chain in one picture

```
 plan_bands ─▶ bands file (fmin, B, usable source range)                     planning.py
                    │
 GWOSC 4 kHz files  ▼
   │ inventory: check each file, group back-to-back files into runs          inventory.py
   │ DQ bits: DATA + NO_STOCH_HW_INJ + NO_DETCHAR_HW_INJ (CW injections kept)
   │ strain check: every DQ-good second finite and not constant
   ▼
 per run, as ONE series (parallel over runs) ───────────────────────────── conditioning.py
   │ DQ stretches (>= 16 s)
   │ zero-phase high-pass: <= 10 Hz at -120 dB, flat >= 15 Hz
   │ 8 s half-Hann taper at each stretch end
   │ gating: whitened 20-1700 Hz, per file with 64 s context, seed 20 sigma / extent 6 sigma,
   │         merged over files; zero +-0.25 s, 4 s Hann tapers  ──▶ one GPS gate list
   ▼
 mean PSD of the gate-free conditioned strain ──▶ in-band flattening gain g(f) per band
   ▼
 FFT channelizer (parallel over 2 h blocks; one FFT per block serves every band) ── channelize.py
   │ bands in chunks of <= chunk_gb, streamed to memory-mapped scratch
   ▼
 per band (parallel over bands) ──────────────────────────────────── pipeline.py, noise.py
   │ window, mask (once per bandwidth); data = 0 where window = 0
   │ noise Sx per 1800 s + lines
   │ narrowband outlier veto (single samples, z > 25)
   │ excess-power veto (10 s - 5 min windows of raised in-band power)
   │ (both repeated until nothing new is found)
   │ noise and lines; leakage diagnostics
   ▼
 H1_<fmin>-<fmax>Hz_<gps>-<dur>.h5 ──▶ export_rustyprune ──▶ data.npy, band.json, mask.npy, ...
```

## 3. Planning the bands (Doppler)

An all-sky search band is searched over **source** frequencies, but the data
hold a signal only while its **detector** frequency stays inside the flat
part of the band, `[fmin + g, fmax − g]` (`g` = 2.18 mHz, §4.7). The
detector frequency of a source with frequency `f0` and spin-down `fdot` is

```
f(t) = (f0 + fdot (t - t_ref)) (1 + v.n/c),     |v.n|/c <= D = 1.06e-4
```

so only source frequencies in the band's **usable range**

```
[ (fmin + g)/(1 - D) - d_lo ,  (fmax - g)/(1 + D) - d_hi ]
```

keep their whole track in the band (`d_lo <= 0 <= d_hi`: most negative and
positive `fdot (t − t_ref)`). At 1 kHz, `D f` alone is 0.106 Hz on each
side: a 0.1 Hz band has **no** usable frequency there.

```
          fmin        fmin+g                              fmax-g       fmax
 band      |──────────|──────────────── flat ───────────────|──────────|
 source    .              |◀──────── usable ────────▶|
                     D·f + |d_lo|                    D·f + d_hi
```

`plan_bands` tiles a search range with overlapping bands whose usable ranges
meet exactly; `--bandwidth auto` picks, at each frequency, the narrowest
rustyprune width (k/10 Hz) that keeps at least half of the band usable:

```bash
python -m readligo plan --fmin 20 --fmax 2000 --tspan-days 246.1 \
       --fdot-min -1e-9 --fdot-max 0 -o bands.txt       # fmin band usable_lo usable_hi
```

`--tspan-days` is the span of the whole observing run that is cleaned
(O3a ≈ 183 days, O4a 246.1 days): the cleaning always runs over a whole run.
For O4a, 20–2000 Hz with fdot in [−1e-9, 0] Hz/s gives 9,137 bands (3,563 Hz
of band in total):

| f (Hz) | auto B (Hz) | usable fraction | fixed B = 0.5 Hz | fixed B = 1 Hz |
|---|---|---|---|---|
| 20 | 0.1 | 0.70 | 0.94 | 0.97 |
| 100 | 0.1 | 0.53 | 0.91 | 0.95 |
| 450 | 0.3 | 0.60 | 0.76 | 0.88 |
| 1000 | 0.5 | 0.52 | 0.52 | 0.76 |
| 1500 | 0.7 | 0.51 | 0.31 | 0.66 |
| 2000 | 0.9 | 0.50 | 0.10 | 0.55 |

A fixed B = 0.1 Hz fails above 351 Hz. The usable range goes into the product
(`usable_source_band`) and the export (`readligo_meta.json`). **The search
must restrict its frequency range to it**; rustyprune today searches the
whole band.

## 4. The steps, one by one

### 4.1 Inventory and DQ (`inventory.py`, `io.py`, `conditioning.RunData`)
- Each file is checked against its own metadata (detector, GPS start,
  duration, sample spacing, sample count, DQ length). Overlaps raise.
- Back-to-back files form a **run**, processed as one continuous series, so
  file boundaries are never data edges.
- A second is good when every bit in `dq_flags` passes. Default:
  `DATA`, `NO_STOCH_HW_INJ`, `NO_DETCHAR_HW_INJ` (stochastic and detchar
  hardware injections add broadband noise or sinusoids for minutes).
  `NO_CW_HW_INJ` is refused: CW injections are on almost all the time.
  `BURST_CAT2`, `NO_BURST_HW_INJ`, `NO_CBC_HW_INJ` can be added. O4a files
  also have `CW_CAT1` and `STOCH_CAT1`; inside `DATA` they remove 0 s and
  119 s of the 160.6 days.

### 4.2 Strain check (`conditioning.check_strain`)
Every DQ-good second must be finite and not constant (a dead or zero-filled
channel). By default such seconds raise before any processing, naming GPS
times; `bad_strain="drop"` treats them as DQ-bad and counts them in the
products. `python -m readligo check` runs only this step.

### 4.3 Conditioning (`condition_run`)
- **Stretches:** runs of good seconds; shorter than 16 s are dropped.
- **High-pass:** linear-phase Kaiser FIR applied centred (zero phase):
  −124 dB at the 7.93 Hz line, −120 dB at 10 Hz, flat to 1e-6 above 15 Hz,
  with odd-reflection padding at stretch ends. Without it every window smears
  the 8–12 Hz content (110 dB above the floor) into the bands.
- **Edge taper: 8 s half-Hann** at both ends of each stretch. A 1 s taper
  spread strong lines over about ±1 Hz at every stretch start and end
  (ISSUES.md A5).

### 4.4 Gating (`detect_gates_run`)
- **Detection.** Each file is searched with 64 s of its neighbours: a
  median-Welch PSD (4 s segments), a 4 s whitening FIR over 20–1700 Hz, a
  robust sigma, and a dual threshold: a transient is a cluster of samples
  above 6σ containing one above 20σ.
- **Glitch storms.** Each file is whitened by the lower of its own median
  PSD and the run's median PSD, so a file full of glitches is still whitened
  by quiet-time noise (680 of 680 glitches gated in a 3 s storm, against 37
  without).
- **Merging.** Detections within 4 s of a block end that cuts through live
  data are not trusted (filter edge); the neighbouring block sees that time
  in its interior. All detections are merged into one interval list, so a
  transient on a file boundary is gated once and in full, even if it is
  longer than the context.
- **Gate shape.** Zero from 0.25 s before the first to 0.25 s after the last
  crossing, with **4 s Hann tapers** outside. Short tapers (0.25 s) spread
  violin-mode lines over ±1 Hz at every gate; 4 s confines that to the line's
  own neighbourhood (§6.3). Cost: 0.75 % of the duty factor.
- **CW signals** never seed a gate (whitened amplitude ≤ 0.03σ) and lose
  nothing but the gated time.

### 4.5 Scratch data
The conditioned strain is written to a private scratch folder
(`work_dir/readligo-<ifo>-XXXX`, default `out_dir/_work`), never shared with
another job, deleted at the end, also when the job fails or is stopped with
SIGTERM (only SIGKILL or a crash of the machine leaves it; then delete it by
hand when no job runs). Gates are applied when the channelizer
reads it. `work_dtype="float32"` halves it (rounding ≤ 1e-5 of the in-band
rms, §6.4).

### 4.6 In-band flattening (`noise.mean_psd`, `noise.inband_shape`, `FFTChannelizer.set_shape`)
The searches use one variance per band, so the noise must be white inside the
band. Below ~35 Hz (steep PSD), near broad features (power-line skirts) and
above ~1.6 kHz (GWOSC anti-alias roll-off) it is not.
- A mean Welch PSD (128 s Hann segments) is taken over the gate-free
  conditioned strain.
- For each band, `log S` is fitted with a polynomial in `u = f − fc` over
  `fc ± max(0.75 B, 0.25 Hz)`, lines dropped iteratively.
- If the fit changes S by more than 1 % across the band and is significant,
  the channelizer multiplies the band's response by
  `g(f) = exp(−½ Σ a_k u^k)`, so the mean noise PSD becomes flat at `S(fc)`.
- The product stores `inband_shape` (fc, a1, a2, ...), `inband_gain_fmin`,
  `inband_gain_fmax`. A signal at f appears with amplitude × g(f); for a
  coherent search this is the optimal (noise-weighted) response.

### 4.7 Channelizer (`channelize.py`)
For each band, `x(t_n) = 2 Re[z(t_n) e^{−2πi fmin (t_n − t0)}]` on
`t_n = t0 + n/(2B)`:

```
kernel  h(t) = Kaiser_L(t) x ideal complex band-pass,  L = 3600 s, 120 dB
blocks  [t0 + b hop - L/2, t0 + b hop + hop + L/2),     hop = 3600 s  (T = 7200 s)
X = rfft(block)                       one FFT serves every band
c_k = X[fmin T + k] H(k/T) g(f_k)/N   bins around the band, flattening gain
Y_j = sum over k = j mod 2BT of c_k   exactly how sampling at 2B folds frequencies
x   = 2 Re[ e^{-2 pi i fmin (s_b - t0)} M ifft(Y) ]   keep the central hop
```

- **Exact rate:** fmin and B on the 1/1800 Hz grid, hop and L/2 multiples of
  900 s (and `fmin T` an integer, checked) make every band edge an FFT bin.
- **Edges (`edge="inside"`):** |H| ≤ 1e-6 outside [fmin, fmax], so nothing
  aliases in; flat to 1e-6 on [fmin + g, fmax − g], g = 7.84/L = 2.18 mHz.
- **Phase:** computed with exact fractions against one epoch; no restarts.
- **Chunk independence:** each output sample comes from one epoch-anchored
  block, so splitting the bands, the blocks or the workers changes nothing
  (bit-identical, tested). Splitting the *GPS range* is different: the cut
  becomes a tapered data edge.

### 4.8 Window and mask (`output_window`)
Sample n stands for the slot `[t_n − 1/(4B), t_n + 1/(4B))`. `window` is the
RMS of `W(t) = stretch × edge taper × gates` over the slot; `mask` is True
only if W ≡ 1 over the slot. Samples with window 0 are set to 0. Computed
once per bandwidth, linear in the number of gates (0.2 s for 8,500 gates
over 180 days).

`mask = True` means "the slot is untouched", not "the sample is exact": each
sample is a band-limited average over ~1/B s, so samples next to a gate or an
edge feel it. With 4 s/8 s tapers the mean noise power there is 0.85–1.0 of
normal (§6.3); for a weak CW the effect is negligible.

### 4.9 Noise and lines (`estimate_noise`)
- Per 1800 s segment (aligned to the epoch): Kaiser-tapered periodogram
  (β = 16, sidelobes below about −130 dB), normalised by the segment's
  window; segments with live fraction < 0.5 get none.
- **Lines** on the segment-averaged spectrum (each segment normalised by its
  own Sx) against a running-median baseline: a run of bins is a line if one
  bin exceeds the baseline at false-alarm probability 1e-3 per product (exact
  Gamma tail for the number of segments), extended while above a looser level.
- **Sx per segment** = mean power of the bins outside lines and their
  skirts; lines and Sx are iterated.
- `sqrt_sx` = √(live-time mean of Sx): with one variance (the searches
  today), noise 2F then has mean 4. The harmonic mean and the per-segment
  table are stored for a noise-weighted search.

### 4.10 Vetoes in each band (`narrowband_veto`, `excess_power_veto`)
With `z = x²/s²`, s² a robust variance per segment (median-based, so a
stationary line raises it and is never vetoed):
- **Narrowband outlier veto:** runs of z > 9 peaking above 25 are zeroed
  (±1 sample). P(z > 25) = 5.7e-7 per sample for Gaussian noise.
- **Excess-power veto:** windows of 10, 30, 100 and 300 s whose mean z
  exceeds the χ² level of false-alarm 1e-3 per band and length are zeroed,
  shortest first. This removes episodes of raised in-band noise lasting
  seconds to minutes (scattered light at 20–60 Hz) that no single sample shows.

- **Repeated until nothing new is found** (`veto_band`, usually 1–2 passes,
  at most 6): a burst that fills most of a segment (a line ringing up at lock
  loss) inflates that segment's robust variance, so one pass removes only its
  core; each pass recomputes s² without the samples vetoed so far. Found on
  O4a: in a 0.5 Hz band at 1001.7 Hz, samples at z ≈ 12,000 survived one pass.

A CW signal is far below the noise per sample and per window, so neither veto
touches it.

### 4.11 Products and export (`products.py`)
- One HDF5 file per detector and band, written under a unique temporary name
  and linked into place (a crash leaves nothing; a parallel job's product is
  never silently replaced).
- Diagnostics per band: `near_gate_mean_z`, `near_edge_mean_z`, `far_mean_z`
  (1 for clean data), `n_vetoes`, `n_excess_vetoes`, `variance_ratio`,
  `n_lines`, the GWOSC anti-alias gain at the band edges.
- `export_rustyprune(products, out, tstart, duration, tseg)` writes
  `data.npy` (`<f8`, one row per detector), `band.json` (only keys rustyprune
  accepts), `mask.npy`, `window.npy`, `readligo_meta.json`. It refuses:
  a `tstart` other than 1238319104 (rustyprune ignores it), a band that is not
  k/10 Hz, a span without live data or noise estimate, any existing output
  file. `sqrt_sx` is computed over the exported span from the per-segment noise.

## 5. Running it

```bash
pip install -e .                                   # or PYTHONPATH=<repo>
python -m readligo plan  --fmin 20 --fmax 2000 --tspan-days 246.1 -o bands.txt   # whole run (O4a)
python -m readligo check --data-dir DATA --ifo H1 --workers 32
python -m readligo clean --data-dir DATA --out-dir OUT --ifo H1 --bands-file bands.txt \
       --workers 64 --work-dtype float32 --chunk-gb 16 --skip-existing
python -m readligo export OUT/H1_100.68-100.78Hz_*.h5 --out-dir ROOT_frames \
       --tstart 1238319104 --duration 345600 --tseg 86400
```

- `--workers N`: conditioning (per run), strain check (per file),
  channelizing (per block) and the per-band steps (per band) run in N
  processes. Memory: about 1 GB per worker for channelizing, up to ~2 GB per
  worker for a 1 Hz band over 250 days in the per-band step.
- `--chunk-gb`: channelizer output held at once in scratch; the FFTs are
  repeated per chunk.
- `--skip-existing`: resume an interrupted job (conditioning is redone).
- Scripts that call `run_pipeline` with `workers > 1` on macOS must use
  `if __name__ == "__main__":` (spawned workers re-import the script).
- The same `epoch` for H1 and L1 makes their products align sample for sample.

On marvin (192 cores, 755 GB): code in `/SMALLDATA/saifa/readligo/code`,
Python in `/SMALLDATA/saifa/readligo/env`, runs in
`/SMALLDATA/saifa/readligo/runs`. See §6.4 for the full-scale timing.

## 6. Validation

### 6.1 Unit tests
`pytest` (68 tests): channelizer gain, aliasing, phase, chunking, half-cycle
blocks; high-pass; gating (glitches, file-boundary transients, storms,
short stretches); window; noise level, lines, both vetoes, flattening fit and
gain, iterated vetoes; band planner; product round trip; export checks; end
to end with workers and chunks (bit-identical to serial), scratch isolation,
SIGTERM cleanup, resume, strain check, injection bits, CLI.

### 6.2 Hardware injections, O3a H1, 5.8 days
Known-template JKS F-statistic on the products (`scripts/validation/hwinj_recovery.py`):

| | P04 | P01 | P06 | P14 | P16 | P05 |
|---|---|---|---|---|---|---|
| f (Hz) | 1390.9 | 848.9 | 145.4 | 1991.3 | 234.6 | 52.8 |
| 2F | 78.8 | 74.6 | 78.9 | 69.8 | 48.0 | 43.1 |
| expected | 80.7 | 77.2 | 76.4 | 76.7 | 37.8 | 52.6 |
| amplitude | 1.02 | 1.01 | 1.02 | 0.98 | 1.19 | 0.83 |

All six are louder than every one of ~800–1,100 off-source templates
(background mean 3.8–4.2) and peak exactly at the injection frequency.

### 6.3 Line leakage, flatness and low-frequency tails, O3a H1
(`scripts/validation/leakage_check.py`, `inband_flatness.py`)

| | before (2026-11-15) | now |
|---|---|---|
| mean z next to gates, violin bands 502–520 Hz and 1000 Hz | 3.6–48 | 0.89–1.37 |
| 4σ outliers vs Gaussian, same bands | 4.7–63× (veto on) | 0.9–3.4× |
| quarter-band noise ratio, 25–26 Hz / 24–24.5 / 30–31 | 0.55 / 0.78 / 0.78 | 0.96 / 0.99 / 0.97 |
| 4σ outliers vs Gaussian, 25–26 Hz / 30–31 Hz | 12.9× / 8.9× | 6.9× / 4.6× |
| duty factor | 0.3739 | 0.3711 (gates and tapers) |

### 6.4 Full scale, O4a H1 on marvin
All 3,901 O4a H1 files (246-day span, 160.6 DQ-good days), 379 bands: planned
bands in nine windows (20–22, 24–31, 59–61, 99–101, 299–301, 505–515,
995–1005, 1495–1505, 1990–2000 Hz; B from 0.1 to 1 Hz) plus bands holding
six hardware injections. 64 workers, float32 scratch, one chunk.
Run script and logs: `/SMALLDATA/saifa/readligo/runs/o4a_final.*`; products
in `o4a_final/` (20 GB).

| Step | Result |
|---|---|
| strain check | 3,901 files in 73–92 s; no NaN/inf or constant second |
| conditioning and gating | 324 runs, 735 stretches, 13,875,014 s live, 8,199 gates (2.1 per hour); 12–19 min (bound by the longest run) |
| in-band PSD | 100,112 segments of 128 s; 316 of 379 bands flattened |
| channelizing | 379 bands in one pass, 287 s |
| per-band steps and products | 426 s |
| total | 26.7 min; scratch ~260 GB during the run, then removed |

Hardware injections (`scripts/validation/hwinj_o4a_check.py`; O3 phase
parameters, O4 amplitudes from arXiv:2603.14168 Table I):

| | P05 | P06 | P16 | P01 | P04 | P14 |
|---|---|---|---|---|---|---|
| f (Hz) | 52.8 | 144.6 | 234.6 | 848.9 | 1387.5 | 1991.1 |
| 2F | 4,732 | 5,976 | 4,781 | 4,195 | 4,034 | 3,286 |
| expected | 5,173 | 6,206 | 5,175 | 4,459 | 4,009 | 3,409 |
| amplitude | 0.956 | 0.981 | 0.962 | 0.970 | 1.004 | 0.981 |
| phase (rad) | +0.016 | −0.008 | −0.019 | −0.028 | −0.024 | −0.045 |

Off-source background: mean 3.3–4.3, maximum 7.9–18. P04 spins down by
0.55 Hz over the span and P16 is a binary (±0.041 Hz orbit); both are
recovered at full amplitude in bands planned for their tracks. (With the O3
amplitudes the high-frequency injections looked 0.41–0.49 too weak: the O4
injections are weaker, not the data.)

Bands at the noise floor (sqrt_sx < 1.3 × the window median), per window
(`scripts/validation/product_summary.py`):

| Window (Hz) | bands | 4σ outliers vs Gaussian, median (max) | E[z²]/3 | mean z next to gates | excess-veto loss |
|---|---|---|---|---|---|
| 20–22 | 24 | 7.1 (7.6) | 1.18–1.23 | 1.02–1.05 | 0.1–0.2 % |
| 24–31 | 90 | 6.4 (10.2) | 0.32–1.25 | 0.70–1.10 | 0–3.3 % |
| 59–61 | 26 | 2.1 (2.8) | 1.04–1.07 | 0.98–1.01 | < 0.03 % |
| 99–101 | 38 | 1.7 (1.9) | 0.92–1.04 | 0.92–1.00 | < 0.1 % |
| 299–301 | 12 | 1.3 (2.5) | 1.01–1.05 | 0.97–1.05 | 0–1.2 % |
| 505–515 | 34 | 1.5 (2.5) | 0.22–1.06 | 0.53–1.61 | 0–8 % |
| 995–1005 | 27 | 1.2 (1.5) | 0.93–1.03 | 0.98–1.83 | 0–7 % |
| 1495–1505 | 22 | 1.1 (1.6) | 1.01–1.03 | 0.97–2.16 | 0–3.6 % |
| 1990–2000 | 23 | 1.1 (1.2) | 0.98–1.01 | 0.97–1.00 | < 0.1 % |

Bands that contain a strong line (violin modes, 60 Hz) are non-Gaussian by
nature (the line itself rings up and down; thousands of vetoes); the search
must veto those frequencies. This run used the code before the flattening
safety limit (§4.6), so a few line-dominated bands there carry large in-band
gains (up to 12); the current code leaves those bands unflattened.

## 7. What the search must still do, and known limits

- **Usable range (A2):** search only `usable_source_band` of each product;
  overlapping bands cover the rest.
- **Mask (D1):** rustyprune and pyloki have no mask input; zero-filled gaps
  count as data (noise 2F ≈ 4 × duty with `sqrt_sx_mode="live"`).
- **Real-data input (D3):** rustyprune hard-codes `tstart = 1238319104`;
  O4a data need a real-data `prepare` or that `tstart`.
- **Per-segment noise (D5):** Sx varies 3–8× between segments below 40 Hz;
  `noise_segments` is stored for a noise-weighted search.
- **Residual low-frequency tails:** at 20–31 Hz, 4σ outliers stay at 6–10×
  Gaussian (E[z²]/3 up to 1.25) after both vetoes, on O3a and O4a alike;
  episodes shorter than ~10 s cannot be resolved in a 0.1 Hz band (5 s per
  sample). Expect a heavier 2F tail there and calibrate thresholds on
  off-source data.
- **Violin-mode regions:** next to gates the mean noise power is up to 1.6–2.2
  times normal in a few floor-level bands near 510, 1000 and 1500 Hz (it was
  up to 48 times before the taper change).
- **Near 2 kHz:** GWOSC 4 kHz data are attenuated by the anti-alias filter
  (0.78 at 1.7 kHz, 0.0071 at 1.99 kHz in amplitude); SNR is kept, absolute
  amplitudes need `gwosc_antialias_gain`. Gating sees only up to 1.7 kHz.
- **Lines** are listed and excluded from Sx, not subtracted.
- **Appending data** to an earlier run is not supported; rerun (with
  `--skip-existing` to resume).
- **Conditioning** is parallel per run, so the longest run sets its wall
  time (12–19 min for O4a on 64 workers).
- **Hardware injections in O4** keep the O3 phase parameters but most have
  new amplitudes (arXiv:2603.14168, Table I).

## 8. Settings reference (defaults)

| Setting | Default | Where |
|---|---|---|
| `dq_flags` | DATA, NO_STOCH_HW_INJ, NO_DETCHAR_HW_INJ | `PipelineConfig` |
| `epoch` | 1238319104 | `PipelineConfig` |
| `gate_context` | 64 s | `PipelineConfig` |
| `workers`, `chunk_gb`, `work_dtype` | 1, 8 GB, float64 | `PipelineConfig` |
| `bad_strain`, `skip_existing` | raise, False | `PipelineConfig` |
| high-pass | 10/15 Hz, 120 dB | `GateParams.hp_*` |
| `edge_taper`, `min_stretch` | 8 s, 16 s | `GateParams` |
| whitening | 20–1700 Hz, 4 s PSD segments, 4 s FIR | `GateParams` |
| `threshold`, `low_threshold` | 20σ, 6σ | `GateParams` |
| `gate_pad`, `gate_taper` | 0.25 s, 4 s | `GateParams` |
| `ref_psd` | True | `GateParams` |
| `kernel_length`, `hop`, `atten_db`, `edge` | 3600 s, 3600 s, 120 dB, inside | `ChannelizerParams` |
| `flatten`, `flatten_psd_seg`, `flatten_min` | True, 128 s, 1 % | `ChannelizerParams` |
| noise `segment`, `min_duty`, `taper_beta` | 1800 s, 0.5, 16 | `NoiseParams` |
| `line_false_alarm`, `line_extend_false_alarm`, `baseline_bins` | 1e-3, 0.05, 31 | `NoiseParams` |
| veto `seed`, `extent`, `pad` | 25, 9, 1 | `VetoParams` |
| `excess_scales`, `excess_false_alarm` | 10, 30, 100, 300 s; 1e-3 | `VetoParams` |
| planner `D`, `fdot` range, `min_efficiency` | 1.06e-4, [−1e-9, 0] Hz/s, 0.5 | `plan_bands` |

Every product records all of these in its `config` attribute.

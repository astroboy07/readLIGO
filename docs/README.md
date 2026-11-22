# readligo documentation

| Document | Read when |
|---|---|
| [`cleaning_pipeline.md`](cleaning_pipeline.md) | Always. How the cleaning works today, how to run it (laptop, marvin), how it was validated, what the search must still do |
| [`cleaning_report.md`](cleaning_report.md) | The v1 → v2 redesign: what was wrong in v1, the first v2 validation (2026-11-15) |
| [`pipeline_architecture.md`](pipeline_architecture.md) | Changing how the stages connect, or the product format |
| [`preprocessing.md`](preprocessing.md) | Only for the v1 chain (`process_band`), kept for comparison |
| [`../ISSUES.md`](../ISSUES.md) | Status of every known issue |

## The chain in one picture

```
plan_bands (Doppler-safe overlapping bands, usable source range per band)
GWOSC files ─▶ runs of back-to-back files ─▶ DQ + injection bits ─▶ strain check
            ─▶ 10/15 Hz zero-phase high-pass ─▶ 8 s edge tapers
            ─▶ broadband gating (merged GPS gate list, 4 s gate tapers)
            ─▶ mean PSD ─▶ in-band flattening gain per band
            ─▶ FFT channelizer (all bands, one FFT per 2 h block, chunked, parallel)
            ─▶ per band: real series at 2B, heterodyned at fmin, epoch 1238319104
            ─▶ window/mask, noise per 1800 s, lines, outlier veto, excess-power veto
            ─▶ H1_<fmin>-<fmax>Hz_<gps>-<dur>.h5 ─▶ export_rustyprune (data.npy, band.json)
```

"""Band planner for all-sky searches (ISSUES.md A2)."""
from __future__ import annotations

import numpy as np
import pytest

from readligo.config import FREQ_GRID_DEN, default_guard
from readligo.planning import DOPPLER_MAX, plan_bands, usable_range

T180 = 180 * 86400.0


@pytest.mark.parametrize("bandwidth", ["auto", 1.0])
def test_plan_tiles_range_with_doppler_safe_bands(bandwidth) -> None:
    bands = plan_bands(20.0, 2000.0, tspan=T180, bandwidth=bandwidth)
    us = np.array([b.usable for b in bands])
    assert us[0, 0] == 20.0 and us[-1, 1] == 2000.0
    assert np.all(us[1:, 0] == us[:-1, 1]) and np.all(us[:, 1] > us[:, 0])
    for b in bands:
        assert (b.fmin * FREQ_GRID_DEN).denominator == 1 and (b.band * 10).denominator == 1
        lo, hi = usable_range(float(b.fmin), float(b.band), tspan=T180)
        assert lo <= b.usable[0] + 1e-9 and b.usable[1] <= hi + 1e-9
        # every source frequency in the usable range keeps its detector track inside the flat band
        g = default_guard()
        lo_f, hi_f = b.usable
        assert (lo_f - 1e-9 * T180) * (1 - DOPPLER_MAX) >= float(b.fmin) + g - 1e-9
        assert hi_f * (1 + DOPPLER_MAX) <= float(b.fmax) - g + 1e-9


def test_fixed_narrow_band_fails_where_doppler_is_too_wide() -> None:
    with pytest.raises(ValueError, match="wider band"):
        plan_bands(20.0, 1000.0, tspan=T180, bandwidth=0.1)
    plan_bands(20.0, 300.0, tspan=T180, bandwidth=0.1)  # fine at low frequency


def test_auto_widens_with_frequency() -> None:
    bands = plan_bands(20.0, 2000.0, tspan=T180)
    width = {round(float(b.fmin)): float(b.band) for b in bands}
    assert min(w for f, w in width.items() if f < 100) == 0.1
    assert max(w for f, w in width.items() if f > 1900) >= 0.8

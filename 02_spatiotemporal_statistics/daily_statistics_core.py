"""Daily-first scientific statistics; no scene-count weighting between days."""
from datetime import datetime
from fractions import Fraction
from collections import OrderedDict
import numpy as np

from statistics_core import TAGS, UNITS, selected_months, lake_scalar

AGGREGATION = 'daily_mean_then_period_mean'
METHOD = {
    'order': 'finite-scene arithmetic mean per pixel within each acquisition date; arithmetic mean of finite daily means per pixel within the period; equal-pixel spatial mean of saved maps',
    'day_key': 'YYYYMMDD from the original YYYYMMDDhhmmss acquisition_time; no timezone conversion; duplicate and noncontiguous same-date records all belong to one day',
    'time_weighting': 'Each finite pixel-day has unit weight; a day with multiple finite scenes has the same period weight as a day with one finite scene; no equal-year or equal-month weighting.',
    'validity': 'Only finite values contribute; finite zero is valid; NaN and infinity are invalid; a pixel with no finite scene on a date has no valid daily value.',
    'global_coverage': 'pixel finite-day count over ALL selected available years/seasonal months / number of distinct selected source acquisition dates >= 0.05',
    'period_coverage': 'pixel finite-day count within the current calendar year or pooled calendar month / number of distinct selected source acquisition dates in that group >= 0.05',
    'coverage_combination': 'global condition AND current-period condition AND original lake polygon; positive count required; zero-day periods are invalid',
    'empty_scenes': 'A date with source records contributes one day to denominators even if every scene is empty. A date without any source record is not fabricated.',
    'climatology_month': 'Pool daily means of the same calendar month across available years; each valid day has equal weight, not each year-month.',
    'annual': 'Pool daily means of all selected dates within each calendar year; not an equal-month mean.',
    'climatology_annual': 'Pool daily means of all selected dates across available years; not an equal-year mean.',
    'missing_years': 'Excluded from denominators; annual coordinate contains only available years; actual and missing years remain explicit.',
    'season': 'EPL and YGPL: January-December; IMXL, TPL and NPML: May-October before ANY denominators or sums.',
    'precision': 'float64 within-day sums and daily means and period sums; uint32 valid-day and raw finite-scene counts; float32 saved maps; spatial means computed in float64 from reopened saved float32 maps.',
    'spatial_weighting': 'Each retained grid pixel has equal weight; valid fraction uses the original polygon pixel-center rasterization.',
    'ndays': 'Raw finite pixel-day counts before coverage filtering; these are the temporal mean divisors and coverage numerators.',
    'nobs': 'Raw finite source-scene counts before coverage filtering; diagnostic only, never the divisor of the period mean.',
    'n_days': 'Number of distinct selected source acquisition dates in a period, including all-empty dates; coverage denominator.',
    'n_scenes': 'Number of selected original source acquisition records in a period, including empty scenes; preserved as a scene-count diagnostic.',
}


def day_groups(acquisition_time, year, allowed_months):
    """Return chronological (YYYYMMDD, month, original indices) groups.

    Original index order is stable within a date, even for unsorted source axes.
    No record is discarded or moved to a different date by timezone conversion.
    """
    times = np.asarray(acquisition_time)
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.integer):
        raise ValueError('acquisition_time must be a one-dimensional integer axis')
    by_day = {}
    allowed = set(allowed_months)
    for index, stamp in enumerate(times):
        text = str(int(stamp))
        if len(text) != 14:
            raise ValueError('acquisition_time must have YYYYMMDDhhmmss format')
        date = datetime.strptime(text, '%Y%m%d%H%M%S')
        if date.year != year:
            raise ValueError('Acquisition year differs from source archive')
        if date.month in allowed:
            key = int(text[:8])
            by_day.setdefault(key, (date.month, []))[1].append(index)
    return [(key, month, np.asarray(indices, dtype=np.int64))
            for key, (month, indices) in sorted(by_day.items())]


def _increment(counts, increment):
    increment = np.asarray(increment, dtype=np.uint64)
    if np.any(increment > np.iinfo(np.uint32).max - counts.astype(np.uint64)):
        raise OverflowError('Pixel count exceeds uint32')
    counts += increment.astype(np.uint32)


def iter_daily_means(array, groups, lake_mask, read_scenes=16):
    """Read each date in bounded blocks, preserving all its source records.

    A two-block cache reads the original 16-scene blocks once for chronological
    source axes, avoiding repeated ZIP decompression for adjacent dates. Arbitrary
    source order remains correct with bounded memory. A date spanning blocks is
    never split. Period aggregation receives unrounded float64 daily values.
    """
    if isinstance(read_scenes, bool) or not isinstance(read_scenes, int) or read_scenes < 1:
        raise ValueError('read_scenes must be a positive integer')
    mask = np.asarray(lake_mask, dtype=bool)
    if len(array.shape) != 3 or tuple(array.shape[1:]) != mask.shape:
        raise ValueError('Expected data(time,y,x) matching the lake mask')
    block_cache = OrderedDict()
    for key, month, indices in groups:
        daily_sum = np.zeros(mask.shape, dtype=np.float64)
        daily_nobs = np.zeros(mask.shape, dtype=np.uint32)
        for index in indices:
            start = int(index) // read_scenes * read_scenes
            if start not in block_cache:
                if len(block_cache) >= 2:
                    block_cache.popitem(last=False)
                block_cache[start] = np.asarray(array[start:min(start + read_scenes, array.shape[0])])
            block_cache.move_to_end(start)
            scene = block_cache[start][int(index) - start]
            finite = np.isfinite(scene)
            if np.any(finite & ~mask):
                raise ValueError('Finite source values outside original lake geometry')
            # Stable original-record order makes results independent of read_scenes.
            np.add(daily_sum, scene, out=daily_sum, where=finite)
            _increment(daily_nobs, finite)
        valid = daily_nobs > 0
        mean = np.full(mask.shape, np.nan, dtype=np.float64)
        np.divide(daily_sum, daily_nobs, out=mean, where=valid)
        if not np.isfinite(mean[valid]).all():
            raise ValueError('Within-day mean overflowed float64')
        yield key, month, mean, daily_nobs


def add_day(sums, ndays, nobs, daily_mean, daily_nobs):
    """Add one daily mean with weight one and retain separate scene counts."""
    if any(np.shape(item) != np.shape(sums) for item in (ndays, nobs, daily_mean, daily_nobs)):
        raise ValueError('Daily and accumulator grids differ')
    finite = np.isfinite(daily_mean)
    if not np.array_equal(finite, np.asarray(daily_nobs) > 0):
        raise ValueError('Finite daily mean and raw scene support differ')
    np.add(sums, daily_mean, out=sums, where=finite)
    _increment(ndays, finite)
    _increment(nobs, daily_nobs)


def passes_day_coverage(counts, n_days, fraction=.05):
    ratio = Fraction(str(fraction))
    if ratio <= 0 or ratio > 1:
        raise ValueError('Coverage fraction must be in (0,1]')
    if isinstance(n_days, bool) or int(n_days) != n_days or not 0 <= n_days <= np.iinfo(np.uint32).max:
        raise ValueError('Source day count outside uint32 range')
    counts = np.asarray(counts)
    if n_days == 0:
        return np.zeros(counts.shape, dtype=bool)
    return (counts > 0) & (counts.astype(np.uint64) * ratio.denominator >= int(n_days) * ratio.numerator)


def daily_temporal_map(sums, ndays, n_days, global_ndays, global_n_days, lake_mask, fraction=.05):
    """Filter and save the mean of daily means, using date coverage twice."""
    if any(np.shape(item) != np.shape(sums) for item in (ndays, global_ndays, lake_mask)):
        raise ValueError('All grids must have the same shape')
    eligible = (np.asarray(lake_mask, dtype=bool)
                & passes_day_coverage(ndays, n_days, fraction)
                & passes_day_coverage(global_ndays, global_n_days, fraction))
    values = np.divide(sums, ndays, out=np.zeros_like(sums, dtype=np.float64), where=ndays > 0)
    output = np.full(np.shape(sums), np.nan, dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        output[eligible] = values[eligible].astype(np.float32)
    if not np.isfinite(output[eligible]).all():
        raise ValueError('Daily-first mean overflowed saved float32')
    return output

"""Numerical definitions for observation-pooled lake product statistics."""
from fractions import Fraction
import numpy as np

TAGS = ('Chla', 'SDD', 'SPM')
UNITS = {'Chla': 'ug L-1', 'SDD': 'm', 'SPM': 'mg L-1'}


def selected_months(region):
    if region in ('EPL', 'YGPL'):
        return tuple(range(1, 13))
    if region in ('IMXL', 'TPL', 'NPML'):
        return tuple(range(5, 11))
    raise ValueError(f'Unknown lake region: {region!r}')


def add_observations(sums, counts, data):
    """Accumulate finite observations only; empty scenes still count elsewhere."""
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[1:] != sums.shape or counts.shape != sums.shape:
        raise ValueError('Expected data(time,y,x) and matching sum/count grids')
    finite = np.isfinite(data)
    sums += np.sum(data, axis=0, where=finite, dtype=np.float64, initial=0.)
    increment = finite.sum(axis=0, dtype=np.uint32)
    if np.any(increment > np.iinfo(np.uint32).max - counts):
        raise OverflowError('Observation count exceeds uint32')
    counts += increment


def passes_coverage(counts, n_scenes, fraction=.05):
    """Exact rational threshold: 1/20 passes 5%; 1/21 does not."""
    ratio = Fraction(str(fraction))
    if ratio <= 0 or ratio > 1:
        raise ValueError('Coverage fraction must be in (0,1]')
    if n_scenes < 0 or n_scenes > np.iinfo(np.uint32).max:
        raise ValueError('Scene count outside uint32 range')
    counts = np.asarray(counts)
    if n_scenes == 0:
        return np.zeros(counts.shape, dtype=bool)
    return ((counts > 0) & (counts.astype(np.uint64) * ratio.denominator >= n_scenes * ratio.numerator))


def temporal_map(sums, counts, n_scenes, global_counts, global_n_scenes, lake_mask, fraction=.05):
    """Apply full-period AND current-period coverage to pooled time means."""
    if any(np.shape(a) != np.shape(sums) for a in (counts, global_counts, lake_mask)):
        raise ValueError('All grids must have the same shape')
    eligible = (np.asarray(lake_mask, dtype=bool)
                & passes_coverage(counts, n_scenes, fraction)
                & passes_coverage(global_counts, global_n_scenes, fraction))
    output = np.full(np.shape(sums), np.nan, dtype=np.float32)
    values = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0)
    with np.errstate(over='ignore', invalid='ignore'):
        output[eligible] = values[eligible].astype(np.float32)
    if not np.isfinite(output[eligible]).all():
        raise ValueError('Finite observations overflowed the saved float32 mean')
    return output


def lake_scalar(saved_map, lake_mask):
    """Equal-pixel spatial mean of the SAVED float32 time-mean distribution."""
    values = np.asarray(saved_map)
    mask = np.asarray(lake_mask, dtype=bool)
    if values.dtype != np.dtype('float32') or values.shape != mask.shape:
        raise ValueError('Spatial statistics require the saved float32 distribution')
    finite = np.isfinite(values)
    if np.any(finite & ~mask):
        raise ValueError('Saved finite values fall outside the original lake geometry')
    n_lakepix = int(mask.sum())
    n_valid = int(finite.sum())
    return {'mean': float(values[finite].mean(dtype=np.float64)) if n_valid else None,
            'n_valid': n_valid, 'valid_fraction': n_valid / n_lakepix if n_lakepix else None,
            'n_lakepix': n_lakepix}

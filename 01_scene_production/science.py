"""Numerically faithful OLCI masks and B10+I9 model inputs.

Rrc input order: 443, 560, 620, 665, 754, 865, 1020 nm.
Rrs input order: original OLCI zero-based bands 2..11 (443..754 nm).
Only NumPy is required. The caller applies its lake polygon AFTER compute_mask.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real, Integral

import numpy as np


_PARAMETER_FIELDS = {
    "water": ("threshold",),
    "ndwi": ("threshold",),
    "cloud1": ("r443", "r560", "r665"),
    "cloud2": ("r560", "r865"),
    "cloud3": ("r1020",),
    "shadow": ("r665",),
    "bloom": ("afai", "turbid"),
    "erosion": ("radius",),
    "source": (),
}


def validate_parameters(params):
    """Reject incomplete/ambiguous settings; never substitute viewer defaults.

    Returns the original mapping without modifying it. All groups and their
    numerical values are required even when a group is disabled, matching the
    complete version-4 viewer export used for this run.
    """
    if not isinstance(params, Mapping):
        raise TypeError("params must be the exported 'params' mapping")
    missing = set(_PARAMETER_FIELDS) - set(params)
    unknown = set(params) - set(_PARAMETER_FIELDS)
    if missing or unknown:
        raise ValueError(f"Invalid parameter groups: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for group, fields in _PARAMETER_FIELDS.items():
        spec = params[group]
        if not isinstance(spec, Mapping):
            raise TypeError(f"{group} must be a mapping")
        expected = {"enabled", *fields}
        if set(spec) != expected:
            raise ValueError(f"{group} requires exactly {sorted(expected)}")
        if not isinstance(spec["enabled"], (bool, np.bool_)):
            raise TypeError(f"{group}.enabled must be a boolean")
        for name in fields:
            value = spec[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{group}.{name} must be numeric")
            if not np.isfinite(value):
                raise ValueError(f"{group}.{name} must be finite")
            if group == "erosion":
                if not isinstance(value, Integral) or value < 0:
                    raise ValueError("erosion.radius must be a non-negative integer")
            elif abs(value) > np.finfo(np.float32).max:
                raise ValueError(f"{group}.{name} cannot be represented as float32")
    return params


def _cross_erosion(mask, radius):
    """Repeat a 4-neighbor cross; omit out-of-domain neighbors like the viewer.

    At radius r this is a Manhattan-distance diamond of radius r. A full-True
    image retains its image-edge pixels; outside-array padding is not False.
    """
    a = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        b = a.copy()
        b[:, 1:] &= a[:, :-1]
        b[:, :-1] &= a[:, 1:]
        b[1:, :] &= a[:-1, :]
        b[:-1, :] &= a[1:, :]
        a = b
    return a


def compute_mask(rrc7, params, source_valid=None):
    """Return the exact viewer spectral/eroded mask, shape (height, width).

    Thresholds and every arithmetic intermediate use float32, including AFAI's
    successive multiply-by-89 and divide-by-200 operations. Strict inequalities
    are preserved. If source.enabled is False, source_valid and finite-value
    checks do not silently exclude pixels: the selected spectral rules alone
    determine validity. If source.enabled is True, an explicit source_valid
    array is required. The lake polygon is deliberately not used in erosion.
    """
    p = validate_parameters(params)
    a = np.asarray(rrc7, dtype=np.float32)
    if a.ndim != 3 or a.shape[0] != 7 or min(a.shape[1:]) < 1:
        raise ValueError("rrc7 must have shape (7, height, width) with non-empty spatial axes")
    if source_valid is not None:
        valid = np.asarray(source_valid, dtype=bool)
        if valid.shape != a.shape[1:]:
            raise ValueError("source_valid must have shape (height, width)")
    elif p["source"]["enabled"]:
        raise ValueError("source_valid is required when source.enabled is True")
    else:
        valid = None
    r443, r560, r620, r665, r754, r865, r1020 = a
    mask = np.ones(a.shape[1:], dtype=bool)
    f = np.float32
    if p["source"]["enabled"]:
        mask &= valid
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if p["water"]["enabled"]:
            mask &= (r560 - r1020) / (r560 + r1020) > f(p["water"]["threshold"])
        if p["ndwi"]["enabled"]:
            mask &= (r560 - r865) / (r560 + r865) > f(p["ndwi"]["threshold"])
        if p["cloud1"]["enabled"]:
            mask &= ~((r443 > f(p["cloud1"]["r443"])) &
                      (r560 > f(p["cloud1"]["r560"])) &
                      (r665 > f(p["cloud1"]["r665"])))
        if p["cloud2"]["enabled"]:
            mask &= ~((r560 > f(p["cloud2"]["r560"])) &
                      (r865 > f(p["cloud2"]["r865"])))
        if p["cloud3"]["enabled"]:
            mask &= ~(r1020 > f(p["cloud3"]["r1020"]))
        if p["shadow"]["enabled"]:
            mask &= ~(r665 < f(p["shadow"]["r665"]))
        if p["bloom"]["enabled"]:
            afai = r754 - r665 - (r865 - r665) * f(89) / f(200)
            mask &= ~((afai > f(p["bloom"]["afai"])) &
                      ((r620 - r1020) < f(p["bloom"]["turbid"])))
    if p["erosion"]["enabled"]:
        mask = _cross_erosion(mask, p["erosion"]["radius"])
    return mask


def make_features(rrs10):
    """Build the original B10+I9 matrix, shape (pixel_count, 19).

    Accept (10, N) or (10, height, width), flattened in row-major order.
    Rrs is already rhos/pi: this function performs no extra pi conversion.
    Float32 arithmetic is assigned into float64 output exactly as original
    make_input's np.zeros allocation. NaN and both infinities become zero only
    after feature generation, preserving the original inference preparation.
    """
    a = np.asarray(rrs10, dtype=np.float32)
    if a.ndim not in (2, 3) or a.shape[0] != 10:
        raise ValueError("rrs10 must have shape (10, N) or (10, height, width)")
    a = a.reshape(10, -1)
    x = np.empty((a.shape[1], 19), dtype=np.float64)
    x[:, :10] = a.T
    f = np.float32
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        for col, (numerator, denominator) in enumerate(
                ((0, 3), (1, 3), (5, 3), (8, 3), (9, 3), (8, 5)), start=10):
            x[:, col] = a[numerator] / a[denominator]
        x[:, 16] = (f(1) / a[5] - f(1) / a[8]) * a[9]
        x[:, 17] = a[8] - a[5] + (a[5] - a[9]) * f(44) / f(89)
        x[:, 18] = a[7] - a[5] + (a[5] - a[8]) * f(16) / f(44)
    x[~np.isfinite(x)] = 0
    return x

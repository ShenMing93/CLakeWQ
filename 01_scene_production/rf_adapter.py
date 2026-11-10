"""Prediction-only adapter for explicitly selected frozen V1.0 RF bundles.

Inputs are Rrs in a (10, N) array, with wavelengths given by INPUT_WAVELENGTHS.
The original ModelBundle owns feature construction, imputation, scaling and
inverse target transformation. Multiple selected bundles are averaged only
AFTER that original predict method has returned physical target values.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

INPUT_WAVELENGTHS = (443, 490, 510, 560, 620, 665, 674, 681, 709, 754)
RUNTIME_FILES = ('__init__.py', 'data.py', 'models.py')
DEFAULT_RUNTIME = Path(__file__).resolve().parent / 'vendor'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime(runtime_path):
    """Import the selected frozen runtime; never silently reuse different code."""
    runtime = Path(runtime_path).resolve()
    package = runtime / 'olci_benchmark'
    for filename in RUNTIME_FILES:
        if not (package / filename).is_file():
            raise FileNotFoundError(f'Frozen runtime file missing: {package / filename}')
    for module_name, filename in (
        ('olci_benchmark', '__init__.py'),
        ('olci_benchmark.data', 'data.py'),
        ('olci_benchmark.models', 'models.py'),
    ):
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            existing = getattr(loaded, '__file__', None)
            if existing is None or _sha256(existing) != _sha256(package / filename):
                raise RuntimeError(f'{module_name} was already imported from a different runtime; use a fresh process')
    sys.path.insert(0, str(runtime))
    try:
        module = importlib.import_module('olci_benchmark.models')
    finally:
        sys.path.remove(str(runtime))
    # A previously imported package may have the same __init__ but point its
    # __path__ at different data/models files. Validate what was actually loaded.
    for module_name, filename in (
        ('olci_benchmark', '__init__.py'),
        ('olci_benchmark.data', 'data.py'),
        ('olci_benchmark.models', 'models.py'),
    ):
        actual = getattr(sys.modules.get(module_name), '__file__', None)
        if actual is None or _sha256(actual) != _sha256(package / filename):
            raise RuntimeError(f'{module_name} resolved to a different runtime; use a fresh process')
    return module, package


def _validate_bundle(bundle, label, runtime_module):
    if not isinstance(bundle, runtime_module.ModelBundle):
        raise ValueError(f'{label}: expected a frozen ModelBundle')
    if bundle.name != 'RF' or not isinstance(bundle.estimator, RandomForestRegressor):
        raise ValueError(f'{label}: expected an RF RandomForestRegressor bundle')
    if bundle.mode != 'B':
        raise ValueError(f'{label}: expected mode B (10 bands plus 8 indices)')
    bands = bundle.cfg.get('bands', {})
    if not isinstance(bands, dict) or len(bands) != 10:
        raise ValueError(f'{label}: expected exactly 10 input bands')
    columns = sorted(bands, key=bands.get)
    wavelengths = tuple(bands[column] for column in columns)
    if wavelengths != INPUT_WAVELENGTHS:
        raise ValueError(f'{label}: incompatible input wavelengths {wavelengths}')
    if len(bundle.cfg.get('indices', [])) != 8:
        raise ValueError(f'{label}: expected exactly 8 saved indices')
    # Ask the frozen implementation itself for the feature order, without
    # duplicating the eight equations in this adapter.
    probe, names = runtime_module.features(pd.DataFrame(np.empty((0, 10)), columns=columns), bundle.cfg, bundle.mode)
    if probe.shape[1] != 18 or names != list(bundle.feature_names):
        raise ValueError(f'{label}: saved 18-column feature schema does not match its configuration')
    for component, expected_type in (
        ('estimator', RandomForestRegressor),
        ('imputer', SimpleImputer),
        ('scaler', StandardScaler),
    ):
        fitted = getattr(bundle, component, None)
        if not isinstance(fitted, expected_type) or getattr(fitted, 'n_features_in_', None) != 18:
            raise ValueError(f'{label}: {component} must be fitted on exactly 18 features')
    if bundle.imputer.strategy != 'median':
        raise ValueError(f'{label}: expected the saved median imputer')
    if bundle.transform not in ('raw', 'log1p'):
        raise ValueError(f'{label}: unsupported saved target transform {bundle.transform!r}')
    if not np.isfinite(bundle.target_mean) or not np.isfinite(bundle.target_std) or bundle.target_std <= 0:
        raise ValueError(f'{label}: invalid saved target mean/std')
    return tuple(columns)


class FrozenRFCollection:
    """Load only bundles explicitly listed by the caller, without refitting.

    ``manifest`` is either a JSON path, a mapping ``tag -> [row, ...]``, or
    ``{'models': mapping}``. Each row has ``local`` pointing to bundle.joblib;
    a supplied ``sha256`` is checked before loading. Other metadata is retained
    in ``manifest``. No protocol, split, seed, or model count is selected here.

    ``predict`` accepts exactly (10, N), not (N, 10). NaN is passed unchanged
    to the bundle's fitted median imputer; +/-infinity is rejected explicitly.
    It returns one float32 array of length N for each manifest tag. Outputs are
    physical values returned by the frozen bundles, with an arithmetic mean
    across the explicitly supplied list. No clipping or extra inverse transform
    is applied. The caller remains responsible for selecting masked pixels.
    """

    def __init__(self, manifest, runtime_path=DEFAULT_RUNTIME, model_threads=2):
        if isinstance(model_threads, bool) or not isinstance(model_threads, (int, np.integer)) or model_threads < 1:
            raise ValueError('model_threads must be a positive integer')
        if isinstance(manifest, (str, Path)):
            with Path(manifest).open(encoding='utf-8-sig') as handle:
                manifest = json.load(handle)
        if not isinstance(manifest, Mapping):
            raise TypeError('manifest must be a mapping or JSON path')
        selected = manifest.get('models', manifest)
        if not isinstance(selected, Mapping) or not selected:
            raise ValueError('manifest must contain a nonempty tag-to-model-list mapping')
        runtime_module, package = _load_runtime(runtime_path)
        self.runtime_path = package.parent
        self.runtime_sha256 = {name: _sha256(package / name) for name in RUNTIME_FILES}
        self.manifest = {}
        self.bundles = {}
        self._columns = {}
        for tag, rows in selected.items():
            if not isinstance(tag, str) or not tag or not isinstance(rows, (list, tuple)) or not rows:
                raise ValueError('Each tag must have an explicit nonempty list of model rows')
            self.manifest[tag], self.bundles[tag], self._columns[tag] = [], [], []
            for index, row in enumerate(rows):
                label = f'{tag}[{index}]'
                if not isinstance(row, Mapping) or not isinstance(row.get('local'), (str, Path)):
                    raise ValueError(f'{label}: a local bundle.joblib path is required')
                path = Path(row['local']).resolve()
                if path.name != 'bundle.joblib' or not path.is_file():
                    raise FileNotFoundError(f'{label}: expected an existing bundle.joblib file: {path}')
                digest = _sha256(path)
                expected = row.get('sha256')
                if expected is not None and (not isinstance(expected, str) or digest != expected.lower()):
                    raise ValueError(f'{label}: SHA-256 mismatch for {path}')
                # Only trusted, locally produced artifacts belong in this
                # explicit manifest: joblib is a Python object serialization.
                bundle = runtime_module.load_model(path.parent)
                columns = _validate_bundle(bundle, label, runtime_module)
                bundle.estimator.set_params(n_jobs=int(model_threads))
                self.bundles[tag].append(bundle)
                self._columns[tag].append(columns)
                self.manifest[tag].append({**row, 'local': str(path), 'sha256': digest})

    def predict(self, rrs10):
        """Predict masked pixels supplied as (10,N), preserving bundle behavior."""
        values = np.asarray(rrs10)
        if values.ndim != 2 or values.shape[0] != 10:
            raise ValueError(f'Expected raw Rrs shape (10, N), got {values.shape}')
        if values.dtype.kind not in 'fiu':
            raise TypeError('Rrs must be a real numeric array')
        if np.isinf(values).any():
            raise ValueError('Rrs contains infinity; values are not replaced or silently excluded')
        n_pixels = values.shape[1]
        if n_pixels == 0:
            return {tag: np.empty(0, dtype=np.float32) for tag in self.bundles}
        frames = {}
        outputs = {}
        for tag, bundles in self.bundles.items():
            total = np.zeros(n_pixels, dtype=np.float64)
            for index, (bundle, columns) in enumerate(zip(bundles, self._columns[tag])):
                if columns not in frames:
                    frames[columns] = pd.DataFrame(values.T, columns=list(columns))
                prediction = np.asarray(bundle.predict(frames[columns]), dtype=np.float64)
                if prediction.shape != (n_pixels,) or not np.isfinite(prediction).all():
                    raise ValueError(f'{tag}[{index}]: bundle returned invalid shape or nonfinite predictions')
                total += prediction
            averaged = total / len(bundles)
            with np.errstate(over='ignore', invalid='ignore'):
                converted = averaged.astype(np.float32)
            if not np.isfinite(converted).all():
                raise ValueError(f'{tag}: ensemble output exceeds finite float32 range')
            outputs[tag] = converted
        return outputs

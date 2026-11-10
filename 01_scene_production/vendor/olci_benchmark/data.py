from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def read_table(path):
    path = Path(path)  # UNC/mapped paths use the current Windows identity.
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',')


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_data(df, cfg):
    c = cfg['columns']
    required = set(c.values()) | set(cfg['bands']) | set(cfg['targets'].values())
    missing = required - set(df)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(cfg['bands']) < 3:
        raise ValueError('At least three explicitly mapped spectral bands are required.')
    forbidden = set(c.values()) | set(cfg['targets'].values())
    if forbidden & set(cfg['bands']):
        raise ValueError('Metadata/targets cannot be spectral predictors.')
    if len(set(cfg['bands'].values())) != len(cfg['bands']):
        raise ValueError('Band wavelengths must be unique.')
    df = df.copy().reset_index(drop=True)
    if df[list(c.values())].isna().any().any():
        raise ValueError('Sample ID, lake, date and pixel metadata must not be missing.')
    if df[c['sample_id']].duplicated().any():
        raise ValueError('sample_id must be unique; duplicate observations need distinct IDs.')
    dates = df[c['date']]
    if pd.api.types.is_numeric_dtype(dates):
        dates = pd.to_datetime(dates.astype('int64').astype(str),format='%Y%m%d',errors='raise')
    else:
        dates = pd.to_datetime(dates,errors='raise')
    df['_date'] = dates.dt.normalize()
    df['_year'] = df['_date'].dt.year
    df['_lake'] = df[c['lake']].astype(str)
    df['_group'] = pd.MultiIndex.from_frame(pd.DataFrame({
        'lake': df['_lake'], 'date': df['_date'], 'pixel': df[c['pixel']].astype(str)
    })).factorize()[0]
    for col in list(cfg['bands']) + list(cfg['targets'].values()):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if np.isinf(df[col]).any():
            raise ValueError(f'Infinite value in {col}')
    for target in cfg['targets'].values():
        if (df[target].dropna() < 0).any():
            raise ValueError(f'Negative water-quality target in {target}; curate explicitly.')
    return df


def features(df, cfg, mode):
    bands = sorted(cfg['bands'], key=cfg['bands'].get)
    x = df[bands].to_numpy(dtype=float)
    names = list(bands)
    if mode == 'B':
        parts = [x]
        for spec in cfg.get('indices', []):
            cols = spec['bands']
            if any(col not in bands for col in cols):
                raise ValueError(f'Index references an unmapped band: {spec}')
            z = [df[col].to_numpy(float) for col in cols]
            def div(a, b):
                return np.divide(a, b, out=np.full_like(a, np.nan), where=np.abs(b) > 1e-10)
            kind = spec['kind']
            if kind == 'ratio':
                v = div(z[0], z[1])
            elif kind == 'difference':
                v = z[0] - z[1]
            elif kind == 'normalized_difference':
                v = div(z[0] - z[1], z[0] + z[1])
            elif kind == 'slope':
                v = (z[1] - z[0]) / (cfg['bands'][cols[1]] - cfg['bands'][cols[0]])
            elif kind == 'curvature':
                w = [cfg['bands'][col] for col in cols]
                v = (z[2]-z[1])/(w[2]-w[1]) - (z[1]-z[0])/(w[1]-w[0])
            elif kind == 'inverse_difference_product':
                v = (div(np.ones_like(z[0]),z[0])-div(np.ones_like(z[1]),z[1]))*z[2]
            elif kind == 'baseline_height':
                w = [cfg['bands'][col] for col in cols]
                v = z[1]-z[0]+(z[0]-z[2])*(w[1]-w[0])/(w[2]-w[0])
            else:
                raise ValueError(f'Unknown spectral index: {kind}')
            names.append(spec['name'])
            parts.append(v[:, None])
        x = np.concatenate(parts, axis=1)
    elif mode not in ('A', 'C'):
        raise ValueError(f'Unknown input mode {mode}')
    if len(set(names)) != len(names):
        raise ValueError('Feature names must be unique.')
    return x, names


def make_splits(df, cfg):
    """Global row IDs. ST excludes BOTH test lakes and test years from training."""
    n = len(df)
    idx = np.arange(n)
    v = cfg['validation']
    folds = []
    def add(sid, protocol, train, test):
        folds.append(dict(split_id=sid, protocol=protocol,
                          train=np.asarray(train, int).tolist(), test=np.asarray(test, int).tolist()))
    groups = df['_group'].to_numpy()
    unique = np.unique(groups)
    k = min(v['folds'], len(unique))
    if k >= 2:
        for repeat in range(v['random_repeats']):
            shuffled = np.random.default_rng(cfg['seed'] + repeat).permutation(unique)
            for fold, held in enumerate(np.array_split(shuffled, k)):
                mask = np.isin(groups, held)
                add(f'random_r{repeat+1:02}_fold_{fold+1:02}', 'random', idx[~mask], idx[mask])
    years = sorted(df['_year'].unique())
    for year in years:
        add(f'temporal_{year}', 'temporal', idx[df['_year'] != year], idx[df['_year'] == year])
    for cutoff in years[:-1]:
        add(f'forward_{cutoff}', 'forward', idx[df['_year'] <= cutoff], idx[df['_year'] > cutoff])
    lake_k = min(v['folds'], df['_lake'].nunique())
    if lake_k >= 2:
        for fold, (train, test) in enumerate(GroupKFold(lake_k).split(idx, groups=df['_lake'])):
            add(f'spatial_fold_{fold+1:02}', 'spatial', train, test)
            held_lakes = df.iloc[test]['_lake'].unique()
            for year in years:
                held_lake = df['_lake'].isin(held_lakes)
                held_year = df['_year'] == year
                add(f'spatiotemporal_fold_{fold+1:02}_{year}', 'spatiotemporal',
                    idx[~held_lake & ~held_year], idx[held_lake & held_year])
    for split in folds:
        assert_split(df, split)
    return folds


def assert_split(df, split):
    a, b = df.iloc[split['train']], df.iloc[split['test']]
    if set(split['train']) & set(split['test']):
        raise AssertionError('Row leakage')
    if set(a['_group']) & set(b['_group']):
        raise AssertionError('lake/date/pixel leakage')
    if split['protocol'] in ('spatial', 'spatiotemporal') and set(a['_lake']) & set(b['_lake']):
        raise AssertionError('Lake leakage')
    if split['protocol'] in ('temporal', 'forward', 'spatiotemporal') and set(a['_year']) & set(b['_year']):
        raise AssertionError('Year leakage')
    if split['protocol'] == 'forward' and len(a) and len(b) and a['_year'].max() >= b['_year'].min():
        raise AssertionError('Future-to-past leakage')


def inner_splits(df, protocol, cfg):
    """Recreate the outer independence rule inside outer training only."""
    inner_cfg = {**cfg, 'validation': {**cfg['validation'],
                  'folds': cfg['validation']['inner_folds'], 'random_repeats': 1}}
    splits = [s for s in make_splits(df, inner_cfg) if s['protocol'] == protocol
              and len(s['train']) >= cfg['validation']['min_train']
              and len(s['test']) >= cfg['validation']['min_test']]
    if not splits:
        raise ValueError(f'Insufficient lakes/years/samples for strict {protocol} inner CV')
    return splits


def write_json(path, obj):
    def clean(value):
        if isinstance(value, dict):
            return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value,float) and not np.isfinite(value):
            return None
        return value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, default=str,allow_nan=False), encoding='utf-8')

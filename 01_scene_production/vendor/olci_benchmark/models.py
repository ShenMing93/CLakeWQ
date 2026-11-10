import importlib.util
import importlib.metadata
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from .data import features, write_json

ML = ['PLSR', 'SVR', 'RF', 'ExtraTrees', 'XGBoost', 'LightGBM', 'CatBoost']
DL = ['MLP', 'CNN', 'ResNet', 'FTTransformer', 'TabNet', 'RNN', 'LSTM', 'SpectralTransformer']
ADVANCED = ['TabPFN', 'TabM']
NEURAL = DL + ['TabM']
PACKAGES = {'XGBoost': 'xgboost', 'LightGBM': 'lightgbm', 'CatBoost': 'catboost',
            'TabNet': 'pytorch_tabnet', 'TabPFN': 'tabpfn', 'TabM': 'tabm'}


def family(name):
    return 'ml' if name in ML else 'dl' if name in DL else 'advanced'


def availability(name, cfg):
    package = PACKAGES.get(name, 'torch' if name in NEURAL else 'sklearn')
    if importlib.util.find_spec(package) is None:
        return f'Missing dependency: {package}'
    if name == 'TabPFN':
        path = cfg.get('tabpfn', {}).get('model_path')
        if not path or not Path(path).is_file():
            return 'TabPFN local regression checkpoint is not configured/accessible; no implicit download.'
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as e:
            return f'Regression unsupported: {e}'
    return None


def candidates(name, cfg, n_features, n_train):
    rng = np.random.default_rng(cfg['seed'])
    def choose(xs):
        return xs[int(rng.integers(len(xs)))]
    out = []
    for _ in range(cfg['search']['trials']):
        if name == 'PLSR':
            p = {'n_components': choose(list(range(1, min(n_features, n_train-1, 12)+1)))}
        elif name == 'SVR':
            p = dict(C=10**rng.uniform(-1, 3), gamma=10**rng.uniform(-4, 0), epsilon=10**rng.uniform(-3, 0))
        elif name in ('RF', 'ExtraTrees'):
            p = dict(n_estimators=choose([200, 500]), max_depth=choose([None, 8, 16]),
                     min_samples_split=choose([2, 5, 10]), min_samples_leaf=choose([1, 2, 5]),
                     max_features=choose([0.5, 0.8, 1.0]))
        elif name == 'XGBoost':
            p = dict(n_estimators=choose([300, 700]), learning_rate=choose([0.02, 0.05, 0.1]),
                     max_depth=choose([3, 5, 7]), min_child_weight=choose([1, 5, 10]),
                     subsample=choose([0.7, 1.0]), colsample_bytree=choose([0.7, 1.0]),
                     reg_alpha=choose([0., 0.1, 1.]), reg_lambda=choose([1., 5., 10.]))
        elif name == 'LightGBM':
            depth = choose([3, 5, 7])
            p = dict(num_leaves=min(choose([7, 15, 31]), 2**depth), max_depth=depth,
                     learning_rate=choose([0.02, 0.05, 0.1]), n_estimators=choose([300, 700]),
                     min_child_samples=choose([10, 20, 40]), colsample_bytree=choose([0.7, 1.]),
                     subsample=choose([0.7, 1.]), subsample_freq=1,
                     reg_alpha=choose([0., 1.]), reg_lambda=choose([1., 5., 10.]))
        elif name == 'CatBoost':
            p = dict(depth=choose([4, 6, 8]), learning_rate=choose([0.02, 0.05, 0.1]),
                     iterations=choose([300, 700]), l2_leaf_reg=choose([3., 10.]),
                     random_strength=choose([0.5, 1., 2.]), bagging_temperature=choose([0., 1.]),
                     bootstrap_type='Bayesian')
        elif name == 'TabPFN':
            p = dict(n_estimators=choose([4, 8]))
        else:
            small = n_train < cfg['training']['small_sample_threshold']
            p = dict(hidden=choose([[128, 64, 32]] if small else [[128, 64, 32], [256, 128, 64]]),
                     dropout=choose([0.1, 0.3]), lr=choose([0.0003, 0.001]),
                     weight_decay=choose([0.0001, 0.01]), batch_size=choose([32, 64]),
                     width=choose([16, 32] if small else [32, 64]), kernel=choose([3, 5]),
                     blocks=choose([2] if small else [2, 3, 4]),
                     embedding=choose([32] if small else [32, 64]), heads=4,
                     layers=choose([1, 2]), ffn=choose([64, 128]), k=8,
                     n_d=choose([8, 16]), n_steps=choose([3, 4]),
                     spectral_features=len(cfg['bands']))
        if cfg.get('smoke'):
            for key in ('n_estimators', 'iterations'):
                if key in p:
                    p[key] = 8
        out.append(p)
    return out


class ModelBundle:
    """Fitted preprocessing, target transform, feature schema and estimator together."""
    def __init__(self, name, cfg, mode, transform, params, seed, balanced=False):
        self.name, self.cfg, self.mode, self.transform = name, cfg, mode, transform
        self.params, self.seed, self.balanced = params, seed, balanced

    def fit(self, df, y, valid_df=None, valid_y=None, rounds=None):
        start = time.perf_counter()
        x, self.feature_names = features(df, self.cfg, self.mode)
        if np.isnan(x).all(axis=0).any():
            raise ValueError('A feature is entirely missing in the training fold')
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(self.imputer.fit_transform(x))
        xv = None if valid_df is None else self.scaler.transform(self.imputer.transform(features(valid_df, self.cfg, self.mode)[0]))
        y = np.asarray(y, dtype=float)
        z = np.log1p(y) if self.transform == 'log1p' else y.copy()
        zv = None if valid_y is None else (np.log1p(valid_y) if self.transform == 'log1p' else np.asarray(valid_y))
        # Target standardization stabilizes NN raw-target training; fit only on train.
        self.target_mean = float(z.mean()) if self.name in NEURAL else 0.
        self.target_std = max(float(z.std()), 1e-8) if self.name in NEURAL else 1.
        z = (z - self.target_mean) / self.target_std
        if zv is not None:
            zv = (zv - self.target_mean) / self.target_std
        weights = None
        if self.balanced:
            counts = df['_lake'].value_counts()
            weights = (1 / df['_lake'].map(counts)).to_numpy()
            weights *= len(weights) / weights.sum()
        p = self.params.copy()
        common = dict(random_state=self.seed, n_jobs=self.cfg['training']['threads'])
        patience = self.cfg['training']['patience']
        if self.name == 'PLSR':
            self.estimator = PLSRegression(**p, scale=False).fit(x, z)
        elif self.name == 'SVR':
            self.estimator = SVR(kernel='rbf', **p).fit(x, z)
        elif self.name in ('RF', 'ExtraTrees'):
            cls = RandomForestRegressor if self.name == 'RF' else ExtraTreesRegressor
            self.estimator = cls(**p, **common).fit(x, z, sample_weight=weights)
        elif self.name == 'XGBoost':
            from xgboost import XGBRegressor
            if rounds:
                p['n_estimators'] = rounds
            self.estimator = XGBRegressor(**p, **common, objective='reg:squarederror',
                                          early_stopping_rounds=patience if xv is not None else None)
            self.estimator.fit(x, z, sample_weight=weights, eval_set=[(xv, zv)] if xv is not None else None, verbose=False)
        elif self.name == 'LightGBM':
            from lightgbm import LGBMRegressor, early_stopping
            if rounds:
                p['n_estimators'] = rounds
            self.estimator = LGBMRegressor(**p, **common, verbosity=-1)
            self.estimator.fit(x, z, sample_weight=weights, eval_set=[(xv, zv)] if xv is not None else None,
                               callbacks=[early_stopping(patience, verbose=False)] if xv is not None else [])
        elif self.name == 'CatBoost':
            from catboost import CatBoostRegressor
            if rounds:
                p['iterations'] = rounds
            self.estimator = CatBoostRegressor(**p, random_seed=self.seed, verbose=False,
                                               thread_count=common['n_jobs'], allow_writing_files=False)
            self.estimator.fit(x, z, sample_weight=weights, eval_set=(xv, zv) if xv is not None else None,
                               early_stopping_rounds=patience if xv is not None else None)
        elif self.name == 'TabNet':
            import torch
            from pytorch_tabnet.tab_model import TabNetRegressor
            torch.set_num_threads(self.cfg['training'].get('threads', 4))
            self.estimator = TabNetRegressor(n_d=p['n_d'], n_a=p['n_d'], n_steps=p['n_steps'], seed=self.seed,
                device_name=self.cfg['training'].get('device', 'auto'),
                optimizer_fn=torch.optim.AdamW, optimizer_params={'lr': p['lr'], 'weight_decay': p['weight_decay']},
                scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={'step_size': 10, 'gamma': 0.8}, verbose=0)
            self.estimator.fit(x.astype('float32'), z.reshape(-1, 1).astype('float32'),
                eval_set=[(xv.astype('float32'), zv.reshape(-1, 1).astype('float32'))] if xv is not None else [],
                max_epochs=rounds or self.cfg['training']['epochs'], patience=patience if xv is not None else 0,
                batch_size=min(p['batch_size'], len(x)),
                virtual_batch_size=min(self.cfg['training'].get('tabnet_virtual_batch_size', 16), p['batch_size'], len(x)),
                weights=weights if weights is not None else 0,
                loss_fn=torch.nn.functional.mse_loss if p['loss'] == 'mse' else torch.nn.functional.huber_loss,
                drop_last=len(x) % min(p['batch_size'], len(x)) == 1)
        elif self.name == 'TabPFN':
            from tabpfn import TabPFNRegressor
            limit = self.cfg['tabpfn']['max_train_samples']
            selected = np.arange(len(x))
            if len(x) > limit:
                selected = np.sort(np.random.default_rng(self.seed).choice(len(x), limit, replace=False))
            self.subset_sample_ids = df.iloc[selected][self.cfg['columns']['sample_id']].astype(str).tolist()
            self.estimator = TabPFNRegressor(**p, model_path=self.cfg['tabpfn']['model_path'],
                random_state=self.seed, device=self.cfg['tabpfn']['device'])
            self.estimator.fit(x[selected], z[selected])
        else:
            from .neural import TorchRegressor
            self.estimator = TorchRegressor(self.name, p, self.seed, self.cfg['training']).fit(x, z, xv, zv, weights, rounds)
        est = self.estimator
        if self.name == 'XGBoost':
            self.best_rounds = est.best_iteration + 1 if xv is not None else p['n_estimators']
        elif self.name == 'LightGBM':
            self.best_rounds = est.best_iteration_ or p['n_estimators']
        elif self.name == 'CatBoost':
            self.best_rounds = est.tree_count_
        elif self.name == 'TabNet':
            self.best_rounds = int(est.best_epoch) + 1 if xv is not None else (rounds or self.cfg['training']['epochs'])
        else:
            self.best_rounds = getattr(est, 'best_rounds', None)
        self.training_seconds = time.perf_counter() - start
        self.train_sample_ids = df[self.cfg['columns']['sample_id']].astype(str).tolist()
        return self

    def predict(self, df):
        x, names = features(df, self.cfg, self.mode)
        if names != self.feature_names:
            raise ValueError('Feature order mismatch')
        x = self.scaler.transform(self.imputer.transform(x))
        if self.name == 'TabNet':
            x = x.astype('float32')
        p = np.asarray(self.estimator.predict(x)).reshape(-1) * self.target_std + self.target_mean
        return np.expm1(p) if self.transform == 'log1p' else p

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        if self.name == 'TabPFN':
            from tabpfn.model_loading import save_fitted_tabpfn_model
            save_fitted_tabpfn_model(self.estimator, path / 'estimator.tabpfn_fit')
            estimator = self.estimator
            try:
                self.estimator = None
                joblib.dump(self, path / 'bundle.joblib')
            finally:
                self.estimator = estimator
        else:
            joblib.dump(self, path / 'bundle.joblib', compress=3)
        if self.name in NEURAL and self.name != 'TabNet':
            import torch
            torch.save(self.estimator.net.state_dict(), path / 'weights.pt')
        joblib.dump({'imputer': self.imputer, 'scaler': self.scaler}, path / 'preprocessing.joblib')
        write_json(path / 'feature_list.json', self.feature_names)
        write_json(path / 'target_transform.json', dict(kind=self.transform, mean=self.target_mean, std=self.target_std))
        write_json(path / 'hyperparameters.json', self.params)
        write_json(path / 'training_metadata.json', dict(name=self.name, seed=self.seed, mode=self.mode,
            balanced=self.balanced, train_sample_ids=self.train_sample_ids, training_seconds=self.training_seconds,
            best_rounds=self.best_rounds, parameter_count=getattr(self.estimator, 'parameter_count', None),
            gpu_memory_bytes=getattr(self.estimator, 'gpu_memory_bytes', None),
            tabpfn_subset_sample_ids=getattr(self, 'subset_sample_ids', None)))
        return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def load_model(path):
    path = Path(path)
    bundle = joblib.load(path / 'bundle.joblib')  # Load trusted locally generated artifacts only.
    if bundle.name == 'TabPFN':
        from tabpfn.model_loading import load_fitted_tabpfn_model
        bundle.estimator = load_fitted_tabpfn_model(path / 'estimator.tabpfn_fit', device='cpu')
    return bundle

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .predict import _MODEL_REGISTRY
from .tree_models import BaseTreeModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _catboost_space(trial, pos_ratio):
    return {
        'iterations': trial.suggest_int('iterations', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'depth': trial.suggest_int('depth', 2, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 100.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.2, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', pos_ratio * 0.7, pos_ratio * 1.5),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_seed': 42,
        'verbose': 0,
        'eval_metric': 'F1',
    }


def _lightgbm_space(trial, pos_ratio):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', pos_ratio * 0.5, pos_ratio * 2.0),
        'random_state': 42,
        'verbose': -1,
    }


def _xgboost_space(trial, pos_ratio):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', pos_ratio * 0.7, pos_ratio * 1.5),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': 42,
        'verbosity': 0,
        'eval_metric': 'logloss',
    }


_SEARCH_SPACES = {
    'catboost': _catboost_space,
    'lightgbm': _lightgbm_space,
    'xgboost': _xgboost_space,
}

# suggest_*로 탐색되지 않는 고정 파라미터 (study.best_params에 저장 안 됨)
_FIXED_PARAMS = {
    'catboost': {'random_seed': 42, 'verbose': 0, 'eval_metric': 'F1'},
    'lightgbm': {'random_state': 42, 'verbose': -1},
    'xgboost':  {'random_state': 42, 'verbosity': 0, 'eval_metric': 'logloss'},
}


class Tuner:
    """
    모델별 Optuna 하이퍼파라미터 튜닝.

    Usage:
        # pipeline 전달 시 fold별 인코딩+스케일링 자동 수행 (leakage 방지)
        tuner = Tuner('catboost', X, y, pipeline=pipe, n_trials=500)
        study = tuner.run()
        # study.best_params → predict_params()에 전달
    """

    def __init__(self, model_cls, X, y, n_trials=100,
                 n_splits=5, n_repeats=3, pipeline=None, pos_cap=None):
        if isinstance(model_cls, str):
            key = model_cls.lower().replace(' ', '')
            if key not in _MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_cls}. 지원: {list(_MODEL_REGISTRY.keys())}")
            self.model_key = key
            self.wrapper_cls = _MODEL_REGISTRY[key]
        elif isinstance(model_cls, type) and issubclass(model_cls, BaseTreeModel):
            self.wrapper_cls = model_cls
            self.model_key = next(
                (k for k, v in _MODEL_REGISTRY.items() if v == model_cls), None
            )
            if self.model_key is None:
                raise ValueError(f"model_cls {model_cls}에 대한 search space가 없습니다")
        else:
            raise TypeError(f"model_cls는 str 또는 BaseTreeModel 서브클래스여야 합니다")

        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.pipeline = pipeline
        self.pos_ratio = (y == 0).sum() / (y == 1).sum()
        self.space_fn = _SEARCH_SPACES[self.model_key]
        self.pos_cap = pos_cap
        self.study = None

    def _objective(self, trial):
        params = self.space_fn(trial, self.pos_ratio)
        wrapper = self.wrapper_cls(params=params)

        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=42
        )
        oof_probs = np.zeros(len(self.X))

        for train_idx, val_idx in rskf.split(self.X, self.y):
            X_tr = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_tr = self.y.iloc[train_idx]
            y_val = self.y.iloc[val_idx]

            # fold별 인코딩+스케일링 (leakage 방지)
            if self.pipeline is not None:
                X_tr, X_val, _ = self.pipeline.encode_fold(X_tr, y_tr, X_val)

            model = wrapper._create_model()
            wrapper._fit_model(model, X_tr, y_tr, X_val, y_val)
            oof_probs[val_idx] += model.predict_proba(X_val)[:, 1] / self.n_repeats

        best_f1 = 0.0
        for t in np.arange(0.10, 0.70, 0.01):
            pred_temp = (oof_probs >= t).astype(int)
            if self.pos_cap is not None and pred_temp.mean() > self.pos_cap:
                continue
            f1 = f1_score(self.y, pred_temp)
            if f1 > best_f1:
                best_f1 = f1
        return best_f1

    @property
    def best_params(self):
        """study.best_params + 고정 파라미터를 병합해서 반환"""
        if self.study is None:
            raise RuntimeError("run()을 먼저 실행하세요")
        fixed = _FIXED_PARAMS.get(self.model_key, {})
        return {**self.study.best_params, **fixed}

    def run(self, show_progress=True):
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress,
        )
        print(f"Best F1: {self.study.best_value:.4f}")
        print(f"Best params: {self.best_params}")
        return self.study

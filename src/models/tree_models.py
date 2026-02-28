import numpy as np
import pandas as pd
import warnings
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')


class BaseTreeModel(ABC):
    """트리 모델 베이스 클래스"""

    def __init__(self, params: dict = None, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.params = params or self.default_params()
        self.models = []
        self.oof_preds = None
        self.best_threshold = 0.5
        self.fold_scores = []

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def default_params(self) -> dict:
        pass

    @abstractmethod
    def _create_model(self) -> object:
        pass

    @abstractmethod
    def _fit_model(self, model, X_train, y_train, X_val, y_val):
        pass

    def _predict_proba(self, model, X) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def _find_best_threshold(self, y_true, y_prob, low=0.10, high=0.70, step=0.01):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(low, high, step):
            f1 = f1_score(y_true, (y_prob >= t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t, best_f1

    def fit(self, X: pd.DataFrame, y: pd.Series, pipeline=None) -> dict:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_preds = np.zeros(len(X))
        self.models = []
        self.fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tqdm(
            skf.split(X, y), total=self.n_splits, desc=f'{self.name}'
        )):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if pipeline is not None:
                X_train, X_val, _ = pipeline.encode_fold(X_train, y_train, X_val)

            model = self._create_model()
            self._fit_model(model, X_train, y_train, X_val, y_val)
            self.models.append(model)

            val_prob = self._predict_proba(model, X_val)
            self.oof_preds[val_idx] = val_prob

            _, fold_f1 = self._find_best_threshold(y_val, val_prob)
            self.fold_scores.append(fold_f1)

        self.best_threshold, oof_f1 = self._find_best_threshold(y, self.oof_preds)
        mean_fold_f1 = np.mean(self.fold_scores)

        return {
            'name': self.name,
            'oof_f1': round(oof_f1, 4),
            'mean_fold_f1': round(mean_fold_f1, 4),
            'std_fold_f1': round(np.std(self.fold_scores), 4),
            'best_threshold': round(self.best_threshold, 2),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        for model in self.models:
            preds += self._predict_proba(model, X)
        return preds / len(self.models)


# ─── LightGBM ───────────────────────────────────────────────
class LGBModel(BaseTreeModel):

    @property
    def name(self):
        return 'LightGBM'

    def default_params(self):
        return {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'is_unbalance': True,
            'random_state': self.random_state,
            'verbose': -1,
        }

    def _create_model(self):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**self.params)

    def _fit_model(self, model, X_train, y_train, X_val, y_val):
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )


# ─── XGBoost ────────────────────────────────────────────────
class XGBModel(BaseTreeModel):

    @property
    def name(self):
        return 'XGBoost'

    def default_params(self):
        return {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 2.35,  # 70/30 비율
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'verbosity': 0,
        }

    def _create_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(**self.params)

    def _fit_model(self, model, X_train, y_train, X_val, y_val):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )


# ─── CatBoost ───────────────────────────────────────────────
class CatModel(BaseTreeModel):

    @property
    def name(self):
        return 'CatBoost'

    def default_params(self):
        return {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'subsample': 0.8,
            'auto_class_weights': 'Balanced',
            'random_seed': self.random_state,
            'verbose': 0,
        }

    def _create_model(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**self.params)

    def _fit_model(self, model, X_train, y_train, X_val, y_val):
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )


# ─── RandomForest ────────────────────────────────────────────
class RFModel(BaseTreeModel):

    @property
    def name(self):
        return 'RandomForest'

    def default_params(self):
        return {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1,
        }

    def _create_model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**self.params)

    def _fit_model(self, model, X_train, y_train, X_val, y_val):
        model.fit(X_train, y_train)


# ─── ExtraTrees ──────────────────────────────────────────────
class ETModel(BaseTreeModel):

    @property
    def name(self):
        return 'ExtraTrees'

    def default_params(self):
        return {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1,
        }

    def _create_model(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**self.params)

    def _fit_model(self, model, X_train, y_train, X_val, y_val):
        model.fit(X_train, y_train)


# ─── Comparator ──────────────────────────────────────────────
class ModelComparator:
    """여러 모델을 한번에 비교"""

    def __init__(self, models: list[BaseTreeModel] = None,
                 n_splits: int = 10, random_state: int = 42):
        if models is None:
            models = [
                LGBModel(n_splits=n_splits, random_state=random_state),
                XGBModel(n_splits=n_splits, random_state=random_state),
                CatModel(n_splits=n_splits, random_state=random_state),
                RFModel(n_splits=n_splits, random_state=random_state),
                ETModel(n_splits=n_splits, random_state=random_state),
            ]
        self.models = models
        self.results = []

    def run(self, X: pd.DataFrame, y: pd.Series, pipeline=None) -> pd.DataFrame:
        self.results = []
        for model in self.models:
            result = model.fit(X, y, pipeline=pipeline)
            self.results.append(result)
            print(f"  → {result['name']}: OOF F1={result['oof_f1']}, "
                  f"Mean Fold F1={result['mean_fold_f1']}±{result['std_fold_f1']}, "
                  f"Threshold={result['best_threshold']}")
        return pd.DataFrame(self.results).sort_values('oof_f1', ascending=False).reset_index(drop=True)

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {m.name: m.predict(X) for m in self.models}

    def predict_proba(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {m.name: m.predict_proba(X) for m in self.models}

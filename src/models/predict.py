import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .tree_models import BaseTreeModel, CatModel, LGBModel, XGBModel

# model_cls 문자열 → BaseTreeModel 서브클래스 매핑
_MODEL_REGISTRY = {
    'catboost': CatModel,
    'lightgbm': LGBModel,
    'xgboost': XGBModel,
}


def predict_params(model_cls, params, X_train, y_train, X_test, test_id,
                        save_dir='submissions', version='v1',
                        n_splits=5, n_repeats=3, fit_fn=None, pipeline=None,
                        pos_cap=None):
    """
    Optuna study 결과 등으로 받은 params로 RepeatedStratifiedKFold CV 예측 + 제출 파일 저장.

    Parameters
    ----------
    model_cls : str | BaseTreeModel 서브클래스
        'catboost', 'lightgbm', 'xgboost' 문자열 또는 BaseTreeModel 서브클래스.
    params : dict
        모델 하이퍼파라미터 (study.best_params 등).
    fit_fn : callable, optional
        커스텀 fit 함수. (model, X_tr, y_tr, X_val, y_val) -> None.
        None이면 BaseTreeModel._fit_model 자동 사용.
    pipeline : Pipeline, optional
        Pipeline 인스턴스. 전달 시 각 fold에서 encode_fold()를 호출하여
        인코딩+스케일링 수행 (leakage 방지).
        None이면 데이터가 이미 인코딩/스케일링된 것으로 간주.

    Returns
    -------
    oof_probs, test_probs, best_threshold
    """
    # model_cls 해석
    if isinstance(model_cls, str):
        key = model_cls.lower().replace(' ', '')
        if key not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_cls}. 지원: {list(_MODEL_REGISTRY.keys())}")
        wrapper_cls = _MODEL_REGISTRY[key]
    elif isinstance(model_cls, type) and issubclass(model_cls, BaseTreeModel):
        wrapper_cls = model_cls
    else:
        raise TypeError(f"model_cls는 str 또는 BaseTreeModel 서브클래스여야 합니다: {type(model_cls)}")

    wrapper = wrapper_cls(params=params)

    n_total_folds = n_splits * n_repeats
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    oof_probs = np.zeros(len(X_train))
    test_probs = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # fold별 인코딩+스케일링 (leakage 방지)
        if pipeline is not None:
            X_tr_enc, X_val_enc, X_test_enc = pipeline.encode_fold(
                X_tr, y_tr, X_val, X_test
            )
        else:
            X_tr_enc, X_val_enc, X_test_enc = X_tr, X_val, X_test

        model = wrapper._create_model()

        if fit_fn is not None:
            fit_fn(model, X_tr_enc, y_tr, X_val_enc, y_val)
        else:
            wrapper._fit_model(model, X_tr_enc, y_tr, X_val_enc, y_val)

        oof_probs[val_idx] += model.predict_proba(X_val_enc)[:, 1] / n_repeats
        test_probs += model.predict_proba(X_test_enc)[:, 1] / n_total_folds

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.70, 0.01):
        pred_temp = (oof_probs >= t).astype(int)
        if pos_cap is not None and pred_temp.mean() > pos_cap:
            continue
        f1 = f1_score(y_train, pred_temp)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"OOF F1: {best_f1:.4f}, Threshold: {best_t:.2f}")

    pred = (test_probs >= best_t).astype(int)
    # test 예측에도 pos_cap 적용 (threshold만으로 통제 안 될 때 top-k로 보정)
    if pos_cap is not None and pred.mean() > pos_cap:
        n_test = len(test_probs)
        k = int(n_test * pos_cap)
        order = np.argsort(-test_probs)
        pred = np.zeros(n_test, dtype=int)
        pred[order[:k]] = 1
        print(f"[pos_cap] test 양성 비율 초과 → top-{k} 보정")
    sub = pd.DataFrame({'ID': test_id, 'completed': pred})

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{version}.csv')
    sub.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    print(f"Class 분포:\n{sub['completed'].value_counts()}")

    return oof_probs, test_probs, best_t

#!/usr/bin/env python
"""BDA 수료 예측 파이프라인

CLI:
    python main.py preprocess                               # 전처리만
    python main.py compare                                  # 전처리 + 모델 비교
    python main.py tune   --model catboost --trials 200     # 전처리 + 튜닝
    python main.py predict --model catboost --version v1    # 전처리 + 튜닝 + 예측
    python main.py run    --model catboost --trials 200 --version v1  # 전체

Jupyter:
    from main import Runner
    r = Runner()
    r.preprocess()
    r.compare()
    r.tune('catboost', n_trials=200)
    r.predict('catboost', version='v1')
    # 또는 한번에
    r.run()
"""

import argparse
import pandas as pd

TRAIN_PATH = 'dataset/train.csv'
TEST_PATH  = 'dataset/test.csv'
MODELS = ['catboost', 'lightgbm', 'xgboost']


class Runner:
    def __init__(self, pipe_config: dict = None):
        from src.preprocess.pipeline import Pipeline
        self.pipe    = Pipeline(config=pipe_config)
        self.X       = None
        self.y       = None
        self.test    = None
        self.test_id = None
        self._tuner  = None

    # ── 단계별 메서드 ─────────────────────────────────────────

    def preprocess(self) -> tuple:
        """데이터 로드 + 전처리. (X, y, test, test_id) 반환."""
        train = pd.read_csv(TRAIN_PATH)
        test  = pd.read_csv(TEST_PATH)
        train_p, self.test, self.test_id = self.pipe.run(train, test)
        self.X = train_p.drop(columns=['completed'])
        self.y = train_p['completed']
        return self.X, self.y, self.test, self.test_id

    def compare(self, models: list = None) -> pd.DataFrame:
        """여러 모델 OOF F1 비교. 결과 DataFrame 반환."""
        from src.models import ModelComparator
        self._ensure_preprocessed()
        results = ModelComparator(models=models).run(self.X, self.y, pipeline=self.pipe)
        print(results.to_string(index=False))
        return results

    def tune(self, model: str = 'catboost', n_trials: int = 100, **kw) -> object:
        """Optuna 하이퍼파라미터 탐색. optuna.Study 반환."""
        from src.models import Tuner
        self._ensure_preprocessed()
        self._tuner = Tuner(model, self.X, self.y, pipeline=self.pipe, n_trials=n_trials, **kw)
        return self._tuner.run()

    def predict(self, model: str = 'catboost', params: dict = None,
                version: str = 'v1', **kw) -> tuple:
        """최종 예측 + 제출 파일 저장. (oof_probs, test_probs, threshold) 반환."""
        from src.models import predict_params
        self._ensure_preprocessed()
        params = params or self._tuner_params()
        return predict_params(
            model, params, self.X, self.y, self.test, self.test_id,
            version=version, pipeline=self.pipe, **kw,
        )

    def run(self, model: str = 'catboost', n_trials: int = 100,
            version: str = 'v1') -> tuple:
        """전체 파이프라인 (전처리 → 모델 비교 → 튜닝 → 예측)."""
        self.preprocess()
        self.compare()
        self.tune(model=model, n_trials=n_trials)
        return self.predict(model=model, version=version)

    # ── 내부 헬퍼 ─────────────────────────────────────────────

    def _ensure_preprocessed(self):
        if self.X is None:
            self.preprocess()

    def _tuner_params(self) -> dict:
        if self._tuner is None:
            raise RuntimeError("params 없음 — tune() 먼저 실행하거나 params= 직접 전달하세요.")
        return self._tuner.best_params


# ── CLI ───────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='BDA 수료 예측 파이프라인',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('preprocess', help='데이터 로드 + 전처리')
    sub.add_parser('compare',    help='전처리 + 모델 비교')

    for cmd, help_text in [
        ('tune',    '전처리 + 하이퍼파라미터 튜닝'),
        ('predict', '전처리 + 튜닝 + 예측 파일 저장'),
        ('run',     '전체 파이프라인 (비교 포함)'),
    ]:
        sp = sub.add_parser(cmd, help=help_text)
        sp.add_argument('--model',   default='catboost', choices=MODELS)
        sp.add_argument('--trials',  type=int, default=100, metavar='N')
        if cmd in ('predict', 'run'):
            sp.add_argument('--version', default='v1')

    return p


def main():
    args   = _build_parser().parse_args()
    runner = Runner()

    if args.cmd == 'preprocess':
        runner.preprocess()

    elif args.cmd == 'compare':
        runner.compare()

    elif args.cmd == 'tune':
        runner.tune(model=args.model, n_trials=args.trials)

    elif args.cmd == 'predict':
        runner.tune(model=args.model, n_trials=args.trials)
        runner.predict(model=args.model, version=args.version)

    elif args.cmd == 'run':
        runner.run(model=args.model, n_trials=args.trials, version=args.version)


if __name__ == '__main__':
    main()

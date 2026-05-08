<div align="center">

<img src="dataset/banner.png" width="100%">

<br><br>

<h1>🎓 BDA 제 2회 학습자 수료 예측 AI 경진대회</h1>

<p>학습자 설문 데이터 기반 수료 여부 이진 분류</p>

<br>

### 🏅 Tech Stack 🏅

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3CB371?style=for-the-badge&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-6C63FF?style=for-the-badge&logoColor=white)

</div>

<br>

**📅 진행기간 : 2026.01 ~ 2026.02**

**👥 인원 : 개인**

**🏆 최종 성적 : Private 상위 4% (733명 중 29위) · F1 Score 0.40921**

<br>

---------------------------------------------------------------------------------

# 프로젝트 개요

- BDA에서 9기 학습자 설문 응답 데이터만으로 10기 학습자의 수료 여부를 예측하는 경진대회
- 748건의 소규모 데이터 + 3:7 클래스 불균형 + 자유서술형 비정형 컬럼이 혼재하는 환경
- Train/Test 간 `major1_1` 컬럼 형식 불일치(계열 대분류 vs 구체적 학과명) 로 직접 규칙 기반 매핑 처리 필요
- 도메인 기반 피처 설계와 Leakage-Free CV를 조합한 접근법 적용

<br>

| 항목 | 내용 |
|------|------|
| 주최 | 빅데이터분석학회 / 데이콘 |
| 평가 지표 | Binary F1 Score |
| 학습 데이터 | 748행 × 46컬럼 (BDA 9기 학습자) |
| 테스트 데이터 | 814행 × 45컬럼 (BDA 10기 학습자) |
| 참가자 수 | 733명 |

<br><br>

# 프로세스

```
EDA  →  전처리  →  모델 V1 (LightGBM, 100 trials)
                        ↓
               피처 선택 V2 (Zero-Importance 제거)
                        ↓
               SMOTE 실험 V3 (클래스 불균형 처리 비교)
                        ↓
               파생변수 V4 (수료 시그널 피처 설계)  ←  최종 채택 피처셋
                        ↓
               파생변수 V5 + 텍스트 임베딩 실험 (비채택 — 소규모 데이터 과적합)
                        ↓
               5-Model 비교 → Leakage-Free CV 설계
                        ↓
               CatBoost Optuna 튜닝 (500 trials)  →  최종 예측 (catboost_tuning_v3)
```

<br><br>

# 데이터

- **[[DACON] BDA 제 2회 학습자 수료 예측 AI 경진대회](링크 삽입)** 에서 학습자 설문 데이터 수집
- **수집 결과: Train 748행 × 46컬럼 / Test 814행 × 45컬럼**
- 타겟 변수 `completed` : 수료(1) 30% · 미수료(0) 70% 클래스 불균형 존재

<br><br>

# 데이터 전처리

### 결측치 분석

EDA 단계에서 결측치의 패턴을 파악하여 **"결측 자체가 정보를 담는 경우"** 와 **"단순 미응답인 경우"** 를 구분하는 것이 핵심이었음

| 컬럼 | 결측 비율 | 결측의 의미 | 처리 방법 |
|------|----------|------------|----------|
| `previous_class_3~8` | 약 80% | 신규 가입자는 이전 기수 수강 이력 없음 | `'해당없음'` 대체 |
| `class2~4` | `class2` 약 77% | 재수강하지 않은 신규 가입자는 분반 없음 | `0` 대체 |
| `contest_participation` | 일부 | 대회 경험 없음 | `'없음'` 대체 |
| `major1_2` | 일부 | 복수전공 없음 | `'없음'` 대체 |
| `major type` | 일부 | 복수전공자만 응답 → 결측 = 단일전공 | `'단일 전공'` 대체 |
| `major1_1`, `major_field` | 소수 | 단순 미응답 | `'미응답'` 대체 |
| `nationality` | 소수 | 단순 미응답 | train 최빈값 대체 |
| `completed_semester` | 3.74% | 단순 미응답 | job 그룹별 중앙값 대체 |
| `contest_award` | **100%** | 데이터에 정보 없음 | **컬럼 삭제** |
| `idea_contest` | **100%** | 데이터에 정보 없음 | **컬럼 삭제** |

<br>

- **핵심 관찰**: `class2` 결측률 **77.41%** 가 `re_registration` "아니요" 비율 **80.48%** 와 일치 → 결측이 오류가 아닌 신규 가입자의 구조적 특성임을 데이터로 확인
- `completed_semester` : 10학기 초과 값을 이상치로 판단하여 NaN 처리 후 **job 그룹별 중앙값 → 전체 중앙값** 순의 2단계 대체 적용
- 전처리 후 train/test 전체 결측치 **0** 달성

<br>

### 불필요 컬럼 제거

| 컬럼 | 제거 이유 |
|------|----------|
| `generation` | 모든 train 데이터가 9기로 동일 → 분산 없음 |
| `contest_award` | 100% 결측 |
| `idea_contest` | 100% 결측 |
| `contest_participation` | 결측 처리 후 인코딩 시 정보 중복 가능성 |

<br>

### Train/Test 형식 불일치 해결 (`major1_1`)

- Train: **계열 대분류** (예: `IT(컴퓨터 공학 포함)`, `경영학`)
- Test: **구체적 학과명** (예: `컴퓨터공학과`, `경영학부`)
- 규칙 기반 매핑 함수 `map_major_to_category()` 를 직접 구현하여 test의 학과명을 train 대분류로 통일
- 의약학 / 법학 / 교육학 / 예체능 / IT / 경영학 / 경제통상학 / 인문학 / 사회과학 / 자연과학 / 기타 **11개 대분류**로 분류

<br><br>

# EDA

### 클래스 불균형

- 타겟 변수 `completed` : **수료(1) 약 30% / 미수료(0) 약 70%**
- 단순 정확도(accuracy)로는 성능 평가 불가 → **F1 Score** 를 최적화 목표로 설정

<br>

### 핵심 수료 시그널

- **직전 기수(8기) 수강 여부**: 수강자 수료율 **42.6%** vs 미수강 **27.5%** — 가장 강력한 예측 변수
- **분반(class1)**: 분반에 따라 수료율이 유의미하게 차이남 → 티어(0~2)로 구조화
- **유입 경로**: 기존 학회원 소개 · 대외활동 사이트를 통한 유입자의 수료율 우위
- **재등록 여부**: 재등록 경험 × 재수강 경험 조합이 헌신도의 이중 지표로 작용

<br><br>

# 피처 엔지니어링

### 비정형 컬럼 구조화 (Multi-hot 인코딩)

자유서술형 텍스트 컬럼을 키워드 매칭으로 이진(0/1) 피처로 변환

| 원본 컬럼 | 처리 방식 | 생성 피처 예시 |
|----------|----------|---------------|
| `certificate_acquisition` | 7가지 자격증 multi-hot | `cert_ADsP`, `cert_SQLD`, `cert_빅데이터분석기사` 등 |
| `desired_certificate` | 동일 기준 multi-hot | `dcert_ADsP`, `dcert_SQLD` 등 |
| `desired_job` | 10개 직무 키워드 | `dj_분석가`, `dj_사이언티스트`, `dj_엔지니어` 등 |
| `desired_job_except_data` | 8개 도메인 키워드 | `dje_금융보험`, `dje_개발자` 등 |
| `onedayclass_topic` | 8개 주제 키워드 | `odc_Python`, `odc_ML_DL`, `odc_SQL` 등 |
| `expected_domain` | 10개 업종 코드 | `ed_정보통신`, `ed_금융보험` 등 |
| `major_field` | 11개 계열 multi-hot | `mf_IT`, `mf_경영학`, `mf_자연과학` 등 |
| `interested_company` | 7개 기업 카테고리 | `company_category`, `company_count` |

<br>

### 텍스트 임베딩 (실험)

- 키워드 매칭이 어려운 자유서술형 컬럼을 `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, 384차원) 로 벡터화
- **PCA (설명분산 95%)** 로 차원 축소 후 피처 추가 실험
- PCA는 train으로만 fit → test에 transform 적용 (leakage 방지)
- 최종 제출에는 미채택 (소규모 데이터에서 일반화 성능 기여 미미)

<br>

### 도메인 기반 파생변수 — 버전별 실험 (총 23개)

EDA에서 발굴한 수료 예측 시그널을 직접 변수로 설계하고 버전별로 추가

| 버전 | 주요 변경 내용 |
|------|--------------|
| **V1** | 기본 파생변수: `prev_count`, `is_returning`, `class_count`, `cert_gap`, `interest_diversity`, `time_per_semester`, `commitment` |
| **V2** | Zero-Importance 피처 4개 제거 (101 → 97개) |
| **V3** | SMOTE 클래스 불균형 처리 실험 (비채택) |
| **V4** ✅ | 수료 시그널 파생변수 추가: `took_8`, `consecutive_78`, `class1_tier`, `loyal_member`, `whyBDA_high`, `is_active_recruit` 등 — **최종 채택** |
| **V5** | 추가 파생변수 보완: `cert_ambition`, `total_engagement`, `data_job_focus`, `career_ambition`, `time_x_cert` 등 / 피처 **130개** — 비채택 (과적합) |

<br>

**핵심 파생변수**

| 변수 | 설명 | 근거 |
|------|------|------|
| `took_8` | 직전 기수(8기) 수강 여부 | 수강자 수료율 **42.6%** vs 미수강 **27.5%** — 최강 시그널 |
| `consecutive_78` | 7·8기 연속 수강 여부 | 장기 헌신도 복합 지표 |
| `class1_tier` | 1분반 수료율 티어 (0~2) | EDA로 분반별 수료율 유의미한 차이 확인 |
| `loyal_member` | 재수강 경험 × 재등록 여부 | 이중 확인된 헌신도 |
| `whyBDA_high` | 높은 내적 동기 여부 | 만족 기반 재수강자의 수료율 우위 |
| `is_active_recruit` | 능동적 유입 경로 여부 | 학회원 소개·대외활동 경로의 수료율 우위 |
| `total_engagement` | 이전 기수 + 자격증 + 원데이클래스 합산 | 전반적 참여 의지 척도 |
| `cert_ambition` | 보유 대비 희망 자격증 비율 | 성장 의지 척도 |

<br><br>

# Leakage-Free CV 설계

타겟 인코딩·순서형 인코딩은 타겟 정보를 간접 사용하므로 fold 외부에서 fit 시 데이터 누출(leakage) 발생

```
모델 비교  → StratifiedKFold(10)
튜닝·예측  → RepeatedStratifiedKFold(5×3, 15 folds)
  └─ 매 fold마다 Preprocessor 신규 생성 (상태 격리)
       ├─ Target Encoding   : fold X_train으로만 fit
       ├─ OrdinalEncoder    : fold X_train으로만 fit
       └─ LabelEncoder      : fold X_train으로만 fit
```

`Pipeline.encode_fold(X_tr, y_tr, X_val, X_test)` 를 fold 내부에서만 호출하여 leakage를 원천 차단

<br><br>

# 모델 비교 및 선택

LightGBM / XGBoost / CatBoost / RandomForest / ExtraTrees 5종으로 베이스라인 비교

| 순위 | Model | OOF F1 | Mean Fold F1 |
|------|-------|--------|--------------|
| 1 | **CatBoost** | **0.4696** | 0.4828 |
| 2 | LightGBM | 0.4657 | 0.4786 |
| 3 | ExtraTrees | 0.4630 | 0.4933 |
| 4 | RandomForest | 0.4607 | 0.4938 |
| 5 | XGBoost | 0.3846 | 0.4125 |

- **CatBoost 선택 이유**: OOF F1 최고, 범주형 변수 자체 처리 능력 + 소규모 데이터에서 안정적 성능
- OOF F1 기준 상위 3개 모델 (CatBoost · LightGBM · ExtraTrees) 의 공통 Zero-Importance 피처 4개 제거

<br>

### Zero-Importance 피처 제거

| 제거 피처 |
|-----------|
| `class4` |
| `dcert_컴퓨터활용능력` |
| `odc_AI모델` |
| `odc_Hadoop` |

> 101개 → **97개** 피처로 축소 → 모델 성능 전반 향상

<br><br>

# 결과

| Metric | Before Tuning | **After Optuna Tuning** |
|--------|--------------|------------------------|
| OOF F1 | 0.4727 | **0.5191** |
| Best Threshold | - | **0.51** |

- Optuna 500 trials, RepeatedStratifiedKFold(5×3) OOF F1 직접 최대화
- Threshold 0.10 ~ 0.70 그리드 서치로 F1 기준 최적 임계값 탐색
- 단일 Tuned CatBoost 채택 (stacking · 앙상블 대비 일반화 성능 우수)
- **최종 제출 파일**: `catboost_tuning_v3` (V4 피처셋 + Optuna 500 trials — 리더보드 최고 기록)

<br>

### 최종 성적

| 구분 | 순위 | F1 Score | 제출 파일 |
|------|------|----------|----------|
| Public (50% 샘플) | - | 0.44327 | `catboost_tuning_v3` |
| **Private (100%)** | **상위 4% (733명 중 29위)** | **0.40921** | `catboost_tuning_v3` |

<br><br>

# Lesson and Learned

- **도메인 기반 피처 설계의 중요성**: 소규모 데이터에서는 모델 복잡도보다 도메인 이해를 바탕으로 한 피처 설계가 성능에 더 큰 영향을 미침 (`took_8` 단일 변수가 수료율 15%p 차이를 설명)
- **결측치 의미 해석의 중요성**: 결측을 단순 통계값으로 채우지 않고 패턴을 분석하여 의미를 해석하는 것이 피처 품질에 직결 (`class2` 결측률 77% = 재등록 아니요 80%의 구조적 일치 발견)
- **Leakage-Free CV 설계**: 타겟 인코딩 등 타겟 정보를 사용하는 인코딩은 반드시 fold 내부에서 fit 해야 하며, 이를 파이프라인으로 구조화하여 실수 방지
- **Zero-Importance 피처 제거**: 단순하지만 Top-3 모델 공통 기준으로 제거 시 일관적인 일반화 성능 향상 효과 확인
- **복잡한 기법의 한계**: 텍스트 임베딩(sentence-transformers) · 스태킹 앙상블 등 복잡한 기법 실험 결과, 기본기(피처 설계 · CV · 튜닝)의 조합이 최종 성적을 결정
- **한계 인식**: 748건의 소규모 데이터로 인해 Public(0.443) → Private(0.409) 점수 하락 발생 → 더 많은 데이터 또는 강한 정규화 전략으로 일반화 성능 보완 가능

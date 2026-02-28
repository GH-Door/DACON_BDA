import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .preprocessing import Preprocessor

DEFAULTS = {
    # 결측치
    'previous_class_cols': [
        'previous_class_3', 'previous_class_4', 'previous_class_5',
        'previous_class_6', 'previous_class_7', 'previous_class_8',
    ],
    'str_fill_map': {
        'contest_participation': '없음',
        'major1_2': '없음',
        'major type': '단일 전공',
        'major1_1': '미응답',
        'major_field': '미응답',
    },
    'typo_map': {
        'major type': {'단일 전공공학 (컴퓨터 공학 제외)': '단일 전공'},
    },
    'num_fill_map': {'class2': 0, 'class3': 0, 'class4': 0},

    # completed_semester
    'semester_outlier_threshold': 10,

    # 제거 컬럼
    'drop_useless': ['generation', 'contest_award', 'idea_contest', 'contest_participation'],

    # 자격증
    'cert_map': {
        'ADsP': ['ADsP', 'adsp', 'ADSP'],
        'SQLD': ['SQLD', 'sqld'],
        '빅데이터분석기사': ['빅데이터 분석 기사', '빅데이터분석기사', '빅분기'],
        '정보처리기사': ['정보처리기사', '정보처리산업기사', '정보처리기능사'],
        '구글애널리스트': ['구글 애널리스트', '구글애널리스트'],
        '컴퓨터활용능력': ['컴퓨터 활용능력', '컴퓨터활용능력', '컴활'],
        '태블로': ['태블로'],
    },

    # 기업
    'company_categories': {
        'IT_플랫폼': ['네이버', '카카오', '토스', '쿠팡', '당근', '무신사', '배민', '배달의민족',
                    '라인', '넥슨', 'NC', '네카라쿠배', '야놀자', '직방', '왓챠', '리디',
                    '카카오뱅크', '토스뱅크', '카카오스타일', '지그재그', '마켓컬리'],
        'IT_글로벌': ['구글', 'Google', 'google', '애플', 'Apple', '마이크로소프트', 'MS',
                    '아마존', 'Amazon', 'AWS', '메타', 'Meta', '엔비디아', 'NVIDIA',
                    'openai', 'OpenAI'],
        '대기업': ['삼성', 'LG', 'SK', 'sk', 'Sk', '현대', 'CJ', '롯데', 'POSCO', '한화',
                 'GS', '두산', '하이닉스', 'SDS', 'CNS'],
        '금융': ['은행', '증권', '보험', '자산운용', 'KB', '신한', '하나', '우리', 'NH', 'IBK'],
        '공공': ['공기업', '공공기관', '한전', '한국전력', '공무원'],
        '제약_바이오': ['셀트리온', '삼성바이오', '한미약품', '제약'],
        '소비재': ['아모레퍼시픽', '매일유업', 'P&G', '올리브영'],
    },
    'company_priority': ['IT_플랫폼', 'IT_글로벌', '대기업', '금융', '공공', '제약_바이오', '소비재'],

    # multi-hot
    'mf_keys': [
        'IT (컴퓨터 공학 포함)', '공학 (컴퓨터 공학 제외)', '경영학', '자연과학',
        '사회과학', '인문학', '경제통상학', '예체능', '교육학', '의약학', '법학',
    ],
    'dj_map': {
        'dj_분석가': '데이터 분석가', 'dj_사이언티스트': '데이터 사이언티스트',
        'dj_엔지니어': '데이터 엔지니어', 'dj_AI전문가': '인공지능 전문가',
        'dj_마케터': '마케터', 'dj_PM기획': 'PM/서비스 기획자',
        'dj_개발자': '소프트웨어 개발자', 'dj_연구자': '연구자',
        'dj_MD': 'MD', 'dj_디자이너': 'UI/UX',
    },
    'dje_map': {
        'dje_금융보험': '금융 / 보험', 'dje_기획전략': '기획 / 전략',
        'dje_MD': 'MD', 'dje_개발자': '소프트웨어 개발자',
        'dje_디자이너': 'UI/UX', 'dje_PM기획': 'PM / 서비스 기획자',
        'dje_연구자': '자연과학계열 연구자', 'dje_마케터영업': '마케터 / 영업',
    },
    'odc_map': {
        'odc_Python': 'Python', 'odc_시각화': '시각화',
        'odc_ML_DL': '머신러닝', 'odc_SQL': 'SQL',
        'odc_크롤링': '크롤링', 'odc_AI모델': 'AI모델',
        'odc_Hadoop': 'Hadoop',
        'odc_Tableau': ['Tableau', '태블로', '테블로'],
    },
    'ed_map': {
        'ed_정보통신': 'J.', 'ed_금융보험': 'K.', 'ed_전문과학기술': 'M.',
        'ed_예술스포츠': 'R.', 'ed_국제외국기관': 'U.', 'ed_제조': 'C.',
        'ed_공공행정': 'O.', 'ed_교육': 'P.', 'ed_도매소매': 'G.', 'ed_보건복지': 'Q.',
    },

    # 원본 제거
    'drop_after_encoding': [
        'certificate_acquisition', 'desired_certificate', 'interested_company',
        'major_field', 'desired_job', 'desired_job_except_data',
        'onedayclass_topic', 'expected_domain', 'incumbents_company_level',
        'incumbents_lecture_scale_reason',
    ],

    # 이진 컬럼 매핑
    'binary_map': {
        're_registration': {'예': 1, '아니요': 0},
        'nationality': {'내국인': 1, '외국인': 0},
        'project_type': {'팀': 1, '개인': 0},
    },

    # 순서형 컬럼 (값 리스트가 순서를 의미)
    'ordinal_map': {
        'hope_for_group': [
            '아니요. 개인적으로 학회 활동을 하고 싶어요',
            '네. 온라인으로 참여하고 싶어요',
            '네. 오프라인으로 참여하고 싶어요',
        ],
        'incumbents_level': [
            '주니어 (0~3년차)',
            '시니어 (10년차 ~)',
        ],
        'incumbents_lecture_type': [
            '온라인',
            '온, 오프라인 동시',
            '오프라인',
        ],
        'incumbents_lecture_scale': [
            '10명 내외의 강의 리스너와 1명의 현직자',
            '3~50명 내외의 강의 리스너와 1명의 현직자',
            '100명 이상의 리스너와 1-2명의 현직자',
            '100명 이상의 리스너와 3명의 현직자',
            '100명 이상의 리스너와 10명 이상의 현직자',
            '...',
        ],
    },

    # Target Encoding 대상 (고카디널리티)
    'target_enc_cols': [
        'school1',
        'previous_class_3', 'previous_class_4', 'previous_class_5',
        'previous_class_6', 'previous_class_7', 'previous_class_8',
        'incumbents_lecture', 'whyBDA', 'what_to_gain',
    ],

    # 스케일러
    'scaler': None,  # 'robust', 'standard', 'minmax', None

    # 파생변수
    'use_features': True,

    # 텍스트 임베딩 (encode_text_embeddings)
    # drop_originals 이전 미활용 자유서술형 + target encoding 보완 컬럼
    'text_emb_cols': [
        'incumbents_lecture_scale_reason',  # 자유서술형 (target enc 미적용 컬럼)
    ],
    'text_emb_model': 'paraphrase-multilingual-MiniLM-L12-v2',  # 384차원, 한국어 지원
    'text_emb_n_components': 0.95,  # 설명분산 비율 (float: 자동 차원 결정, int: 고정 차원)

    # 건너뛸 단계
    'skip_steps': [],
}


def _split_certs(text):
    if pd.isna(text) or str(text).strip() in ('없음', '.', '해당없음', '딱히 없음'):
        return []
    cleaned = re.sub(r'\(([^)]*)\)', lambda m: m.group(0).replace(',', ';'), str(text))
    parts = [p.strip().replace(';', ',') for p in cleaned.split(',')]
    return [p for p in parts if p and p not in ('없음', '.', '해당없음')]


def _match_cert(cert_name, cert_map):
    for key, aliases in cert_map.items():
        if any(alias in cert_name for alias in aliases):
            return key
    return None


def _encode_certs(series, cert_map, prefix):
    keys = list(cert_map.keys())
    result = pd.DataFrame(0, index=series.index, columns=[f'{prefix}_{k}' for k in keys])
    result[f'{prefix}_count'] = 0
    for idx, val in series.items():
        certs = _split_certs(val)
        result.at[idx, f'{prefix}_count'] = len(certs)
        for c in certs:
            matched = _match_cert(c, cert_map)
            if matched:
                result.at[idx, f'{prefix}_{matched}'] = 1
    return result


def _category_company(text, categories, priority):
    if pd.isna(text):
        return '없음'
    t = str(text).strip()
    if t in ('.', '/', '없음', '아직 없음', '없습니다', '딱히 없음', '모름', '아직없음', ''):
        return '없음'
    matched = set()
    for cat, keywords in categories.items():
        if any(kw in t for kw in keywords):
            matched.add(cat)
    if not matched:
        return '기타'
    for p in priority:
        if p in matched:
            return p
    return '기타'


def _count_companies(text):
    if pd.isna(text):
        return 0
    t = str(text).strip()
    if t in ('.', '/', '없음', '아직 없음', '없습니다', '딱히 없음', '모름', ''):
        return 0
    return len([p for p in t.split(',') if p.strip()])


def _map_company_level(text):
    if pd.isna(text):
        return '기타'
    t = str(text).strip()
    if '빅테크' in t and '국내' in t:
        return '국내빅테크'
    if '대기업' in t:
        return '국내대기업'
    if '해외' in t:
        return '해외빅테크'
    if '스타트업' in t:
        return '스타트업'
    return '기타'


class Pipeline:
    """
    run()     → 기본 전처리 (결측치, 매핑, drop, multi-hot, 파생변수)
    encode_fold() → fold 내부에서 인코딩 + 스케일링 (leakage 방지)
    """

    def __init__(self, config: dict = None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.job_median = None
        self.overall_median = None
        self.train_nationality_mode = None

    def _skip(self, step_name: str) -> bool:
        return step_name in self.cfg['skip_steps']

    # ── 기본 전처리 단계들 ─────────────────────────────────────

    def fill_na(self, train: pd.DataFrame, test: pd.DataFrame):
        """결측치 처리 + 오타 교정"""
        self.train_nationality_mode = train['nationality'].mode()[0]

        for df in [train, test]:
            prev_cols = [c for c in self.cfg['previous_class_cols'] if c in df.columns]
            df[prev_cols] = df[prev_cols].fillna('해당없음')
            for col, val in self.cfg['str_fill_map'].items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)
            df['nationality'] = df['nationality'].fillna(self.train_nationality_mode)
            for col, val in self.cfg['num_fill_map'].items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)
            for col, mapping in self.cfg['typo_map'].items():
                if col in df.columns:
                    df[col] = df[col].replace(mapping)

        print("[fill_na] 결측치 처리 완료")
        return train, test

    def fix_completed_semester(self, train: pd.DataFrame, test: pd.DataFrame):
        """completed_semester 이상치 제거 + job 그룹 중앙값 대체 (train 기준)"""
        threshold = self.cfg['semester_outlier_threshold']
        train.loc[train['completed_semester'] > threshold, 'completed_semester'] = pd.NA
        self.job_median = train.groupby('job')['completed_semester'].median()
        self.overall_median = train['completed_semester'].median()

        for df in [train, test]:
            if df is not train:
                df.loc[df['completed_semester'] > threshold, 'completed_semester'] = pd.NA
            for job, med in self.job_median.items():
                mask = (df['completed_semester'].isna()) & (df['job'] == job)
                df.loc[mask, 'completed_semester'] = med
            df['completed_semester'] = df['completed_semester'].fillna(self.overall_median)

        print("[fix_completed_semester] completed_semester 처리 완료")
        return train, test

    def map_major(self, train: pd.DataFrame, test: pd.DataFrame):
        """test의 major1_1, major1_2를 train 대분류로 매핑"""
        test['major1_1'] = test['major1_1'].apply(Preprocessor.map_major_to_category)
        test['major1_2'] = test['major1_2'].apply(Preprocessor.map_major_to_category)
        print("[map_major] major 매핑 완료 (test only)")
        return train, test

    def drop_useless(self, train: pd.DataFrame, test: pd.DataFrame):
        """불필요 컬럼 제거"""
        drop = self.cfg['drop_useless']
        for df in [train, test]:
            cols = [c for c in drop if c in df.columns]
            df.drop(columns=cols, inplace=True)
        print(f"[drop_useless] {drop} 제거 완료")
        return train, test

    def encode_certs(self, train: pd.DataFrame, test: pd.DataFrame):
        """자격증 multi-hot 인코딩"""
        cert_map = self.cfg['cert_map']
        for df in [train, test]:
            cert = _encode_certs(df['certificate_acquisition'], cert_map, 'cert')
            dcert = _encode_certs(df['desired_certificate'], cert_map, 'dcert')
            for col in cert.columns:
                df[col] = cert[col].values
            for col in dcert.columns:
                df[col] = dcert[col].values
        print("[encode_certs] 자격증 인코딩 완료")
        return train, test

    def encode_companies(self, train: pd.DataFrame, test: pd.DataFrame):
        """관심 기업 카테고리화"""
        cats = self.cfg['company_categories']
        prio = self.cfg['company_priority']
        for df in [train, test]:
            df['company_category'] = df['interested_company'].apply(
                lambda x: _category_company(x, cats, prio)
            )
            df['company_count'] = df['interested_company'].apply(_count_companies)
        print("[encode_companies] 기업 카테고리 인코딩 완료")
        return train, test

    def encode_multi_hot(self, train: pd.DataFrame, test: pd.DataFrame):
        """multi-hot 인코딩 (major_field, desired_job 등)"""
        mf_keys = self.cfg['mf_keys']
        dj_map = self.cfg['dj_map']
        dje_map = self.cfg['dje_map']
        odc_map = self.cfg['odc_map']
        ed_map = self.cfg['ed_map']

        for df in [train, test]:
            for key in mf_keys:
                short = key.split('(')[0].strip().replace(' ', '')
                pattern = key.replace('(', r'\(').replace(')', r'\)')
                df[f'mf_{short}'] = df['major_field'].fillna('').str.contains(pattern).astype(int)
            df.loc[df['major_field'].fillna('').str.contains('자연고학'), 'mf_자연과학'] = 1
            mf_cols = [c for c in df.columns if c.startswith('mf_')]
            df['mf_count'] = df[mf_cols].sum(axis=1)

            text_dj = df['desired_job'].fillna('')
            for col, kw in dj_map.items():
                df[col] = text_dj.str.contains(kw, case=False).astype(int)
            df['dj_count'] = text_dj.apply(
                lambda x: len([p for p in x.split(',') if p.strip()]) if x else 0
            )

            text_dje = df['desired_job_except_data'].fillna('')
            for col, kw in dje_map.items():
                df[col] = text_dje.str.contains(kw, case=False).astype(int)
            df['dje_count'] = text_dje.apply(
                lambda x: len([p for p in x.split(',') if p.strip()]) if x else 0
            )

            text_odc = df['onedayclass_topic'].fillna('')
            for col, kw in odc_map.items():
                if isinstance(kw, list):
                    df[col] = text_odc.apply(lambda x, kws=kw: int(any(k in x for k in kws)))
                else:
                    df[col] = text_odc.str.contains(kw, case=False).astype(int)
            df['odc_count'] = text_odc.apply(
                lambda x: len([p for p in re.sub(r'\([^)]*\)', '', x).split(',') if p.strip()]) if x else 0
            )

            text_ed = df['expected_domain'].fillna('')
            for col, code in ed_map.items():
                df[col] = text_ed.str.contains(re.escape(code)).astype(int)
            df['ed_count'] = text_ed.apply(
                lambda x: len(re.findall(r'[A-Z]\.', x)) if x else 0
            )

            df['company_level'] = df['incumbents_company_level'].apply(_map_company_level)

        print("[encode_multi_hot] multi-hot 인코딩 완료")
        return train, test

    def encode_text_embeddings(self, train: pd.DataFrame, test: pd.DataFrame):
        """미활용 자유서술형 텍스트 → sentence-transformers 임베딩 → PCA 축소 후 피처 추가.

        - drop_originals 이전에 호출 (원본 텍스트 컬럼이 아직 존재하는 시점)
        - PCA는 train으로만 fit → leakage 방지
        - skip 가능: Pipeline(config={'skip_steps': ['encode_text_embeddings']})
        """
        from sentence_transformers import SentenceTransformer

        cols = [c for c in self.cfg['text_emb_cols'] if c in train.columns]
        if not cols:
            print("[encode_text_embeddings] 사용할 텍스트 컬럼 없음, 건너뜀")
            return train, test

        def _build_text(row):
            parts = []
            for c in cols:
                v = str(row[c]).strip() if pd.notna(row[c]) else ''
                if v and v not in ('nan', '없음', '해당없음', '.', ''):
                    parts.append(v)
            return ' [SEP] '.join(parts) if parts else '없음'

        train_texts = train.apply(_build_text, axis=1).tolist()
        test_texts  = test.apply(_build_text, axis=1).tolist()

        model = SentenceTransformer(self.cfg['text_emb_model'])
        train_embs = model.encode(train_texts, batch_size=64, show_progress_bar=False)
        test_embs  = model.encode(test_texts,  batch_size=64, show_progress_bar=False)

        n_comp_cfg = self.cfg['text_emb_n_components']
        max_comp = min(train_embs.shape[1], len(train_embs) - 1)
        if isinstance(n_comp_cfg, float) and 0.0 < n_comp_cfg < 1.0:
            # float → 설명분산 비율로 자동 차원 결정 (sklearn PCA 기본 동작)
            n_comp = min(n_comp_cfg, 1.0 - 1e-10)  # 1.0 미만 보장
        else:
            n_comp = min(int(n_comp_cfg), max_comp)

        pca = PCA(n_components=n_comp, random_state=42)
        train_pca = pca.fit_transform(train_embs)   # train으로만 fit
        test_pca  = pca.transform(test_embs)
        actual_n = pca.n_components_  # 실제 선택된 차원 수

        for i in range(actual_n):
            train[f'emb_{i}'] = train_pca[:, i]
            test[f'emb_{i}']  = test_pca[:, i]

        if isinstance(n_comp_cfg, float) and 0.0 < n_comp_cfg < 1.0:
            explained = pca.explained_variance_ratio_.sum()
            print(f"[encode_text_embeddings] {len(cols)}개 컬럼 → {actual_n}차원 (설명분산 {explained:.3f})")
        else:
            print(f"[encode_text_embeddings] {len(cols)}개 컬럼 → {actual_n}차원 임베딩 피처 추가")
        return train, test

    def drop_originals(self, train: pd.DataFrame, test: pd.DataFrame):
        """인코딩 후 원본 컬럼 제거"""
        drop = self.cfg['drop_after_encoding']
        for df in [train, test]:
            cols = [c for c in drop if c in df.columns]
            df.drop(columns=cols, inplace=True)
        print("[drop_originals] 원본 컬럼 제거 완료")
        return train, test

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        """파생변수 생성 (인코딩/스케일링 전, 원본 값 기준)"""
        prev_cols = [c for c in self.cfg['previous_class_cols'] if c in train.columns]

        for df in [train, test]:
            # ── 기존 파생변수 ───────────────────────────────────────
            df['prev_count'] = (df[prev_cols] != '해당없음').sum(axis=1)
            df['is_returning'] = (df['prev_count'] > 0).astype(int)

            class_cols = ['class2', 'class3', 'class4']
            existing = [c for c in class_cols if c in df.columns]
            df['class_count'] = 1 + (df[existing] != 0).sum(axis=1)

            df['cert_gap'] = df['dcert_count'] - df['cert_count']
            df['interest_diversity'] = df['dj_count'] + df['mf_count'] + df['ed_count']
            df['time_per_semester'] = df['time_input'] / (df['completed_semester'].abs() + 1)

            re_reg = (df['re_registration'] == '예').astype(int) if df['re_registration'].dtype == 'object' else df['re_registration']
            df['commitment'] = df['cert_count'] + re_reg

            # ── 신규 파생변수 ───────────────────────────────────────

            # 1. 동기 강도 (whyBDA): 재수강 만족/혜택=고동기, 현직자 강의=저동기
            why = df['whyBDA'].fillna('')
            df['whyBDA_high'] = why.isin([
                '이전 기수에 매우 만족해서',
                'BDA 학회원만의 혜택을 누리고 싶어서(현직자 강연, 잡 페스티벌, 기업연계 공모전 등)',
            ]).astype(int)
            df['whyBDA_passive'] = (why == '현직자의 강의를 듣고 싶어서').astype(int)

            # 2. 유입 경로 능동성: 기존 학회원/운영진·대외활동 사이트는 수료율이 높음
            route = df['inflow_route'].fillna('')
            df['is_active_recruit'] = route.isin([
                '기존 학회원 또는 운영진',
                '대외활동 사이트(링커리어, 캠퍼스픽, 캠퍼즈, 위비티 등)',
            ]).astype(int)

            # 3. 복수전공 여부: 복수전공자 수료율이 단일전공보다 높음
            df['is_double_major'] = df['major type'].fillna('').str.contains('복수').astype(int)

            # 4. 자격증 야망 지수: 보유 자격증 대비 희망 자격증 비율
            df['cert_ambition'] = df['dcert_count'] / (df['cert_count'] + 1)

            # 5. 데이터 핵심 자격증 보유 여부 (ADsP, SQLD, 빅데이터분석기사)
            key_cert_cols = [c for c in ['cert_ADsP', 'cert_SQLD', 'cert_빅데이터분석기사'] if c in df.columns]
            df['has_key_cert'] = (df[key_cert_cols].sum(axis=1) > 0).astype(int) if key_cert_cols else 0

            # 6. 데이터 직무 집중도: 분석가/사이언티스트/엔지니어/AI 희망 수
            data_job_cols = [c for c in ['dj_분석가', 'dj_사이언티스트', 'dj_엔지니어', 'dj_AI전문가'] if c in df.columns]
            df['data_job_focus'] = df[data_job_cols].sum(axis=1) if data_job_cols else 0

            # 7. 종합 활동 참여 지수: 이전 기수 + 보유 자격증 + 원데이클래스 관심도
            df['total_engagement'] = df['prev_count'] + df['cert_count'] + df['odc_count']

            # 8. 충성 회원 지표: 재수강 경험 × 재등록 (이중 확인된 헌신도)
            df['loyal_member'] = df['is_returning'] * re_reg

            # 9. 경력 야망: 창업 또는 대학원 진학 희망 (수료율이 상대적으로 높음)
            career = df['desired_career_path'].fillna('')
            df['career_ambition'] = career.isin(['창업', '대학원 진학']).astype(int)

            # 10. 시간 × 자격증 상호작용: 시간 투자 의지와 자격증 노력의 결합
            df['time_x_cert'] = df['time_input'] * df['cert_count']

            # 11. 온라인 선호 여부 (오프라인 그룹 대비 온라인/개인이 수료율 더 높음)
            df['prefers_online'] = (
                df['hope_for_group'].fillna('') != '네. 오프라인으로 참여하고 싶어요'
            ).astype(int)

            # 12. 직전 기수(8기) 수강 여부: 수료율 42.6% vs 27.5% — 가장 강력한 시그널
            if 'previous_class_8' in df.columns:
                df['took_8'] = (df['previous_class_8'] != '해당없음').astype(int)
            else:
                df['took_8'] = 0

            # 13. 7+8기 연속 수강 여부
            took_7 = (df['previous_class_7'] != '해당없음').astype(int) if 'previous_class_7' in df.columns else 0
            df['consecutive_78'] = df['took_8'] * took_7

            # 14. class1 분반 수료율 티어 (high: 12,6 / mid: 11,5,2 / low: 나머지)
            def _class1_tier(v):
                if v in (12, 6):
                    return 2
                if v in (11, 5, 2):
                    return 1
                return 0
            df['class1_tier'] = df['class1'].apply(_class1_tier)

            # 15. 이전 기수 분반 수준 (기초=1, 입문=2, 중급=3, 고급=4)
            def _class_level(text):
                if pd.isna(text) or text == '해당없음':
                    return 0
                t = str(text)
                if '모델링' in t:
                    return 4
                if '적용' in t:
                    return 3
                if '전처리' in t or '머신러닝' in t:
                    return 3
                if '입문' in t or '개념' in t:
                    return 2
                if '기초' in t:
                    return 1
                return 1

            level_cols = [c for c in prev_cols if c in df.columns]
            if level_cols:
                df['prev_max_level'] = df[level_cols].apply(
                    lambda row: max(_class_level(v) for v in row), axis=1
                )
                df['prev_8_level'] = df['previous_class_8'].apply(_class_level) if 'previous_class_8' in df.columns else 0
            else:
                df['prev_max_level'] = 0
                df['prev_8_level'] = 0

        print("[create_features] 파생변수 23개 생성 완료")
        return train, test

    # ── run: 기본 전처리만 (인코딩/스케일링 제외) ──────────────

    def run(self, train: pd.DataFrame, test: pd.DataFrame):
        """기본 전처리 실행. 인코딩/스케일링은 encode_fold()에서."""
        train = train.copy()
        test = test.copy()

        steps = [
            ('fill_na',                lambda: self.fill_na(train, test)),
            ('fix_completed_semester', lambda: self.fix_completed_semester(train, test)),
            ('map_major',              lambda: self.map_major(train, test)),
            ('drop_useless',           lambda: self.drop_useless(train, test)),
            ('encode_certs',           lambda: self.encode_certs(train, test)),
            ('encode_companies',       lambda: self.encode_companies(train, test)),
            ('encode_multi_hot',       lambda: self.encode_multi_hot(train, test)),
            ('encode_text_embeddings', lambda: self.encode_text_embeddings(train, test)),
            ('drop_originals',         lambda: self.drop_originals(train, test)),
            ('create_features',        lambda: self.create_features(train, test)),
        ]

        for name, fn in steps:
            if self._skip(name):
                print(f"[{name}] 건너뜀")
                continue
            result = fn()
            train, test = result[0], result[1]

        # ID 분리
        train.drop(columns=['ID'], inplace=True, errors='ignore')
        test_id = test['ID'].copy() if 'ID' in test.columns else None
        test.drop(columns=['ID'], inplace=True, errors='ignore')
        print(f"Preprocessing completed: train {train.shape}, test {test.shape}")
        return train, test, test_id

    # ── encode_fold: CV fold 안에서 호출 (leakage 방지) ──────

    def encode_fold(self, X_tr, y_tr, X_val, X_test=None):
        """fold 내부에서 인코딩 + 스케일링. X_tr로만 fit, 나머지는 transform.

        매 fold마다 새 Preprocessor를 생성하여 leakage를 원천 차단.

        Parameters
        ----------
        X_tr : DataFrame  - fold 학습 데이터
        y_tr : Series     - fold 학습 타겟 (target encoding용)
        X_val : DataFrame - fold 검증 데이터
        X_test : DataFrame or None - 테스트 데이터

        Returns
        -------
        X_tr, X_val, X_test (모두 수치형, 복사본)
        """
        X_tr = X_tr.copy()
        X_val = X_val.copy()
        X_test = X_test.copy() if X_test is not None else None
        others = [X_val] + ([X_test] if X_test is not None else [])

        # 매 fold 새 Preprocessor (상태 격리)
        prep = Preprocessor()

        # 1) bool → int
        for df in [X_tr] + others:
            for col in df.select_dtypes(include='bool').columns:
                df[col] = df[col].astype(int)

        # 2) 이진 컬럼 → 0/1
        for col, mapping in self.cfg['binary_map'].items():
            for df in [X_tr] + others:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].map(mapping).fillna(0).astype(int)

        # 3) 순서형 → OrdinalEncoder (X_tr로 fit)
        ordinal_map = self.cfg['ordinal_map']
        ordinal_cols = [c for c in ordinal_map if c in X_tr.columns and X_tr[c].dtype == 'object']
        if ordinal_cols:
            categories = [ordinal_map[c] for c in ordinal_cols]
            X_tr = prep.encoder(X_tr, ordinal_cols, method='ordinal', fit=True, categories=categories)
            for df_idx, df in enumerate(others):
                others[df_idx] = prep.encoder(df, ordinal_cols, method='ordinal', fit=False)

        # 4) Target Encoding (X_tr + y_tr로 fit)
        te_cols = [c for c in self.cfg['target_enc_cols'] if c in X_tr.columns]
        if te_cols:
            X_tr = prep.target_encoder(X_tr, te_cols, y=y_tr, fit=True)
            for df_idx, df in enumerate(others):
                others[df_idx] = prep.target_encoder(df, te_cols, fit=False)

        # 5) 명목형 → LabelEncoder (나머지 object 컬럼)
        str_cols = X_tr.select_dtypes(include='object').columns.tolist()
        if str_cols:
            X_tr = prep.encoder(X_tr, str_cols, method='label', fit=True)
            for df_idx, df in enumerate(others):
                others[df_idx] = prep.encoder(df, str_cols, method='label', fit=False)

        # 6) 스케일링 (X_tr로 fit)
        scaler_name = self.cfg['scaler']
        if scaler_name:
            exclude = {'completed'}
            num_cols = [c for c in X_tr.select_dtypes(include='number').columns if c not in exclude]
            X_tr = prep.scaler(X_tr, columns=num_cols, method=scaler_name, fit=True)
            for df_idx, df in enumerate(others):
                others[df_idx] = prep.scaler(df, columns=num_cols, method=scaler_name, fit=False)

        X_val = others[0]
        X_test = others[1] if X_test is not None else None
        return X_tr, X_val, X_test

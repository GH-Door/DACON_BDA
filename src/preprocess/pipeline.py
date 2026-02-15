"""
BDA 수료 예측 대회 전처리 파이프라인

사용법:
    # 기본 설정으로 한번에 실행
    pipe = BDAPipeline()
    train, test, test_id = pipe.run(train_raw, test_raw)

    # 단계별 실행
    pipe = BDAPipeline()
    train, test = pipe.step1_fill_na(train, test)
    train, test = pipe.step2_completed_semester(train, test)
    ...

    # 커스텀 설정
    pipe = BDAPipeline(config={
        'scaler': 'standard',          # 'robust', 'standard', 'minmax'
        'encoder': 'label',            # 'label', 'ordinal'
        'cert_map': my_custom_map,
        'drop_useless': ['generation', 'contest_award', 'idea_contest', 'extra_col'],
        'skip_steps': ['step10_scaling'],
    })
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler
)

from .preprocessing import Preprocessor


DEFAULTS = {
    # Step1: 결측치
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
    'num_fill_map': {'class2': 0, 'class3': 0, 'class4': 0},

    # Step2: completed_semester
    'semester_outlier_threshold': 10,

    # Step4: 제거 컬럼
    'drop_useless': ['generation', 'contest_award', 'idea_contest', 'contest_participation'],

    # Step5: 자격증
    'cert_map': {
        'ADsP': ['ADsP', 'adsp', 'ADSP'],
        'SQLD': ['SQLD', 'sqld'],
        '빅데이터분석기사': ['빅데이터 분석 기사', '빅데이터분석기사', '빅분기'],
        '정보처리기사': ['정보처리기사', '정보처리산업기사', '정보처리기능사'],
        '구글애널리스트': ['구글 애널리스트', '구글애널리스트'],
        '컴퓨터활용능력': ['컴퓨터 활용능력', '컴퓨터활용능력', '컴활'],
        '태블로': ['태블로'],
    },

    # Step6: 기업
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

    # Step7: multi-hot
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

    # Step8: 원본 제거
    'drop_after_encoding': [
        'certificate_acquisition', 'desired_certificate', 'interested_company',
        'major_field', 'desired_job', 'desired_job_except_data',
        'onedayclass_topic', 'expected_domain', 'incumbents_company_level',
        'incumbents_lecture_scale_reason',
    ],

    # Step9: 인코딩 방식
    'encoder': 'label',  # 'label' or 'ordinal'

    # Step10: 스케일러
    'scaler': 'robust',  # 'robust', 'standard', 'minmax', None

    # 건너뛸 단계
    'skip_steps': [],
}

SCALERS = {
    'robust': RobustScaler,
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
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
    config dict로 모든 설정을 동적 변경 가능.
    skip_steps로 특정 단계 건너뛰기 가능.
    """

    def __init__(self, config: dict = None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.label_encoders = {}
        self.ordinal_encoder = None
        self.scaler_obj = None
        self.job_median = None
        self.overall_median = None
        self.train_nationality_mode = None

    def _skip(self, step_name: str) -> bool:
        return step_name in self.cfg['skip_steps']

    def step1_fill_na(self, train: pd.DataFrame, test: pd.DataFrame):
        """결측치 처리"""
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

        print("[Step1] 결측치 처리 완료")
        return train, test

    def step2_completed_semester(self, train: pd.DataFrame, test: pd.DataFrame):
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

        print("[Step2] completed_semester 처리 완료")
        return train, test

    def step3_map_major(self, train: pd.DataFrame, test: pd.DataFrame):
        """test의 major1_1, major1_2를 train 대분류로 매핑"""
        test['major1_1'] = test['major1_1'].apply(Preprocessor.map_major_to_category)
        test['major1_2'] = test['major1_2'].apply(Preprocessor.map_major_to_category)
        print("[Step3] major 매핑 완료 (test only)")
        return train, test

    def step4_drop_useless(self, train: pd.DataFrame, test: pd.DataFrame):
        """불필요 컬럼 제거"""
        drop = self.cfg['drop_useless']
        for df in [train, test]:
            cols = [c for c in drop if c in df.columns]
            df.drop(columns=cols, inplace=True)
        print(f"[Step4] {drop} 제거 완료")
        return train, test

    def step5_cert_encoding(self, train: pd.DataFrame, test: pd.DataFrame):
        """자격증 multi-hot 인코딩"""
        cert_map = self.cfg['cert_map']
        for df in [train, test]:
            cert = _encode_certs(df['certificate_acquisition'], cert_map, 'cert')
            dcert = _encode_certs(df['desired_certificate'], cert_map, 'dcert')
            for col in cert.columns:
                df[col] = cert[col].values
            for col in dcert.columns:
                df[col] = dcert[col].values
        print("[Step5] 자격증 인코딩 완료")
        return train, test

    def step6_company_encoding(self, train: pd.DataFrame, test: pd.DataFrame):
        """관심 기업 카테고리화"""
        cats = self.cfg['company_categories']
        prio = self.cfg['company_priority']
        for df in [train, test]:
            df['company_category'] = df['interested_company'].apply(
                lambda x: _category_company(x, cats, prio)
            )
            df['company_count'] = df['interested_company'].apply(_count_companies)
        print("[Step6] 기업 카테고리 인코딩 완료")
        return train, test

    def step7_multi_hot(self, train: pd.DataFrame, test: pd.DataFrame):
        """multi-hot 인코딩 (major_field, desired_job 등)"""
        mf_keys = self.cfg['mf_keys']
        dj_map = self.cfg['dj_map']
        dje_map = self.cfg['dje_map']
        odc_map = self.cfg['odc_map']
        ed_map = self.cfg['ed_map']

        for df in [train, test]:
            # major_field
            for key in mf_keys:
                short = key.split('(')[0].strip().replace(' ', '')
                pattern = key.replace('(', r'\(').replace(')', r'\)')
                df[f'mf_{short}'] = df['major_field'].fillna('').str.contains(pattern).astype(int)
            df.loc[df['major_field'].fillna('').str.contains('자연고학'), 'mf_자연과학'] = 1
            mf_cols = [c for c in df.columns if c.startswith('mf_')]
            df['mf_count'] = df[mf_cols].sum(axis=1)

            # desired_job
            text_dj = df['desired_job'].fillna('')
            for col, kw in dj_map.items():
                df[col] = text_dj.str.contains(kw, case=False).astype(int)
            df['dj_count'] = text_dj.apply(
                lambda x: len([p for p in x.split(',') if p.strip()]) if x else 0
            )

            # desired_job_except_data
            text_dje = df['desired_job_except_data'].fillna('')
            for col, kw in dje_map.items():
                df[col] = text_dje.str.contains(kw, case=False).astype(int)
            df['dje_count'] = text_dje.apply(
                lambda x: len([p for p in x.split(',') if p.strip()]) if x else 0
            )

            # onedayclass_topic
            text_odc = df['onedayclass_topic'].fillna('')
            for col, kw in odc_map.items():
                if isinstance(kw, list):
                    df[col] = text_odc.apply(lambda x, kws=kw: int(any(k in x for k in kws)))
                else:
                    df[col] = text_odc.str.contains(kw, case=False).astype(int)
            df['odc_count'] = text_odc.apply(
                lambda x: len([p for p in re.sub(r'\([^)]*\)', '', x).split(',') if p.strip()]) if x else 0
            )

            # expected_domain
            text_ed = df['expected_domain'].fillna('')
            for col, code in ed_map.items():
                df[col] = text_ed.str.contains(re.escape(code)).astype(int)
            df['ed_count'] = text_ed.apply(
                lambda x: len(re.findall(r'[A-Z]\.', x)) if x else 0
            )

            # incumbents_company_level
            df['company_level'] = df['incumbents_company_level'].apply(_map_company_level)

        print("[Step7] multi-hot 인코딩 완료")
        return train, test

    def step8_drop_originals(self, train: pd.DataFrame, test: pd.DataFrame):
        """인코딩 후 원본 컬럼 제거"""
        drop = self.cfg['drop_after_encoding']
        for df in [train, test]:
            cols = [c for c in drop if c in df.columns]
            df.drop(columns=cols, inplace=True)
        print(f"[Step8] 원본 컬럼 제거 완료")
        return train, test

    def step9_label_encoding(self, train: pd.DataFrame, test: pd.DataFrame):
        """인코딩 + ID 분리"""
        train.drop(columns=['ID'], inplace=True, errors='ignore')
        test_id = test['ID'].copy() if 'ID' in test.columns else None
        test.drop(columns=['ID'], inplace=True, errors='ignore')

        str_cols = train.select_dtypes(include='object').columns.tolist()
        method = self.cfg['encoder']

        if method == 'label':
            self.label_encoders = {}
            for col in str_cols:
                le = LabelEncoder()
                le.fit(pd.concat([train[col], test[col]]).astype(str))
                train[col] = le.transform(train[col].astype(str))
                test[col] = le.transform(test[col].astype(str))
                self.label_encoders[col] = le

        elif method == 'ordinal':
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1
            )
            self.ordinal_encoder.fit(
                pd.concat([train[str_cols], test[str_cols]]).astype(str)
            )
            train[str_cols] = self.ordinal_encoder.transform(train[str_cols].astype(str))
            test[str_cols] = self.ordinal_encoder.transform(test[str_cols].astype(str))

        print(f"[Step9] {method} 인코딩 {len(str_cols)}개 컬럼 완료, ID 분리됨")
        return train, test, test_id

    def step10_scaling(self, train: pd.DataFrame, test: pd.DataFrame,
                       exclude: list[str] = None):
        """스케일링 (train fit → test transform)"""
        scaler_name = self.cfg['scaler']
        if scaler_name is None:
            print("[Step10] 스케일링 건너뜀 (scaler=None)")
            return train, test

        if exclude is None:
            exclude = ['completed']
        num_cols = train.select_dtypes(include='number').columns.tolist()
        num_cols = [c for c in num_cols if c not in exclude]

        self.scaler_obj = SCALERS[scaler_name]()
        train[num_cols] = self.scaler_obj.fit_transform(train[num_cols])
        test[num_cols] = self.scaler_obj.transform(test[num_cols])

        print(f"[Step10] {scaler_name} 스케일링 {len(num_cols)}개 컬럼 완료")
        return train, test

    def run(self, train: pd.DataFrame, test: pd.DataFrame):
        """전체 파이프라인 한번에 실행 (skip_steps로 건너뛰기 가능)"""
        train = train.copy()
        test = test.copy()
        test_id = None

        steps = [
            ('step1_fill_na',            lambda: self.step1_fill_na(train, test)),
            ('step2_completed_semester',  lambda: self.step2_completed_semester(train, test)),
            ('step3_map_major',           lambda: self.step3_map_major(train, test)),
            ('step4_drop_useless',        lambda: self.step4_drop_useless(train, test)),
            ('step5_cert_encoding',       lambda: self.step5_cert_encoding(train, test)),
            ('step6_company_encoding',    lambda: self.step6_company_encoding(train, test)),
            ('step7_multi_hot',           lambda: self.step7_multi_hot(train, test)),
            ('step8_drop_originals',      lambda: self.step8_drop_originals(train, test)),
            ('step9_label_encoding',      None),  # special: returns test_id
            ('step10_scaling',            lambda: self.step10_scaling(train, test)),
        ]

        for name, fn in steps:
            if self._skip(name):
                print(f"[{name}] 건너뜀")
                continue

            if name == 'step9_label_encoding':
                train, test, test_id = self.step9_label_encoding(train, test)
            else:
                result = fn()
                train, test = result[0], result[1]

        print(f"\n전처리 완료: train {train.shape}, test {test.shape}")
        return train, test, test_id

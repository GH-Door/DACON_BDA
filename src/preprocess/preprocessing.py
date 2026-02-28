import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler, RobustScaler, StandardScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder
)
from category_encoders import TargetEncoder
warnings.filterwarnings('ignore')


class Preprocessor:
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
    }

    def __init__(self) -> None:
        self.encoders = {}
        self.scaler_obj = None

    # scaler
    def scaler(self, df: pd.DataFrame, columns: list[str] = None,
              method: str = 'standard', fit: bool = True) -> pd.DataFrame:

        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return df

        if fit:
            self.scaler_obj = self.SCALERS[method]()
            self.scaler_obj.fit(df[cols])

        df[cols] = self.scaler_obj.transform(df[cols])
        return df

    # encoder
    def encoder(self, df: pd.DataFrame, columns: list[str],
               method: str = 'label', fit: bool = True,
               drop_first: bool = True, categories: list[list[str]] = None) -> pd.DataFrame:

        cols = [c for c in columns if c in df.columns]
        if not cols:
            return df

        if method == 'label':
            for col in cols:
                if fit:
                    le = LabelEncoder()
                    le.fit(df[col].dropna().astype(str))
                    self.encoders[col] = le
                le = self.encoders[col]
                mask = df[col].notna()
                known = set(le.classes_)
                df.loc[mask, col] = df.loc[mask, col].astype(str).apply(
                    lambda x, k=known, e=le: e.transform([x])[0] if x in k else -1
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)

        elif method == 'ordinal':
            key = '_ordinal_'
            if fit:
                oe = OrdinalEncoder(
                    categories=categories if categories else 'auto',
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )
                oe.fit(df[cols].astype(str))
                self.encoders[key] = oe
            df[cols] = self.encoders[key].transform(df[cols].astype(str))

        elif method == 'onehot':
            key = '_onehot_'
            if fit:
                ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None,
                                    handle_unknown='ignore')
                ohe.fit(df[cols].astype(str))
                self.encoders[key] = ohe
            ohe = self.encoders[key]
            encoded = ohe.transform(df[cols].astype(str))
            feature_names = ohe.get_feature_names_out(cols)
            df = df.drop(columns=cols)
            df[feature_names] = encoded

        return df

    # target encoder
    def target_encoder(self, df: pd.DataFrame, columns: list[str], y: pd.Series = None,
                       fit: bool = True, min_samples_leaf: int = 5,
                       smoothing: float = 10.0) -> pd.DataFrame:
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return df

        for c in cols:
            df[c] = df[c].astype(str)

        key = '_target_enc_'
        if fit:
            te = TargetEncoder(cols=cols, min_samples_leaf=min_samples_leaf, smoothing=smoothing)
            df[cols] = te.fit_transform(df[cols], y)
            self.encoders[key] = te
        else:
            df[cols] = self.encoders[key].transform(df[cols])

        return df

    # drop columns
    def drop_cols(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        cols = [c for c in columns if c in df.columns]
        return df.drop(columns=cols)

    # split
    @staticmethod
    def split(df: pd.DataFrame, target: str = 'completed',
              test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if stratify else None
        )

    @staticmethod
    def map_major_to_category(major):
        """test의 구체적 학과명을 train의 대분류 카테고리로 매핑"""
        if pd.isna(major):
            return '미응답'

        m = str(major).strip()
        m_upper = m.upper()

        # 없음/오타 처리
        if m in ('없음', '해당없음'):
            return '없음'
        if m == '걍영학부':
            return '경영학'

        # 의약학
        if any(k in m for k in ['의예', '의학과', '약학', '수의학', '의료', '의공학', '의생명', '의과학']):
            return '의약학'

        # 법학
        if '법학' in m or '법무' in m:
            return '법학'

        # 교육학
        if '교육' in m:
            return '교육학'

        # 예체능
        if any(k in m for k in ['스포츠', '체육', '디자인', '미술', '음악', '작곡',
                                '의류', '의상', '실내건축', '바둑', '레저']):
            return '예체능'

        # IT(컴퓨터 공학 포함) - 전자/전기 포함
        it_kw = ['컴퓨터', '소프트웨어', '인공지능', '빅데이터', '데이터사이언스',
                 '데이터과학', '데이터 사이언스', '정보통신', '정보시스템', '정보융합',
                 '디지털소프트웨어', '정보보안', '융합보안', '산업보안', '데이터정보',
                 '반도체', '데이터', '전자', '전기']
        if any(k in m for k in it_kw):
            return 'IT(컴퓨터 공학 포함)'
        if 'AI' in m_upper and any(k in m for k in ['학', '부', '과']):
            return 'IT(컴퓨터 공학 포함)'
        if 'ICT' in m_upper:
            return 'IT(컴퓨터 공학 포함)'
        if 'SW' in m_upper and any(k in m for k in ['학', '부', '과']):
            return 'IT(컴퓨터 공학 포함)'
        if 'IT' in m_upper and any(k in m for k in ['학', '부', '과']):
            return 'IT(컴퓨터 공학 포함)'

        # 경영학 (공학 제외)
        if '경영' in m and '공학' not in m:
            return '경영학'

        # 경제통상학
        if any(k in m for k in ['경제', '통상', '무역', '금융', '국제물류', '계량위험']):
            return '경제통상학'

        # 인문학
        hum_kw = ['국어', '국문', '영어', '영문', '일어', '일문', '중어', '중문',
                  '중국어', '한문', '한국어', '언어', '문학', '사학', '동양사', '한국사',
                  '인문', '문화', '노어', '스페인', '아랍어', '포르투갈어', '독어', '독문',
                  '루마니아', '몽골어', '베트남어', '태국', '스칸디나비아', '체코슬로바키아',
                  '터키', '러시아', '중국학', 'ELLT', '영어통번역']
        if any(k in m for k in hum_kw):
            return '인문학'

        # 사회과학
        soc_kw = ['사회', '심리', '정치', '행정', '정책', '미디어', '커뮤니케이션',
                  '광고', '홍보', '문헌정보', '부동산', '관광', '지리',
                  '소비자', '보건', '국제관계', '국제사무', '국제학', '정보사회',
                  '방송', '지적', '물류']
        if any(k in m for k in soc_kw):
            return '사회과학'

        # 자연과학 (순수과학 + 공학 계열)
        sci_kw = ['수학', '통계', '물리', '화학', '생명', '생물', '식품', '환경',
                  '바이오', '나노', '동물', '농업', '수리', '화공', '신소재',
                  '에너지', '유기재료', '공학', '기계', '건축', '토목', '항공',
                  '산업', '시스템', '융합', '스마트', '조경']
        if any(k in m for k in sci_kw):
            return '자연과학'

        return '기타'

    def pca(self):
        # TODO: PCA 구현
        pass
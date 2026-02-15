import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
import seaborn as sns


def set_korean_font(verbose: bool = False) -> bool:
    """한글 폰트 설정. verbose=True면 메시지 출력."""
    path = '/Library/Fonts/Arial Unicode.ttf'

    if platform.system() == 'Darwin':
        rc('font', family='Arial Unicode MS')
        if verbose:
            print('Hangul OK in your MAC!')
        success = True
    elif platform.system() == 'Windows':
        try:
            font_name = font_manager.FontProperties(fname=path).get_name()
            rc('font', family=font_name)
            if verbose:
                print(f'Hangul OK with {font_name}!')
            success = True
        except:
            if verbose:
                print('Font not found')
            success = False
    else:
        if verbose:
            print('Unknown system')
        success = False

    plt.rcParams['axes.unicode_minus'] = False
    return success


def count_plot(df, col, ax=None, figsize=(10, 6), palette="Blues_r", rotation=None, title=None, xlabel=None, ylabel=None, order='desc', orient='v', show=True):
    """
    order: 'desc'(내림차순), 'asc'(오름차순), None(정렬 안 함)
    orient: 'v'(세로, 기본값), 'h'(가로)
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # 정렬 순서 결정
    if order == 'desc':
        order_list = df[col].value_counts().index.tolist()
    elif order == 'asc':
        order_list = df[col].value_counts(ascending=True).index.tolist()
    else:
        order_list = None

    # 방향에 따라 x/y 설정
    if orient == 'h':
        sns.countplot(data=df, y=col, palette=palette, ax=ax, order=order_list)
        ax.set_ylabel(ylabel if ylabel is not None else col)
        ax.set_xlabel(xlabel if xlabel is not None else 'Count')
    else:
        sns.countplot(data=df, x=col, palette=palette, ax=ax, order=order_list)
        ax.set_xlabel(xlabel if xlabel is not None else col)
        ax.set_ylabel(ylabel if ylabel is not None else 'Count')

    ax.set_title(title if title else f'{col} Count')
    if rotation:
        ax.tick_params(axis='x' if orient == 'v' else 'y', rotation=rotation)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def bar_plot(df, x_col, y_col, ax=None, figsize=(10, 6), hue=None, palette="Blues_r", rotation=None, title=None, xlabel=None, ylabel=None, show=True):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if hue:
        sns.barplot(x=x_col, y=y_col, hue=hue, data=df, palette=palette, ax=ax)
    else:
        sns.barplot(x=x_col, y=y_col, data=df, palette=palette, ax=ax)

    if rotation:
        ax.tick_params(axis='x', rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def line_plot(df, x_col, y_col, ax=None, figsize=(12, 6), color=None, marker='o', linewidth=2, rotation=None, title=None, show=True):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    sns.lineplot(x=df[x_col], y=df[y_col], data=df, color=color, marker=marker, linewidth=linewidth, ax=ax)
    if rotation:
        ax.tick_params(axis='x', rotation=rotation)
    ax.set_title(title if title else f'{y_col} by {x_col} (Line Plot)')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def box_plot(df, col, hue=None, ax=None, figsize=(8, 6), palette=None, title=None, show=True):
    """
    단일 boxplot 또는 hue로 그룹 비교
    - hue: 그룹 비교할 컬럼명 (예: 'completed')
    - palette: 색상 리스트 (예: ['salmon', 'skyblue'])
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if palette is None:
        palette = ['skyblue', 'salmon']

    if hue:
        sns.boxplot(data=df, x=hue, y=col, palette=palette, ax=ax)
        ax.set_title(title if title else f'{col} by {hue}')
    else:
        sns.boxplot(y=df[col], color=palette[0], ax=ax)
        ax.set_title(title if title else f'{col} (Box Plot)')

    ax.set_ylabel(col)

    if show:
        plt.tight_layout()
        plt.show()
    return ax

def hist_plot(df, col, ax=None, figsize=(10, 6), bins='auto', color='steelblue', kde=False, stat='count', title=None, xlabel=None, ylabel=None, show=True):
    """
    히스토그램 시각화
    - bins: 구간 수 ('auto', 정수, 리스트 등)
    - kde: True면 KDE 곡선 함께 표시
    - stat: 'count', 'density', 'probability', 'frequency' 등
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    sns.histplot(data=df, x=col, bins=bins, color=color, kde=kde, stat=stat, ax=ax)

    ax.set_title(title if title else f'{col} Distribution')
    ax.set_xlabel(xlabel if xlabel else col)
    ax.set_ylabel(ylabel if ylabel else stat.capitalize())

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def kde_plot(df, col, ax=None, figsize=(10, 6), color='steelblue', fill=True, alpha=0.3, linewidth=2, hue=None, palette='Blues', title=None, xlabel=None, ylabel=None, show=True):
    """
    KDE(커널 밀도 추정) 시각화
    - fill: True면 곡선 아래 영역 채움
    - alpha: 채움 투명도 (0~1)
    - hue: 그룹별 비교할 컬럼명
    - palette: hue 사용 시 색상 팔레트
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if hue:
        sns.kdeplot(data=df, x=col, hue=hue, fill=fill, alpha=alpha, linewidth=linewidth, palette=palette, ax=ax)
    else:
        sns.kdeplot(data=df, x=col, color=color, fill=fill, alpha=alpha, linewidth=linewidth, ax=ax)

    ax.set_title(title if title else f'{col} KDE')
    ax.set_xlabel(xlabel if xlabel else col)
    ax.set_ylabel(ylabel if ylabel else 'Density')

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def heatmap_plot(df, ax=None, figsize=(10, 8), cmap='Blues', annot=True, fmt='.2f', linewidths=0.5, cbar=True, mask=None, title=None, xlabel=None, ylabel=None, rotation_x=45, rotation_y=0, show=True):
    """
    히트맵 시각화 (상관행렬 자동 계산)
    - df: DataFrame (수치형 컬럼만 추출하여 상관행렬 계산)
    - cmap: 색상맵 ('Blues', 'Reds', 'YlGnBu', 'coolwarm' 등)
    - annot: True면 셀에 값 표시
    - fmt: annot 포맷 ('.2f'=소수점 2자리 등)
    - linewidths: 셀 간 선 두께
    - cbar: True면 컬러바 표시
    - mask: 'upper' (상삼각 마스크), 'lower' (하삼각 마스크), None (마스크 없음)
    - rotation_x, rotation_y: x축, y축 레이블 회전 각도
    """
    corr = df.select_dtypes(include=['number']).corr()

    # 삼각 마스크 생성
    if mask == 'upper':
        mask_arr = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif mask == 'lower':
        mask_arr = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    else:
        mask_arr = None

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, linewidths=linewidths, cbar=cbar, mask=mask_arr, ax=ax)

    ax.set_title(title if title else 'Heatmap')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if rotation_x:
        ax.tick_params(axis='x', rotation=rotation_x)
    if rotation_y:
        ax.tick_params(axis='y', rotation=rotation_y)

    if show:
        plt.tight_layout()
        plt.show()
    return ax
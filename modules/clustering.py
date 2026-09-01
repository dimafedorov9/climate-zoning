"""
=============================================================================
МОДУЛЬ КЛАСТЕРНОГО АНАЛИЗА (РАЙОНИРОВАНИЕ)
K-means с варьируемым числом кластеров, оценка качества (силуэт),
PCA для снижения размерности и визуализации, иерархическая кластеризация
(Ward) для верификации структуры. Логика перенесена из рабочего скрипта
nso_climate_analysis.py.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def prepare_matrix(df, feature_cols):
    """
    Готовит матрицу признаков к кластеризации: отбрасывает строки с
    пропусками в выбранных признаках, стандартизирует (z-score).
    Возвращает (df_clean, X_scaled, scaler).
    """
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    X = d[feature_cols].apply(pd.to_numeric, errors='coerce').values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return d, Xs, scaler


def silhouette_by_k(Xs, k_range=range(2, 7), random_state=42, n_init=20):
    """Силуэтный коэффициент для каждого k в диапазоне. Возвращает dict {k: score}."""
    scores = {}
    for k in k_range:
        if k >= len(Xs):
            continue
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            scores[k] = np.nan
            continue
        scores[k] = silhouette_score(Xs, labels)
    return scores


def best_k(scores):
    """Выбирает k с максимальным силуэтом (игнорируя NaN)."""
    valid = {k: v for k, v in scores.items() if pd.notna(v)}
    if not valid:
        return None
    return max(valid, key=valid.get)


def ari_by_k(Xs, lat_values, k_range=range(2, 7), random_state=42, n_init=20):
    """
    ARI (Adjusted Rand Index) между K-means кластерами и делением станций
    на k широтных зон (квантили широты) — внешний критерий валидации,
    проверяющий, насколько кластеры согласуются с географическим
    (широтным) градиентом. Требует широту станций (lat_values).
    Возвращает dict {k: ARI}.
    """
    scores = {}
    lat_series = pd.Series(lat_values).reset_index(drop=True)
    if lat_series.isna().all():
        return {k: np.nan for k in k_range}
    for k in k_range:
        if k >= len(Xs):
            continue
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Xs)
        try:
            lat_q = pd.qcut(lat_series, k, labels=False, duplicates='drop')
            scores[k] = adjusted_rand_score(lat_q, labels)
        except Exception:
            scores[k] = np.nan
    return scores


def combined_scores(sil_scores, ari_scores, w_sil=0.6, w_ari=0.4):
    """Комбинированный критерий выбора k: w_sil·силуэт + w_ari·ARI."""
    combined = {}
    for k in sil_scores:
        s = sil_scores.get(k, np.nan)
        a = ari_scores.get(k, np.nan) if ari_scores else np.nan
        if pd.isna(s):
            combined[k] = np.nan
        elif pd.isna(a):
            combined[k] = s
        else:
            combined[k] = w_sil * s + w_ari * a
    return combined


def run_clustering(Xs, k, algorithm='kmeans', Z=None, random_state=42, n_init=20):
    """
    Единая точка запуска разных алгоритмов кластеризации.
    algorithm: 'kmeans', 'hierarchical' (Ward, через fcluster), 'gmm'
    (Gaussian Mixture). Возвращает (labels, silhouette).
    """
    if algorithm == 'kmeans':
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Xs)
    elif algorithm == 'hierarchical':
        if Z is None:
            Z = hierarchical_linkage(Xs)
        labels = fcluster(Z, k, criterion='maxclust')
    elif algorithm == 'gmm':
        gm = GaussianMixture(n_components=k, random_state=random_state,
                              n_init=min(n_init, 10))
        labels = gm.fit_predict(Xs)
    else:
        raise ValueError(f'Неизвестный алгоритм: {algorithm}')

    sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else np.nan
    return labels, sil


def run_kmeans(Xs, k, random_state=42, n_init=20):
    """Запускает K-means с заданным k. Возвращает (labels, model, silhouette)."""
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else np.nan
    return labels, km, sil


def relabel_clusters_by_feature(df, labels, order_feature, ascending=False):
    """
    Перенумеровывает кластеры (1, 2, 3, ...) по убыванию (или возрастанию)
    среднего значения указанного признака — чтобы номера кластеров были
    содержательно интерпретируемы и стабильны между запусками.
    """
    tmp = df.copy()
    tmp['_label_raw'] = labels
    means = tmp.groupby('_label_raw')[order_feature].mean()
    order = means.sort_values(ascending=ascending).index.tolist()
    remap = {old: new for new, old in enumerate(order, start=1)}
    return tmp['_label_raw'].map(remap).values


def pca_transform(Xs):
    """PCA без ограничения компонент. Возвращает (Xpca, pca_model, explained_var_ratio)."""
    pca = PCA()
    Xpca = pca.fit_transform(Xs)
    return Xpca, pca, pca.explained_variance_ratio_


def hierarchical_linkage(Xs, method='ward'):
    """Матрица связей для дендрограммы (иерархическая кластеризация)."""
    return linkage(Xs, method=method)


def cluster_summary_table(df, feature_cols, cluster_col='Кластер'):
    """Средние/std по каждому признаку внутри каждого кластера + численность."""
    rows = []
    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c]
        row = {'Кластер': c, 'N': len(sub)}
        for f in feature_cols:
            row[f'{f} (среднее)'] = sub[f].mean()
            row[f'{f} (ст.откл.)'] = sub[f].std()
        rows.append(row)
    return pd.DataFrame(rows)

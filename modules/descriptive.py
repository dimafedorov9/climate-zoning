"""
=============================================================================
МОДУЛЬ ОПИСАТЕЛЬНОЙ СТАТИСТИКИ
Таблица 1 (целевая переменная, расширенная) и Таблица 2 (все признаки, сводная).
Тесты на нормальность: Шапиро-Уилка, Д'Агостино-Пирсона.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_target(series, name='Переменная'):
    """
    Расширенная описательная статистика одной (целевой) переменной —
    для Таблицы 1 из ТЗ: N, среднее, медиана, ст.откл., min/max, размах,
    CV, асимметрия, эксцесс, квартили, IQR, тесты на нормальность.
    """
    x = pd.to_numeric(series, errors='coerce').dropna().values
    n = len(x)
    if n == 0:
        return pd.DataFrame()

    mean = x.mean()
    std = x.std(ddof=1) if n > 1 else np.nan
    q1, q3 = np.percentile(x, [25, 75])
    cv = (std / mean * 100) if mean != 0 else np.nan

    row = {
        'Показатель': name,
        'N': n,
        'Среднее': mean,
        'Медиана': np.median(x),
        'Ст.откл.': std,
        'Минимум': x.min(),
        'Максимум': x.max(),
        'Размах': x.max() - x.min(),
        'CV, %': cv,
        'Асимметрия': stats.skew(x) if n > 2 else np.nan,
        'Эксцесс': stats.kurtosis(x) if n > 3 else np.nan,
        'Q1 (25%)': q1,
        'Q3 (75%)': q3,
        'IQR': q3 - q1,
    }

    # Тесты на нормальность
    if 3 <= n <= 5000:
        sh_stat, sh_p = stats.shapiro(x)
        row['Шапиро-Уилка W'] = sh_stat
        row['Шапиро-Уилка p'] = sh_p
    else:
        row['Шапиро-Уилка W'] = np.nan
        row['Шапиро-Уилка p'] = np.nan

    if n >= 20:
        try:
            da_stat, da_p = stats.normaltest(x)
            row["Д'Агостино-Пирсона K2"] = da_stat
            row["Д'Агостино-Пирсона p"] = da_p
        except Exception:
            row["Д'Агостино-Пирсона K2"] = np.nan
            row["Д'Агостино-Пирсона p"] = np.nan
    else:
        row["Д'Агостино-Пирсона K2"] = np.nan
        row["Д'Агостино-Пирсона p"] = np.nan

    return pd.DataFrame([row])


def normality_conclusion(row):
    """
    Формирует текстовый вывод о нормальности распределения на основе
    строки из descriptive_target (для авто-интерпретации, Промпт 2).
    """
    p_sh = row.get('Шапиро-Уилка p')
    p_da = row.get("Д'Агостино-Пирсона p")
    p_use = p_sh if pd.notna(p_sh) else p_da
    test_name = 'Шапиро-Уилка' if pd.notna(p_sh) else "Д'Агостино-Пирсона"
    if p_use is None or pd.isna(p_use):
        return 'Недостаточно данных для теста на нормальность.'
    if p_use < 0.05:
        return (f'Тест {test_name} выявил статистически значимое отклонение '
                f'распределения от нормального (p = {p_use:.4f} < 0,05), '
                f'что указывает на целесообразность использования '
                f'непараметрических методов анализа.')
    return (f'Тест {test_name} не выявил статистически значимого отклонения '
            f'распределения от нормального (p = {p_use:.4f} ≥ 0,05).')


def descriptive_all_features(df, columns=None):
    """
    Сводная описательная статистика по всем числовым признакам —
    Таблица 2: среднее, медиана, ст.откл., min, max, размах, CV%.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in columns:
        x = pd.to_numeric(df[col], errors='coerce').dropna().values
        if len(x) == 0:
            continue
        mean = x.mean()
        std = x.std(ddof=1) if len(x) > 1 else np.nan
        cv = (std / mean * 100) if mean != 0 else np.nan
        rows.append({
            'Признак': col,
            'N': len(x),
            'Среднее': mean,
            'Медиана': np.median(x),
            'Ст.откл.': std,
            'Минимум': x.min(),
            'Максимум': x.max(),
            'Размах': x.max() - x.min(),
            'CV, %': cv,
        })
    return pd.DataFrame(rows)


def find_redundant_features(df, columns, corr_threshold=0.95):
    """
    Ищет пары признаков с очень высокой взаимной корреляцией (|r| > порога) —
    кандидаты на исключение из совместного анализа (пересечение по смыслу).
    Возвращает DataFrame пар с коэффициентом корреляции.
    """
    num_df = df[columns].apply(pd.to_numeric, errors='coerce')
    corr = num_df.corr(method='pearson')
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > corr_threshold:
                pairs.append({
                    'Признак 1': cols[i],
                    'Признак 2': cols[j],
                    'r (Пирсон)': r,
                })
    return pd.DataFrame(pairs).sort_values(
        'r (Пирсон)', key=abs, ascending=False) if pairs else pd.DataFrame(
        columns=['Признак 1', 'Признак 2', 'r (Пирсон)'])

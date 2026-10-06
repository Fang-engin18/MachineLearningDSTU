from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA


# ============================================================
# Загрузка данных
# ============================================================

def load_pca_dataset(
    path: Path,
    variant: int = 11
):
    """
    Загружает PCA-датасет для указанного варианта.

    Каждый вариант занимает 6 столбцов:
    a, b, c, d, e, y.
    """

    raw = pd.read_excel(
        path,
        header=None
    )

    BLOCK_SIZE = 6
    MAX_VARIANTS = 24

    if variant < 1 or variant > MAX_VARIANTS:
        raise ValueError(
            f"Номер варианта должен быть от 1 до {MAX_VARIANTS}"
        )

    # Определяем начало нужного блока
    start = (variant - 1) * BLOCK_SIZE
    end = start + BLOCK_SIZE

    # Берём только нужный вариант.
    #
    # Строка 0:
    # "Вариант 1", "Вариант 2", ...
    #
    # Строка 1:
    # a, b, c, d, e, y
    #
    # Строки 2+:
    # реальные данные
    data = raw.iloc[2:, start:end].copy()

    # Задаём нормальные имена столбцов
    data.columns = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "y"
    ]

    # Преобразуем значения в числа
    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Удаляем строки, в которых есть пропуски
    data = data.dropna().reset_index(drop=True)

    # Отдельно признаки и целевую переменную
    X = data[
        ["a", "b", "c", "d", "e"]
    ]

    y = data["y"]

    feature_names = [
        "a",
        "b",
        "c",
        "d",
        "e"
    ]

    print("\n" + "=" * 70)
    print(f"ЗАГРУЗКА PCA — ВАРИАНТ {variant}")
    print("=" * 70)

    print(f"Размер данных: {data.shape}")
    print(f"Признаки: {feature_names}")
    print("Целевая переменная: y")

    print("\nПервые 5 строк:")
    print(data.head())

    return X, y, feature_names
# ============================================================
# 1.1 Корреляционная матрица
# ============================================================

def correlation_analysis(X, output_dir: Path):
    print("\n" + "=" * 70)
    print("1.1 МАТРИЦА КОРРЕЛЯЦИЙ")
    print("=" * 70)

    correlation = X.corr()

    print(correlation.round(4))

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title("Матрица взаимных корреляций признаков")
    plt.tight_layout()

    plt.savefig(
        output_dir / "01_correlation_matrix.png",
        dpi=200
    )

    plt.show()

    correlation.to_excel(
        output_dir / "correlation_matrix.xlsx"
    )

    return correlation


# ============================================================
# 1.2 Scatter matrix
# ============================================================

def scatter_matrix(X, y, feature_names, output_dir: Path):
    print("\n" + "=" * 70)
    print("1.2 МАТРИЦА РАЗБРОСА")
    print("=" * 70)

    # Добавляем y только для визуализации
    plot_data = X.copy()
    plot_data["y"] = y.values

    axes = pd.plotting.scatter_matrix(
        plot_data[feature_names],
        figsize=(12, 12),
        diagonal="hist",
        alpha=0.65
    )

    plt.suptitle(
        "Матрица разброса признаков",
        y=1.02
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "02_scatter_matrix.png",
        dpi=200
    )

    plt.show()


# ============================================================
# Стандартизация
# ============================================================

def standardize_data(X):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ============================================================
# Вспомогательная функция определения компонент
# ============================================================

def classify_components(
    importance,
    method_name,
    threshold=0.95
):
    """
    Для PCA/SVD:

    main      - компоненты до достижения 95% накопленной дисперсии
    secondary - остальные компоненты с заметным вкладом
    noise     - компоненты с вкладом < 1%

    Для ICA importance трактуется отдельно.
    """

    importance = np.asarray(importance)

    cumulative = np.cumsum(importance)

    main_count = np.searchsorted(
        cumulative,
        threshold
    ) + 1

    main_count = min(main_count, len(importance))

    main = list(range(1, main_count + 1))

    secondary = []
    noise = []

    for i, value in enumerate(importance):
        component = i + 1

        if component in main:
            continue

        if value < 0.01:
            noise.append(component)
        else:
            secondary.append(component)

    print(f"\n{method_name}:")
    print(f"Главные компоненты: {main}")
    print(f"Второстепенные компоненты: {secondary}")
    print(f"Шумовые компоненты: {noise}")

    return main, secondary, noise


# ============================================================
# 1.3 PCA
# ============================================================

def pca_analysis(X_scaled, feature_names, y, output_dir):
    print("\n" + "=" * 70)
    print("1.3 PCA — АНАЛИЗ ГЛАВНЫХ КОМПОНЕНТ")
    print("=" * 70)

    pca = PCA()

    scores = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_

    print("\nExplained variance ratio:")

    for i, value in enumerate(explained):
        print(
            f"PC{i + 1}: "
            f"{value * 100:.2f}%"
        )

    print(
        f"\nНакопленная дисперсия: "
        f"{np.cumsum(explained) * 100}"
    )

    main, secondary, noise = classify_components(
        explained,
        "PCA"
    )

    # --------------------------------------------------------
    # График объяснённой дисперсии
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    components = np.arange(1, len(explained) + 1)

    plt.bar(
        components,
        explained * 100,
        alpha=0.7
    )

    plt.plot(
        components,
        np.cumsum(explained) * 100,
        marker="o",
        linewidth=2
    )

    plt.axhline(
        95,
        linestyle="--",
        label="95%"
    )

    plt.xlabel("Компонента")
    plt.ylabel("Объяснённая дисперсия, %")
    plt.title("PCA — значимость компонент")
    plt.xticks(components)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "03_pca_explained_variance.png",
        dpi=200
    )

    plt.show()

    # --------------------------------------------------------
    # Loadings
    # --------------------------------------------------------

    loadings = pca.components_.T

    print("\nМатрица loadings PCA:")

    loading_df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=[
            f"PC{i + 1}"
            for i in range(loadings.shape[1])
        ]
    )

    print(loading_df.round(4))

    loading_df.to_excel(
        output_dir / "pca_loadings.xlsx"
    )

    # --------------------------------------------------------
    # PCA 2D
    # --------------------------------------------------------

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        scores[:, 0],
        scores[:, 1],
        c=y,
        cmap="viridis",
        alpha=0.75,
        edgecolors="k"
    )

    plt.xlabel(
        f"PC1 ({explained[0] * 100:.1f}%)"
    )

    plt.ylabel(
        f"PC2 ({explained[1] * 100:.1f}%)"
    )

    plt.title("PCA — проекция на первые две компоненты")
    plt.grid(True, alpha=0.3)

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        sorted(pd.unique(y)),
        title="Классы"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "04_pca_projection.png",
        dpi=200
    )

    plt.show()

    return pca, scores, loadings


# ============================================================
# PCA Biplot
# ============================================================

def biplot(
    scores,
    loadings,
    feature_names,
    y,
    x_label,
    y_label,
    title,
    output_path
):
    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # Масштабирование score
    scalex = 1.0 / (
        scores[:, 0].max() -
        scores[:, 0].min()
    )

    scaley = 1.0 / (
        scores[:, 1].max() -
        scores[:, 1].min()
    )

    scatter = ax.scatter(
        scores[:, 0] * scalex,
        scores[:, 1] * scaley,
        c=y,
        cmap="viridis",
        alpha=0.7,
        edgecolors="k"
    )

    handles, _ = scatter.legend_elements()

    ax.legend(
        handles,
        sorted(pd.unique(y)),
        title="Классы"
    )

    # Векторы признаков
    for i in range(loadings.shape[0]):

        ax.arrow(
            0,
            0,
            loadings[i, 0],
            loadings[i, 1],
            color="red",
            alpha=0.7,
            head_width=0.03,
            head_length=0.03,
            linewidth=1.5,
            length_includes_head=True
        )

        ax.text(
            loadings[i, 0] * 1.15,
            loadings[i, 1] * 1.15,
            feature_names[i],
            color="darkred",
            ha="center",
            va="center",
            fontweight="bold"
        )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=0.5
    )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=0.5
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(title)

    ax.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.show()


# ============================================================
# 1.3 SVD
# ============================================================

def svd_analysis(
    X_scaled,
    feature_names,
    y,
    output_dir
):
    print("\n" + "=" * 70)
    print("1.3 SVD — СИНГУЛЯРНОЕ РАЗЛОЖЕНИЕ")
    print("=" * 70)

    # SVD:
    # X = U * S * Vt

    U, S, Vt = np.linalg.svd(
        X_scaled,
        full_matrices=False
    )

    # Сингулярные значения
    explained = S ** 2
    explained = explained / explained.sum()

    scores = U * S

    loadings = Vt.T

    print("\nСингулярные значения:")

    for i, value in enumerate(S):
        print(
            f"S{i + 1}: {value:.6f}"
        )

    print("\nДоля дисперсии:")

    for i, value in enumerate(explained):
        print(
            f"Component {i + 1}: "
            f"{value * 100:.2f}%"
        )

    main, secondary, noise = classify_components(
        explained,
        "SVD"
    )

    # --------------------------------------------------------
    # График
    # --------------------------------------------------------

    components = np.arange(
        1,
        len(S) + 1
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        components,
        explained * 100,
        alpha=0.7
    )

    plt.plot(
        components,
        np.cumsum(explained) * 100,
        marker="o"
    )

    plt.axhline(
        95,
        linestyle="--",
        label="95%"
    )

    plt.xlabel("Компонента")
    plt.ylabel("Доля дисперсии, %")
    plt.title("SVD — значимость компонент")
    plt.xticks(components)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "05_svd_significance.png",
        dpi=200
    )

    plt.show()

    # --------------------------------------------------------
    # SVD projection
    # --------------------------------------------------------

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        scores[:, 0],
        scores[:, 1],
        c=y,
        cmap="viridis",
        alpha=0.75,
        edgecolors="k"
    )

    plt.xlabel(
        f"SVD1 ({explained[0] * 100:.1f}%)"
    )

    plt.ylabel(
        f"SVD2 ({explained[1] * 100:.1f}%)"
    )

    plt.title(
        "SVD — проекция на первые две компоненты"
    )

    plt.grid(True, alpha=0.3)

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        sorted(pd.unique(y)),
        title="Классы"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "06_svd_projection.png",
        dpi=200
    )

    plt.show()

    # Biplot
    biplot(
        scores,
        loadings,
        feature_names,
        y,
        f"SVD1 ({explained[0] * 100:.1f}%)",
        f"SVD2 ({explained[1] * 100:.1f}%)",
        "SVD Biplot",
        output_dir / "07_svd_biplot.png"
    )

    return U, S, Vt, scores, loadings


# ============================================================
# 1.3 ICA
# ============================================================

def ica_analysis(
    X_scaled,
    feature_names,
    y,
    output_dir
):
    print("\n" + "=" * 70)
    print("1.3 ICA — АНАЛИЗ НЕЗАВИСИМЫХ КОМПОНЕНТ")
    print("=" * 70)

    n_components = X_scaled.shape[1]

    ica = FastICA(
        n_components=n_components,
        random_state=42,
        max_iter=2000,
        whiten="unit-variance"
    )

    scores = ica.fit_transform(
        X_scaled
    )

    # Для ICA нет explained_variance_ratio_,
    # поэтому оцениваем компоненты по абсолютной
    # избыточной куртозисности.
    kurt = kurtosis(
        scores,
        axis=0,
        fisher=True,
        bias=False
    )

    importance = np.abs(kurt)

    if importance.sum() != 0:
        importance_normalized = (
            importance /
            importance.sum()
        )
    else:
        importance_normalized = np.ones(
            n_components
        ) / n_components

    print("\nКуртозис компонент:")

    for i in range(n_components):
        print(
            f"IC{i + 1}: "
            f"kurtosis={kurt[i]:.4f}, "
            f"importance={importance_normalized[i] * 100:.2f}%"
        )

    # Сортируем компоненты по степени негауссовости
    order = np.argsort(
        importance
    )[::-1]

    print("\nРейтинг ICA-компонент:")

    for rank, index in enumerate(order):
        print(
            f"{rank + 1}. IC{index + 1} "
            f"| importance="
            f"{importance_normalized[index] * 100:.2f}%"
        )

    # Первые две считаем наиболее значимыми
    main = [
        int(order[0] + 1)
    ]

    if len(order) > 1:
        main.append(
            int(order[1] + 1)
        )

    # Слабые компоненты
    noise = [
        int(i + 1)
        for i, value in enumerate(
            importance_normalized
        )
        if value < 0.05
    ]

    secondary = [
        int(i + 1)
        for i in range(n_components)
        if i + 1 not in main
        and i + 1 not in noise
    ]

    print("\nICA:")
    print(f"Главные компоненты: {main}")
    print(f"Второстепенные компоненты: {secondary}")
    print(f"Шумовые компоненты: {noise}")

    # --------------------------------------------------------
    # График
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    components = np.arange(
        1,
        n_components + 1
    )

    plt.bar(
        components,
        importance_normalized * 100
    )

    plt.xlabel("Компонента")
    plt.ylabel(
        "Нормированная |куртозис|, %"
    )

    plt.title(
        "ICA — значимость независимых компонент"
    )

    plt.xticks(components)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "08_ica_significance.png",
        dpi=200
    )

    plt.show()

    # --------------------------------------------------------
    # ICA projection
    # --------------------------------------------------------

    # Используем две наиболее значимые компоненты
    c1 = order[0]
    c2 = order[1]

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        scores[:, c1],
        scores[:, c2],
        c=y,
        cmap="viridis",
        alpha=0.75,
        edgecolors="k"
    )

    plt.xlabel(
        f"IC{c1 + 1}"
    )

    plt.ylabel(
        f"IC{c2 + 1}"
    )

    plt.title(
        "ICA — проекция на наиболее значимые компоненты"
    )

    plt.grid(True, alpha=0.3)

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        sorted(pd.unique(y)),
        title="Классы"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "09_ica_projection.png",
        dpi=200
    )

    plt.show()

    # --------------------------------------------------------
    # ICA loadings
    # --------------------------------------------------------

    # components_ имеет размер:
    # n_components x n_features
    #
    # Для отображения направления признаков
    # транспонируем.

    loadings = ica.components_.T

    selected_loadings = loadings[
        :,
        [c1, c2]
    ]

    selected_scores = scores[
        :,
        [c1, c2]
    ]

    biplot(
        selected_scores,
        selected_loadings,
        feature_names,
        y,
        f"IC{c1 + 1}",
        f"IC{c2 + 1}",
        "ICA Biplot",
        output_dir / "10_ica_biplot.png"
    )

    return (
        ica,
        scores,
        loadings,
        kurt
    )


# ============================================================
# Главная функция раздела PCA/SVD/ICA
# ============================================================

def run_dimensionality_analysis(
    dataset_path: Path,
    output_dir: Path,
    variant: int = 11
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y, feature_names = load_pca_dataset(
        dataset_path,
        variant
    )

    correlation_analysis(
        X,
        output_dir
    )

    scatter_matrix(
        X,
        y,
        feature_names,
        output_dir
    )

    X_scaled, scaler = standardize_data(
        X
    )

    pca, pca_scores, pca_loadings = pca_analysis(
        X_scaled,
        feature_names,
        y,
        output_dir
    )

    svd_result = svd_analysis(
        X_scaled,
        feature_names,
        y,
        output_dir
    )

    ica_result = ica_analysis(
        X_scaled,
        feature_names,
        y,
        output_dir
    )

    print("\n" + "=" * 70)
    print(f"РАЗДЕЛ 1 ЗАВЕРШЁН — ВАРИАНТ {variant}")
    print("=" * 70)

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "pca": pca,
        "pca_scores": pca_scores,
        "pca_loadings": pca_loadings,
        "svd": svd_result,
        "ica": ica_result
    }
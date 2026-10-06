from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch,
    BisectingKMeans
)

from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score
)

from scipy.cluster.hierarchy import linkage


# ============================================================
# HDBSCAN
# ============================================================

def create_hdbscan():
    """
    Сначала пытаемся использовать HDBSCAN из sklearn.
    Если его нет — используем отдельную библиотеку hdbscan.
    """

    try:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(
            min_cluster_size=10,
            min_samples=5
        )

    except ImportError:
        try:
            import hdbscan

            return hdbscan.HDBSCAN(
                min_cluster_size=10,
                min_samples=5
            )

        except ImportError:
            raise ImportError(
                "HDBSCAN недоступен. "
                "Установите: pip install hdbscan"
            )


# ============================================================
# Загрузка
# ============================================================

def load_clustering_dataset(
    path: Path,
    variant: int = 11
):
    """
    Загружает датасет кластеризации
    для указанного варианта.

    Каждый вариант занимает 3 столбца:
    a, b, y.
    """

    raw = pd.read_excel(
        path,
        header=None
    )

    BLOCK_SIZE = 3
    MAX_VARIANTS = 24

    if variant < 1 or variant > MAX_VARIANTS:
        raise ValueError(
            f"Номер варианта должен быть от 1 до {MAX_VARIANTS}"
        )

    # Начало блока нужного варианта
    start = (variant - 1) * BLOCK_SIZE
    end = start + BLOCK_SIZE

    # Структура Excel:
    #
    # строка 0:
    # Вариант 1 | ... | Вариант 2 | ...
    #
    # строка 1:
    # a | b | y | a | b | y | ...
    #
    # строки 2+:
    # реальные данные

    data = raw.iloc[2:, start:end].copy()

    # Называем столбцы
    data.columns = [
        "a",
        "b",
        "y"
    ]

    # Преобразуем в числа
    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Удаляем пустые строки
    data = data.dropna().reset_index(drop=True)

    # Признаки
    X = data[
        ["a", "b"]
    ]

    # Истинные классы.
    # Используются только для визуальной проверки
    # и сравнения результатов кластеризации.
    y = data["y"]

    feature_names = [
        "a",
        "b"
    ]

    print("\n" + "=" * 70)
    print(f"ЗАГРУЗКА КЛАСТЕРИЗАЦИИ — ВАРИАНТ {variant}")
    print("=" * 70)

    print(f"Размер данных: {data.shape}")
    print(f"Признаки: {feature_names}")
    print("Целевая переменная: y")

    print("\nПервые 5 строк:")
    print(data.head())

    return X, y, feature_names


# ============================================================
# Визуализация
# ============================================================

def plot_clusters(
    ax,
    X,
    labels,
    title,
    y_true=None
):
    """
    Рисует результат кластеризации.
    label=-1 считается выбросом.
    """

    labels = np.asarray(labels)

    unique_labels = np.unique(labels)

    for label in unique_labels:

        mask = labels == label

        if label == -1:
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                marker="x",
                s=70,
                label="Outliers"
            )

        else:
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                alpha=0.75,
                edgecolors="k",
                label=f"Cluster {label}"
            )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    ax.grid(
        True,
        alpha=0.3
    )


# ============================================================
# Метрики
# ============================================================

def calculate_metrics(
    X,
    labels,
    y_true
):
    labels = np.asarray(labels)

    # Убираем выбросы при расчёте silhouette
    valid = labels != -1

    unique = np.unique(
        labels[valid]
    )

    if (
        len(unique) >= 2
        and valid.sum() > len(unique)
    ):
        try:
            silhouette = silhouette_score(
                X[valid],
                labels[valid]
            )
        except Exception:
            silhouette = np.nan
    else:
        silhouette = np.nan

    # ARI используется ТОЛЬКО для сравнения
    # с эталонными метками после кластеризации
    try:
        ari = adjusted_rand_score(
            y_true,
            labels
        )
    except Exception:
        ari = np.nan

    n_clusters = len(
        set(labels) - {-1}
    )

    n_outliers = np.sum(
        labels == -1
    )

    return (
        n_clusters,
        n_outliers,
        silhouette,
        ari
    )


# ============================================================
# Создание всех алгоритмов
# ============================================================

def create_algorithms():

    algorithms = {}

    algorithms["K-Means"] = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=20
    )

    algorithms["Affinity Propagation"] = (
        AffinityPropagation(
            random_state=42
        )
    )

    algorithms["Mean-Shift"] = MeanShift(
        bin_seeding=True
    )

    algorithms["Spectral Clustering"] = (
        SpectralClustering(
            n_clusters=3,
            affinity="nearest_neighbors",
            random_state=42
        )
    )

    algorithms["Ward"] = (
        AgglomerativeClustering(
            n_clusters=3,
            linkage="ward"
        )
    )

    algorithms["Agglomerative"] = (
        AgglomerativeClustering(
            n_clusters=3,
            linkage="average"
        )
    )

    algorithms["DBSCAN"] = DBSCAN(
        eps=0.25,
        min_samples=5
    )

    algorithms["HDBSCAN"] = create_hdbscan()

    algorithms["OPTICS"] = OPTICS(
        min_samples=5,
        xi=0.05,
        min_cluster_size=0.05
    )

    algorithms["Gaussian Mixtures"] = (
        GaussianMixture(
            n_components=3,
            covariance_type="full",
            random_state=42
        )
    )

    algorithms["BIRCH"] = Birch(
        n_clusters=3,
        threshold=0.5
    )

    algorithms["Bisecting K-Means"] = (
        BisectingKMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )
    )

    return algorithms


# ============================================================
# Запуск алгоритма
# ============================================================

def fit_algorithm(
    name,
    algorithm,
    X
):

    # GaussianMixture использует predict()
    if name == "Gaussian Mixtures":

        algorithm.fit(X)

        labels = algorithm.predict(X)

    else:

        algorithm.fit(X)

        if hasattr(
            algorithm,
            "labels_"
        ):
            labels = algorithm.labels_

        else:
            labels = algorithm.predict(X)

    return np.asarray(labels)


# ============================================================
# 2.2 Все графики
# ============================================================

def run_all_algorithms(
    X,
    y,
    output_dir
):

    algorithms = create_algorithms()

    results = {}

    print("\n" + "=" * 70)
    print("КЛАСТЕРИЗАЦИЯ — 12 АЛГОРИТМОВ")
    print("=" * 70)

    # --------------------------------------------------------
    # Один общий график
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18, 13)
    )

    axes = axes.ravel()

    for index, (
        name,
        algorithm
    ) in enumerate(
        algorithms.items()
    ):

        ax = axes[index]

        try:

            labels = fit_algorithm(
                name,
                algorithm,
                X
            )

            metrics = calculate_metrics(
                X,
                labels,
                y
            )

            n_clusters, n_outliers, silhouette, ari = metrics

            results[name] = {
                "algorithm": algorithm,
                "labels": labels,
                "n_clusters": n_clusters,
                "n_outliers": n_outliers,
                "silhouette": silhouette,
                "ari": ari,
                "status": "OK"
            }

            plot_clusters(
                ax,
                X,
                labels,
                name,
                y
            )

            ax.set_title(
                f"{name}\n"
                f"clusters={n_clusters}, "
                f"outliers={n_outliers}\n"
                f"silhouette={silhouette:.3f}"
                if not np.isnan(silhouette)
                else
                f"{name}\n"
                f"clusters={n_clusters}, "
                f"outliers={n_outliers}"
            )

            print(
                f"\n{name}:"
            )

            print(
                f"  Кластеров: {n_clusters}"
            )

            print(
                f"  Выбросов: {n_outliers}"
            )

            print(
                f"  Silhouette: "
                f"{silhouette:.4f}"
                if not np.isnan(silhouette)
                else
                "  Silhouette: N/A"
            )

            print(
                f"  ARI: {ari:.4f}"
            )

        except Exception as error:

            results[name] = {
                "algorithm": algorithm,
                "labels": None,
                "status": "FAILED",
                "error": str(error)
            }

            ax.text(
                0.5,
                0.5,
                "FAILED\n\n"
                + str(error),
                ha="center",
                va="center",
                wrap=True
            )

            ax.set_title(
                f"{name} — FAILED"
            )

            print(
                f"\n{name}: ОШИБКА"
            )

            print(
                str(error)
            )

    plt.suptitle(
        "Сравнение алгоритмов кластеризации",
        fontsize=18
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "11_all_clustering_algorithms.png",
        dpi=200
    )

    plt.show()

    return results


# ============================================================
# Таблица результатов
# ============================================================

def create_results_table(
    results,
    output_dir
):

    rows = []

    for name, result in results.items():

        if result["status"] == "OK":

            rows.append({
                "Algorithm": name,
                "Status": "OK",
                "Clusters": result[
                    "n_clusters"
                ],
                "Outliers": result[
                    "n_outliers"
                ],
                "Silhouette": result[
                    "silhouette"
                ],
                "ARI": result[
                    "ari"
                ]
            })

        else:

            rows.append({
                "Algorithm": name,
                "Status": "FAILED",
                "Clusters": np.nan,
                "Outliers": np.nan,
                "Silhouette": np.nan,
                "ARI": np.nan
            })

    table = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 70)

    print(
        table.to_string(
            index=False
        )
    )

    table.to_excel(
        output_dir /
        "clustering_results.xlsx",
        index=False
    )

    return table


# ============================================================
# 2.3 Подбор параметров
# ============================================================

def tune_dbscan(
    X,
    y,
    output_dir
):

    print("\n" + "=" * 70)
    print("ПОДБОР ПАРАМЕТРОВ DBSCAN")
    print("=" * 70)

    eps_values = np.arange(
        0.10,
        1.01,
        0.05
    )

    min_samples_values = [
        3,
        5,
        7,
        10
    ]

    best = None

    for eps in eps_values:

        for min_samples in min_samples_values:

            model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )

            labels = model.fit_predict(X)

            n_clusters = len(
                set(labels) - {-1}
            )

            if n_clusters < 2:
                continue

            valid = labels != -1

            if valid.sum() <= n_clusters:
                continue

            try:

                silhouette = silhouette_score(
                    X[valid],
                    labels[valid]
                )

            except Exception:
                continue

            if (
                best is None
                or silhouette >
                best["silhouette"]
            ):

                best = {
                    "eps": eps,
                    "min_samples": min_samples,
                    "silhouette": silhouette,
                    "labels": labels
                }

    if best is None:

        print(
            "Не удалось подобрать параметры DBSCAN."
        )

        return None

    print(
        f"Лучший eps = {best['eps']:.2f}"
    )

    print(
        f"Лучший min_samples = "
        f"{best['min_samples']}"
    )

    print(
        f"Silhouette = "
        f"{best['silhouette']:.4f}"
    )

    labels = best["labels"]

    plt.figure(figsize=(9, 7))

    unique = np.unique(labels)

    for label in unique:

        mask = labels == label

        if label == -1:

            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                marker="x",
                s=80,
                label="Outliers"
            )

        else:

            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                alpha=0.75,
                edgecolors="k",
                label=f"Cluster {label}"
            )

    plt.title(
        f"DBSCAN после подбора параметров\n"
        f"eps={best['eps']:.2f}, "
        f"min_samples={best['min_samples']}"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "12_dbscan_tuned.png",
        dpi=200
    )

    plt.show()

    return best


# ============================================================
# Поиск лучшего алгоритма
# ============================================================

def find_best_algorithm(
    results
):

    valid_results = []

    for name, result in results.items():

        if (
            result["status"] == "OK"
            and not np.isnan(
                result["silhouette"]
            )
        ):

            valid_results.append(
                (
                    name,
                    result["silhouette"],
                    result["ari"]
                )
            )

    if not valid_results:

        return None

    valid_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("ЛУЧШИЕ АЛГОРИТМЫ")
    print("=" * 70)

    for rank, item in enumerate(
        valid_results,
        start=1
    ):

        name, silhouette, ari = item

        print(
            f"{rank}. {name}: "
            f"Silhouette={silhouette:.4f}, "
            f"ARI={ari:.4f}"
        )

    return valid_results[0][0]


# ============================================================
# Финальный график лучшего метода
# ============================================================

def plot_best_result(
    X,
    y,
    results,
    best_name,
    output_dir
):

    if best_name is None:
        return

    labels = results[
        best_name
    ]["labels"]

    plt.figure(figsize=(9, 7))

    unique_labels = np.unique(
        labels
    )

    for label in unique_labels:

        mask = labels == label

        if label == -1:

            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                marker="x",
                s=100,
                label="Аномалии"
            )

        else:

            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                alpha=0.8,
                edgecolors="k",
                label=f"Кластер {label}"
            )

    plt.title(
        f"Наиболее эффективный алгоритм: "
        f"{best_name}"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "13_best_clustering_result.png",
        dpi=200
    )

    plt.show()


# ============================================================
# Главная функция раздела 2
# ============================================================

def run_clustering_analysis(
    dataset_path: Path,
    output_dir: Path,
    variant: int = 11
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y, feature_names = load_clustering_dataset(
    dataset_path,
    variant
    )

    # Стандартизация
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # Все алгоритмы
    results = run_all_algorithms(
        X_scaled,
        y,
        output_dir
    )

    # Таблица
    table = create_results_table(
        results,
        output_dir
    )

    # DBSCAN tuning
    dbscan_best = tune_dbscan(
        X_scaled,
        y,
        output_dir
    )

    # Лучший алгоритм
    best_name = find_best_algorithm(
        results
    )

    plot_best_result(
        X_scaled,
        y,
        results,
        best_name,
        output_dir
    )

    print("\n" + "=" * 70)
    print("РАЗДЕЛ 2 ЗАВЕРШЁН")
    print("=" * 70)

    print(
        f"\nНаиболее эффективный алгоритм "
        f"по Silhouette: {best_name}"
    )

    return {
        "X": X,
        "X_scaled": X_scaled,
        "y": y,
        "feature_names": feature_names,
        "results": results,
        "table": table,
        "dbscan_best": dbscan_best,
        "best_algorithm": best_name
    }
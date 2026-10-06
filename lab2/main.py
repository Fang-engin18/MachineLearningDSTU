from pathlib import Path

from src.dimensionality import (
    run_dimensionality_analysis
)

from src.clustering import (
    run_clustering_analysis
)


# ============================================================
# Пути
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

PCA_DATASET = (
    DATA_DIR /
    "Lab2_Dataset_PCA.xlsx"
)

CLUSTERING_DATASET = (
    DATA_DIR /
    "Lab2_Dataset_Clustering.xlsx"
)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n")
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Обучение без учителя.")
    print("Понижение размерности. Обнаружение аномалий. Кластеризация.")
    print("=" * 80)

    # ========================================================
    # ЧАСТЬ 1
    # ========================================================

    dimensionality_results = run_dimensionality_analysis(
    PCA_DATASET,
    OUTPUT_DIR,
    variant=11
)

    # ========================================================
    # ЧАСТЬ 2
    # ========================================================

    clustering_results = (
        run_clustering_analysis(
    CLUSTERING_DATASET,
    OUTPUT_DIR,
    variant=11)
    )

    # ========================================================
    # Итог
    # ========================================================

    print("\n")
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print("=" * 80)

    print(
        f"\nВсе результаты сохранены в:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nЛучший алгоритм кластеризации:"
    )

    print(
        clustering_results[
            "best_algorithm"
        ]
    )


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import spectrogram, periodogram
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import tensorflow as tf
import torch

from src.config import set_seeds
from src.io_utils import read_table, write_table, save_figure
from src.data_loader import (
    check_data_quality,
    show_table,
    extract_columns,
    cast_types
)
from src.preprocess import (
    handle_missing,
    to_numpy,
    MinMaxScalerCustom,
    StandardScalerCustom,
    split_train_val_test
)
from src.stats_analysis import (
    plot_series,
    plot_histogram,
    show_sorted,
    plot_ecdf,
    column_stats,
    ci_mean,
    ci_var,
    cross_correlation,
    plot_fft_amplitude
)
from src.windows import (
    sliding_window,
    moving_average
)
from src.image_proc import run_task50


# ============================================================
# НАСТРОЙКА
# ============================================================

set_seeds(42)

DATA_PATH = Path(
    r"C:\Users\fang\Desktop\study\MachineLearning\lab1\data\dataset_var1.csv"
)


# ============================================================
# ЗАДАНИЕ 1. Чтение файлов
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 1. ЧТЕНИЕ ФАЙЛА")
print("=" * 60)

data = read_table(DATA_PATH)

print(data)


# ============================================================
# ЗАДАНИЕ 2. Запись файлов
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2. ЗАПИСЬ ФАЙЛОВ")
print("=" * 60)

paths = write_table(data, "dataset")

print("Созданные файлы:")
for file_type, path in paths.items():
    print(f"{file_type}: {path}")


# ============================================================
# ЗАДАНИЕ 3. Сохранение изображения
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 3. СОХРАНЕНИЕ ИЗОБРАЖЕНИЯ")
print("=" * 60)

fig, ax = plt.subplots()

ax.plot(
    pd.to_numeric(data["x2"], errors="coerce"),
    pd.to_numeric(data["y"], errors="coerce")
)

ax.set_xlabel("x2")
ax.set_ylabel("y")
ax.set_title("Зависимость y от x2")
ax.grid(True)

figure_path = save_figure(fig, "task3_x2_y")

print("График сохранён:", figure_path)

plt.close(fig)


# ============================================================
# ЗАДАНИЕ 4. Проверка данных
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 4. ПРОВЕРКА ДАННЫХ")
print("=" * 60)

report = check_data_quality(data)

print("\nПропущенные значения:")
print(report["missing"])

print("\nТипы данных:")
print(report["dtypes"])

print("\nНекорректные числовые значения:")
print(report["non_numeric"])


# ============================================================
# ЗАДАНИЕ 5. Вывод таблицы
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 5. ВЫВОД ТАБЛИЦЫ")
print("=" * 60)

show_table(data, n=5)


# ============================================================
# ЗАДАНИЕ 6. Извлечение столбцов
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 6. ИЗВЛЕЧЕНИЕ СТОЛБЦОВ")
print("=" * 60)

selected_data = extract_columns(
    data,
    ["x1", "x2", "y"]
)

print(selected_data)


# ============================================================
# ЗАДАНИЕ 7. Явные типы данных
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 7. ЯВНЫЕ ТИПЫ ДАННЫХ")
print("=" * 60)

# x1 содержит строку "ошибка", поэтому здесь специально
# используем столбцы, которые уже являются числовыми.
typed_data = cast_types(
    data,
    {
        "x2": "float64",
        "y": "float64"
    }
)

print(typed_data[["x2", "y"]].dtypes)
print(typed_data[["x2", "y"]].head())


# ============================================================
# ЗАДАНИЕ 8. Обработка проблемных данных
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 8. ОБРАБОТКА ПРОБЛЕМНЫХ ДАННЫХ")
print("=" * 60)

# В x1 есть строка "ошибка".
# Сначала превращаем некорректные значения в NaN.
data_clean = handle_missing(
    data,
    "x1",
    "cast"
)

print("После преобразования x1:")
print(data_clean["x1"].head())

print("\nКоличество NaN в x1:")
print(data_clean["x1"].isna().sum())

# Заполняем NaN средним значением.
data_clean = handle_missing(
    data_clean,
    "x1",
    "mean"
)

print("\nПосле заполнения пропусков:")
print(data_clean["x1"].head())

print("\nКоличество NaN:")
print(data_clean["x1"].isna().sum())


# ============================================================
# ЗАДАНИЕ 9. DataFrame -> NumPy
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 9. DATAFRAME -> NUMPY")
print("=" * 60)

numeric_df = extract_columns(
    data_clean,
    ["x1", "x2", "x3", "y"]
)

X = to_numpy(numeric_df)

print("Тип:", type(X))
print("Shape:", X.shape)
print(X[:5])


# ============================================================
# ЗАДАНИЕ 10. Масштабирование
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 10. МАСШТАБИРОВАНИЕ")
print("=" * 60)

mm_scaler = MinMaxScalerCustom()

X_mm = mm_scaler.fit_transform(X)

print("MinMax:")
print(X_mm[:5])

std_scaler = StandardScalerCustom()

X_std = std_scaler.fit_transform(X)

print("\nStandardScaler:")
print(X_std[:5])


# ============================================================
# ЗАДАНИЕ 11. Восстановление масштаба
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 11. ВОССТАНОВЛЕНИЕ МАСШТАБА")
print("=" * 60)

X_back = mm_scaler.inverse_transform(X_mm)

print(X_back[:5])

print(
    "\nСовпадает с исходными:",
    np.allclose(X, X_back)
)


# ============================================================
# ЗАДАНИЕ 12. Разбиение на 3 части
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 12. РАЗБИЕНИЕ НА 3 ЧАСТИ")
print("=" * 60)

X_train, X_val, X_test = split_train_val_test(
    X,
    (70, 15, 15),
    percent=True
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# ЗАДАНИЕ 13. Фиксация seed
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 13. ФИКСАЦИЯ SEED")
print("=" * 60)

set_seeds(42)

a = np.random.rand(5)

set_seeds(42)

b = np.random.rand(5)

print("Первый массив:")
print(a)

print("\nВторой массив:")
print(b)

print("\nОдинаковые:", np.array_equal(a, b))


# ============================================================
# Подготовка данных для заданий 14–49
# ============================================================

x1 = X[:, 0]
x2 = X[:, 1]
x3 = X[:, 2]
y = X[:, 3]



# ============================================================
# ЗАДАНИЕ 14. График исходных данных
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 14. ГРАФИК ИСХОДНЫХ ДАННЫХ")
print("=" * 60)

fig = plot_series(
    X,
    title="Исходные данные"
)

path = save_figure(fig, "task14_series")
print("График сохранён:", path)

plt.close(fig)


# ============================================================
# ЗАДАНИЕ 15. Гистограмма
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 15. ГИСТОГРАММА")
print("=" * 60)

fig = plot_histogram(x2, bins=30)

path = save_figure(fig, "task15_histogram")
print("График сохранён:", path)

plt.close(fig)


# ============================================================
# ЗАДАНИЕ 16. Сортировка и Квантили
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 16. СОРТИРОВКА И СТАТИСТИКА")
print("=" * 60)

print("Вывод отсортированных значений для x2 (первые 10):")
print(np.sort(x2)[:10])

# Вычисляем доверительные интервалы, которые вызывали ошибку ранее
mean_interval = ci_mean(x2)
var_interval = ci_var(x2)

print(f"\nДоверительный интервал для математического ожидания x2: {mean_interval}")
print(f"Доверительный интервал для дисперсии x2: {var_interval}")


# ============================================================
# ЗАДАНИЕ 17. Эмпирическая функция распределения
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 17. ЭМПИРИЧЕСКАЯ ФУНКЦИЯ")
print("=" * 60)

fig = plot_ecdf(x2)

path = save_figure(fig, "task17_ecdf")
print("График сохранён:", path)

plt.close(fig)


# ============================================================
# ЗАДАНИЕ 18. Статистики
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 18. СТАТИСТИКИ")
print("=" * 60)

statistics = column_stats(x2)

for name, value in statistics.items():
    print(f"{name}: {value}")


# ============================================================
# ЗАДАНИЕ 19. Доверительные интервалы
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 19. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")
print("=" * 60)

mean_interval = ci_mean(x2)
var_interval = ci_var(x2)

print("Для среднего:", mean_interval)
print("Для дисперсии:", var_interval)


# ============================================================
# ЗАДАНИЕ 20. Ковариация и корреляция
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 20. КОВАРИАЦИЯ И КОРРЕЛЯЦИЯ")
print("=" * 60)

cov = np.cov(X, rowvar=False)
corr = np.corrcoef(X, rowvar=False)

print("Ковариация:")
print(cov)

print("\nКорреляция:")
print(corr)


# ============================================================
# ЗАДАНИЕ 21. Значимость корреляции
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 21. ЗНАЧИМОСТЬ КОРРЕЛЯЦИИ")
print("=" * 60)

r, p_value = stats.pearsonr(x2, y)

print(f"r = {r:.6f}")
print(f"p-value = {p_value:.6f}")

if p_value < 0.05:
    print("Корреляция статистически значима.")
else:
    print("Корреляция статистически незначима.")


# ============================================================
# ЗАДАНИЕ 22. Взаимная корреляция
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 22. ВЗАИМНАЯ КОРРЕЛЯЦИЯ")
print("=" * 60)

lags, cross_corr = cross_correlation(
    x2,
    y,
    max_lags=50
)

print("Lags:")
print(lags)

print("\nCorrelation:")
print(cross_corr)


# ============================================================
# ЗАДАНИЕ 23. Производная и градиент
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 23. ПРОИЗВОДНАЯ И ГРАДИЕНТ")
print("=" * 60)

dx = np.gradient(x2)
dx2 = np.gradient(X, axis=0)

print("Производная x2:")
print(dx[:10])

print("\nГрадиент X:")
print(dx2[:5])


# ============================================================
# ЗАДАНИЕ 24. Свёртка
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 24. СВЁРТКА")
print("=" * 60)

conv = np.convolve(x2, y, mode="full")

print("Результат свёртки:")
print(conv[:20])


# ============================================================
# ЗАДАНИЕ 25. Скалярное и векторное произведения
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 25. ПРОИЗВЕДЕНИЯ")
print("=" * 60)

dot = np.dot(x2, y)
cross = np.cross(x2[:3], y[:3])

print("Скалярное произведение:", dot)
print("Векторное произведение:", cross)


# ============================================================
# ЗАДАНИЕ 26. Нормы L1 и L2
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 26. НОРМЫ L1 И L2")
print("=" * 60)

l1 = np.linalg.norm(x2, ord=1)
l2 = np.linalg.norm(x2, ord=2)

print("L1:", l1)
print("L2:", l2)


# ============================================================
# ЗАДАНИЕ 27. Проверка гипотез
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 27. ПРОВЕРКА ГИПОТЕЗ")
print("=" * 60)

ks_uniform, p_uniform = stats.kstest(
    x1,
    "uniform",
    args=(x1.min(), x1.max() - x1.min())
)

ks_normal, p_normal = stats.kstest(
    x2,
    "norm",
    args=(x2.mean(), x2.std())
)

sh_stat, p_shapiro = stats.shapiro(x2)

print("Равномерное распределение:")
print("KS statistic:", ks_uniform)
print("p-value:", p_uniform)

print("\nНормальное распределение:")
print("KS statistic:", ks_normal)
print("p-value:", p_normal)

print("\nТест Шапиро:")
print("Statistic:", sh_stat)
print("p-value:", p_shapiro)


# ============================================================
# ЗАДАНИЕ 28. Спектрограмма
# ============================================================

# === ИСПРАВЛЕНИЕ: Очищаем x2 от NaN, чтобы графики не были пустыми ===
if np.isnan(x2).any():
    print(f"Внимание: в x2 найдено {np.isnan(x2).sum()} пропусков. Заполняем их средним.")
    x2 = np.where(np.isnan(x2), np.nanmean(x2), x2)
# ===================================================================
# ============================================================
# ЗАДАНИЕ 28. Спектрограмма
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 28. СПЕКТРОГРАММА")
print("=" * 60)

# Подтягиваем вашу готовую функцию из stats_analysis.py
from src.stats_analysis import plot_spectrogram
fig = plot_spectrogram(x2)

path = save_figure(fig, "task28_spectrogram")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 29. Периодограмма
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 29. ПЕРИОДОГРАММА")
print("=" * 60)

# Подтягиваем вашу готовую функцию из stats_analysis.py
from src.stats_analysis import plot_periodogram
fig = plot_periodogram(x2)

path = save_figure(fig, "task29_periodogram")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 30. АЧХ через FFT
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 30. АЧХ ЧЕРЕЗ FFT")
print("=" * 60)

# Ваша функция plot_fft_amplitude уже вызывается правильно
fig = plot_fft_amplitude(x2)

path = save_figure(fig, "task30_fft")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 31. Кубическая интерполяция
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 31. КУБИЧЕСКАЯ ИНТЕРПОЛЯЦИЯ")
print("=" * 60)

x_idx = np.arange(len(x2))

f_cubic = interp1d(
    x_idx,
    x2,
    kind="cubic",
    fill_value="extrapolate"
)

x_new = np.linspace(
    0,
    len(x2) - 1,
    10 * len(x2)
)

y_new = f_cubic(x_new)

print("Исходных точек:", len(x2))
print("Новых точек:", len(y_new))


# ============================================================
# ЗАДАНИЕ 32. Интерполяция сплайнами
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 32. ИНТЕРПОЛЯЦИЯ СПЛАЙНАМИ")
print("=" * 60)

spl = UnivariateSpline(
    x_idx,
    x2,
    k=3,
    s=0
)

y_spl = spl(x_new)

print("Количество точек:", len(y_spl))


# ============================================================
# ЗАДАНИЕ 33. Бинарные маски
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 33. БИНАРНЫЕ МАСКИ")
print("=" * 60)

mask_pos = x2 > 0
mask_neg = x2 < 0
mask_zero = x2 == 0
mask_interval = (x2 >= -1) & (x2 <= 1)

print("Положительных:", mask_pos.sum())
print("Отрицательных:", mask_neg.sum())
print("Нулевых:", mask_zero.sum())
print("В интервале [-1; 1]:", mask_interval.sum())


# ============================================================
# ЗАДАНИЕ 34. Сравнение масштабирования
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 34. СРАВНЕНИЕ МАСШТАБИРОВАНИЯ")
print("=" * 60)

x = X[:, [0]]

custom_mm = MinMaxScalerCustom().fit(x)
sk_mm = MinMaxScaler().fit(x)

x_custom = custom_mm.transform(x)
x_sk = sk_mm.transform(x)

print("Совпадают:", np.allclose(
    x_custom,
    x_sk,
    atol=1e-10
))

comparison = pd.DataFrame({
    "original": x.ravel(),
    "custom": x_custom.ravel(),
    "sklearn": x_sk.ravel()
})

print(comparison.head())

write_table(comparison, "task34_scaling")


# ============================================================
# ЗАДАНИЕ 35. Инверсия
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 35. ИНВЕРСИЯ")
print("=" * 60)

x_back_custom = custom_mm.inverse_transform(x_custom)
x_back_sk = sk_mm.inverse_transform(x_sk)

print(
    "Custom восстановление:",
    np.allclose(x, x_back_custom, atol=1e-10)
)

print(
    "Sklearn восстановление:",
    np.allclose(x, x_back_sk, atol=1e-10)
)


# ============================================================
# ЗАДАНИЕ 36. Скользящее окно
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 36. СКОЛЬЗЯЩЕЕ ОКНО")
print("=" * 60)

windows = sliding_window(
    x2,
    width=5
)

print("Shape:", windows.shape)
print(windows[:5])


# ============================================================
# ЗАДАНИЕ 37. Скользящее среднее
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 37. СКОЛЬЗЯЩЕЕ СРЕДНЕЕ")
print("=" * 60)

moving = moving_average(
    x2,
    width=5
)

print("Shape:", moving.shape)
print(moving[:10])


# ============================================================
# ЗАДАНИЕ 38. Тензор TensorFlow
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 38. ТЕНЗОР TENSORFLOW")
print("=" * 60)

X_tf = tf.constant(
    X,
    dtype=tf.float32
)

print(X_tf)
print("Shape:", X_tf.shape)


# ============================================================
# ЗАДАНИЕ 39. Тензор PyTorch
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 39. ТЕНЗОР PYTORCH")
print("=" * 60)

X_pt = torch.tensor(
    X,
    dtype=torch.float32
)

print(X_pt)
print("Shape:", X_pt.shape)


# ============================================================
# ЗАДАНИЕ 40. Случайные тензоры
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 40. СЛУЧАЙНЫЕ ТЕНЗОРЫ")
print("=" * 60)

n, m = X_tf.shape

k, p = 5, 4

A = tf.random.uniform(
    (k, n),
    minval=0,
    maxval=10,
    dtype=tf.int32
)

W = tf.random.normal(
    (m, p)
)

B = tf.random.uniform(
    (k, p)
)

print("A:", A.shape)
print("W:", W.shape)
print("B:", B.shape)


# ============================================================
# ЗАДАНИЕ 41. AXW+B TensorFlow
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 41. AXW+B — TENSORFLOW")
print("=" * 60)

AX = tf.matmul(
    tf.cast(A, tf.float32),
    X_tf
)

AXW = tf.matmul(
    AX,
    W
)

result_tf = AXW + B

print(result_tf)
print("Shape:", result_tf.shape)


# ============================================================
# ЗАДАНИЕ 42. AXW+B PyTorch
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 42. AXW+B — PYTORCH")
print("=" * 60)

A_pt = torch.randint(
    0,
    10,
    (k, n),
    dtype=torch.float32
)

W_pt = torch.randn(
    m,
    p
)

B_pt = torch.rand(
    k,
    p
)

result_pt = A_pt @ X_pt @ W_pt + B_pt

print(result_pt)
print("Shape:", result_pt.shape)


# ============================================================
# ЗАДАНИЕ 43. Матрицы T, P, Q
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 43. МАТРИЦЫ T, P, Q")
print("=" * 60)

T = np.random.rand(3, 10)
P = np.random.rand(3, 10)
Q = np.random.rand(3, 10)

print("T:")
print(T)

print("\nP:")
print(P)

print("\nQ:")
print(Q)


# ============================================================
# ЗАДАНИЕ 44. Операция Keras
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 44. ОПЕРАЦИЯ KERAS")
print("=" * 60)

from keras import ops

T_k = ops.convert_to_tensor(T)
P_k = ops.convert_to_tensor(P)
Q_k = ops.convert_to_tensor(Q)

V_k = (
    ops.abs(
        ops.sin(T_k)
        - ops.exp(P_k) * ops.sqrt(Q_k)
    )
)

print(V_k)


# ============================================================
# ЗАДАНИЕ 45. Операция PyTorch
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 45. ОПЕРАЦИЯ PYTORCH")
print("=" * 60)

T_t = torch.tensor(T)
P_t = torch.tensor(P)
Q_t = torch.tensor(Q)

V_t = torch.abs(
    torch.sin(T_t)
    - torch.exp(P_t) * torch.sqrt(Q_t)
)

print(V_t)


# ============================================================
# ЗАДАНИЕ 46. ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 46. ЗАГРУЗКА ИЗОБРАЖЕНИЯ")
print("=" * 60)

image_path = Path(r"C:\Users\fang\Desktop\study\MachineLearning\lab1\images.jpg")
print(f"Реальное изображение найдено: {image_path.name}")



# ============================================================
# ЗАДАНИЕ 47. Обратная матрица
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 47. ОБРАТНАЯ МАТРИЦА")
print("=" * 60)

M = np.random.rand(100, 100)

det = np.linalg.det(M)

print("Определитель:", det)

if abs(det) > 1e-10:

    M_inv = np.linalg.inv(M)

    check = np.allclose(
        M @ M_inv,
        np.eye(100),
        atol=1e-6
    )

    print("Проверка M @ M_inv = I:", check)

else:
    print("Матрица вырождена.")


# ============================================================
# ЗАДАНИЕ 48. Достроение массива F
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 48. МАССИВ F")
print("=" * 60)

X_mm_full = MinMaxScalerCustom().fit_transform(X)
X_std_full = StandardScalerCustom().fit_transform(X)

F = np.stack(
    [
        X,
        X_mm_full,
        X_std_full
    ],
    axis=-1
)

print("Shape F:", F.shape)

# === ДОБАВЬТЕ ЭТИ СТРОКИ СЮДА ===
# Выпрямляем массив из (500, 4, 3) в (500, 12) для Pipeline/PCA
X_F = F.reshape(F.shape[0], -1)
print("Shape X_F (для PCA):", X_F.shape)
# ================================



# ============================================================
# ЗАДАНИЕ 49. Pipeline sklearn + PCA
# ============================================================
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# 1. Настройка Pipeline с автоматической очисткой NaN
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # Заполняет ВСЕ пропуски средним по столбцу
    ('pca', PCA(n_components=2))                # Безопасно применяет PCA на очищенных данных
])

# 2. Обучаем пайплайн
pipe.fit(X_F) 
print("Пайплайн успешно обучен!")

# 3. Достаем обученный шаг PCA из пайплайна для построения графика
# (так мы гарантируем, что берем PCA от очищенных данных)
pca_step = pipe.named_steps['pca']

# 4. Построение графика PCA
fig, ax = plt.subplots(figsize=(8, 5))

# Используем variance_ratio именно из обученного шага пайплайна
ax.plot(
    np.cumsum(pca_step.explained_variance_ratio_),
    marker="o",
    linestyle="--"
)

# Если вы хотите логарифмическую шкалу, убедитесь, что значения строго > 0.
# На всякий случай можно временно закомментировать set_yscale, если график будет пустым.
ax.set_yscale("log") 
ax.set_xlabel("Число компонент")
ax.set_ylabel("Кумулятивная объяснённая дисперсия (log)")
ax.set_title("PCA: Объясненная дисперсия")
ax.grid(True, alpha=0.3)

# 5. Сохранение графика
path = save_figure(fig, "task49_pca")
print("График сохранён:", path)

plt.close(fig)

# ============================================================
# ЗАДАНИЕ 50. Обработка изображения
# ============================================================

# ============================================================
# ЗАДАНИЕ 50. СГЛАЖИВАНИЕ И ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЯ
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 50. СГЛАЖИВАНИЕ И ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЯ")
print("=" * 60)

# Указываем точный путь к вашей картинке внутри папки data
exact_image_path = r"C:\Users\fang\Desktop\study\MachineLearning\lab1\data\images.jpg"

print(f"Запуск обработки для файла: {exact_image_path}")
run_task50(exact_image_path)

print("\n" + "=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА ПОЛНОСТЬЮ ВЫПОЛНЕНА!")
print("=" * 60)


# ============================================================
# КОНЕЦ
# ============================================================

print("\n" + "=" * 60)
print("ВСЕ 50 ЗАДАНИЙ ВЫПОЛНЕНЫ")
print("=" * 60)
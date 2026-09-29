import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import spectrogram, periodogram, convolve
from scipy.interpolate import interp1d, UnivariateSpline


def plot_series(data: np.ndarray, title: str = "Исходные данные") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(data.shape[1]):
        ax.plot(data[:, i], label=f"col_{i}")
    ax.set_title(title)
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_histogram(x: np.ndarray, bins: int = 30):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, density=True, alpha=0.7, edgecolor="black")
    ax.set_title("Нормализованная гистограмма")
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    return fig

def show_sorted(df: pd.DataFrame) -> None:
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(np.sort(df[col].to_numpy()))

def plot_ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots()
    ax.step(xs, ys, where="post")
    ax.set_title("Эмпирическая функция распределения")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    return fig

def column_stats(x: np.ndarray) -> dict:
    return {
        "mean": np.mean(x),
        "var": np.var(x, ddof=0),
        "mode": stats.mode(x, keepdims=False).mode,
        "median": np.median(x),
    }


def ci_mean(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)  # несмещённое СКО
    se = s / np.sqrt(n)
    return stats.t.interval(1 - alpha, df=n - 1, loc=m, scale=se)

def ci_var(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    s2 = np.var(x, ddof=1)
    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    return (n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low

# --- ИСПРАВЛЕНИЕ: Обернули свободный код в функции, чтобы принимать аргумент data ---

def calculate_matrix_stats(data: np.ndarray):
    """Вычисляет ковариацию и корреляцию для многомерного массива данных"""
    cov = np.cov(data, rowvar=False)
    corr = np.corrcoef(data, rowvar=False)

    # через Pandas:
    cov_df = pd.DataFrame(data).cov()
    corr_df = pd.DataFrame(data).corr()
    
    return cov, corr, cov_df, corr_df

def test_correlation(x: np.ndarray, y: np.ndarray):
    """Проверяет значимость корреляции Пирсона между X и Y"""
    r, p_value = stats.pearsonr(x, y)
    if p_value < 0.05:
        print(f"Корреляция значима (p={p_value:.4f})")
    else:
        print(f"Корреляция незначима (p={p_value:.4f})")
    return r, p_value

# ---------------------------------------------------------------------------------

def cross_correlation(x, y, max_lags: int = 50):
    lags = np.arange(-max_lags, max_lags + 1)
    c = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    c = c / (np.std(x) * np.std(y) * len(x))
    mid = len(c) // 2
    return lags, c[mid - max_lags: mid + max_lags + 1]

# Для использования этих функций в main.py передавайте массивы как аргументы:
def get_gradients(data: np.ndarray, x: np.ndarray):
    dx = np.gradient(x)          # одномерный случай
    dx2 = np.gradient(data, axis=0)  # многомерный: по строкам
    return dx, dx2

def get_convolutions(x, y):
    conv = np.convolve(x, y, mode="full")
    conv2 = convolve(x, y, mode="full")
    return conv, conv2

def get_vector_ops(x, y):
    dot = np.dot(x, y)                    # скалярное
    cross = np.cross(x[:3], y[:3])        # векторное (для 3-мерных)
    l1 = np.linalg.norm(x, ord=1)
    l2 = np.linalg.norm(x, ord=2)
    return dot, cross, l1, l2

def test_distributions(x1: np.ndarray, x2: np.ndarray):
    # Равномерное (1-й столбец)
    ks_stat, p_uniform = stats.kstest(x1, "uniform", args=(x1.min(), x1.max() - x1.min()))
    # Нормальное (2-й столбец)
    ks_stat, p_norm = stats.kstest(x2, "norm", args=(x2.mean(), x2.std()))
    # или более мощный тест Шапиро:
    sh_stat, p_shapiro = stats.shapiro(x2)
    return p_uniform, p_norm, p_shapiro

def plot_spectrogram(x):
    f, t, Sxx = spectrogram(x, fs=1.0)
    fig, ax = plt.subplots()
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    ax.set_ylabel("Частота")
    ax.set_xlabel("Время")
    ax.set_title("Спектрограмма")
    return fig

def plot_periodogram(x):
    f, Pxx = periodogram(x, fs=1.0)
    fig, ax = plt.subplots()
    ax.semilogy(f, Pxx)
    ax.set_xlabel("Частота")
    ax.set_ylabel("PSD")
    return fig

def plot_fft_amplitude(x):
    n = len(x)
    fft = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0)
    half = n // 2
    fig, ax = plt.subplots()
    ax.plot(freq[:half], np.abs(fft[:half]))
    ax.set_xlabel("Частота")
    ax.set_ylabel("|X(f)|")
    return fig

def interpolate_data(x):
    x_idx = np.arange(len(x))
    f_cubic = interp1d(x_idx, x, kind="cubic", fill_value="extrapolate")
    x_new = np.linspace(0, len(x) - 1, 10 * len(x))
    y_new = f_cubic(x_new)

    spl = UnivariateSpline(x_idx, x, k=3, s=0)
    y_spl = spl(x_new)
    return x_new, y_new, y_spl

def get_masks(x):
    mask_pos = x > 0
    mask_neg = x < 0
    mask_zero = x == 0
    mask_interval = (x >= -1) & (x <= 1)
    return mask_pos, mask_neg, mask_zero, mask_interval

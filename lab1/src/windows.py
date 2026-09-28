def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    """Возвращает матрицу окон формы (n - width + 1, width)."""
    if width <= 0 or width > len(x):
        raise ValueError("Некорректная ширина окна")
    n = len(x)
    idx = np.arange(width)[None, :] + np.arange(n - width + 1)[:, None]
    return x[idx]

def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    w = sliding_window(x, width)
    return w.mean(axis=1)

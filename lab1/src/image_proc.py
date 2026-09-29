import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import rotate, gaussian_filter, convolve
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from src.config import save_figure


# ----------------------------------------------------------------------
# 1. Разделение цветовых каналов
# ----------------------------------------------------------------------
def split_channels(img: np.ndarray) -> dict:
    """
    Разделяет цветное изображение на отдельные каналы R, G, B.
    Возвращает словарь с массивами и сохраняет их как отдельные PNG.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается цветное изображение (H, W, 3)")

    r = img[..., 0].copy()
    g = img[..., 1].copy()
    b = img[..., 2].copy()

    # Визуализация каналов
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img);   axes[0].set_title("Исходное")
    axes[1].imshow(r, cmap="Reds");    axes[1].set_title("Красный канал")
    axes[2].imshow(g, cmap="Greens");  axes[2].set_title("Зелёный канал")
    axes[3].imshow(b, cmap="Blues");   axes[3].set_title("Синий канал")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_channels")
    plt.close(fig)

    return {"r": r, "g": g, "b": b}


# ----------------------------------------------------------------------
# 2. Поворот изображения на 40 градусов
# ----------------------------------------------------------------------
def rotate_image(img: np.ndarray, angle: float = 40.0) -> np.ndarray:
    """
    Поворачивает изображение на заданный угол.
    Используется scipy.ndimage.rotate с бикубической интерполяцией.
    """
    rotated = rotate(
        img.astype(np.float32),
        angle=angle,
        reshape=True,             # расширяем холст, чтобы изображение не обрезалось
        order=3,                  # бикубическая интерполяция
        mode="constant",
        cval=0,
    )
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);      axes[0].set_title("Исходное")
    axes[1].imshow(rotated);  axes[1].set_title("Поворот на 40°")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_rotated")
    plt.close(fig)

    return rotated


# ----------------------------------------------------------------------
# 3. Преобразование в чёрно-белое изображение
# ----------------------------------------------------------------------
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Перевод в ч/б через scipy/skimage (взвешенная яркость по Rec. 601/709).
    Возвращает массив uint8 формы (H, W).
    """
    gray = rgb2gray(img)                 # значения в [0, 1]
    gray_u8 = (gray * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);                axes[0].set_title("Исходное")
    axes[1].imshow(gray_u8, cmap="gray"); axes[1].set_title("Ч/б")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_gray")
    plt.close(fig)

    return gray_u8


# ----------------------------------------------------------------------
# 4. Гистограмма чёрно-белого изображения
# ----------------------------------------------------------------------
def gray_histogram(gray: np.ndarray, bins: int = 256) -> dict:
    """
    Строит гистограмму яркостей ч/б изображения.
    Возвращает значения частот и границы бинов.
    """
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], hist, width=1, color="black", align="edge")
    ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Частота")
    ax.set_xlim(0, 255)
    save_figure(fig, "task50_gray_hist")
    plt.close(fig)

    return {"hist": hist, "bin_edges": bin_edges}


# ----------------------------------------------------------------------
# 5. Разбиение изображения на 3 части по гистограмме
# ----------------------------------------------------------------------
def split_by_histogram(gray: np.ndarray, hist_info: dict) -> tuple:
    """
    Делит изображение на 3 части по порогам, полученным из гистограммы.
    Пороги выбираются так, чтобы суммарная частота в каждой части
    была примерно одинаковой (метод равных квантилей по гистограмме).
    """
    hist = hist_info["hist"]
    edges = hist_info["bin_edges"]

    # кумулятивная сумма частот
    cum = np.cumsum(hist)
    total = cum[-1]

    # ищем границы, где накопленная частота достигает 33 % и 66 %
    t1 = edges[np.searchsorted(cum, total * 0.33)]
    t2 = edges[np.searchsorted(cum, total * 0.66)]

    # три бинарно-взвешенных изображения
    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(part1, cmap="gray"); axes[0].set_title(f"Тёмные (≤ {t1})")
    axes[1].imshow(part2, cmap="gray"); axes[1].set_title(f"Средние ({t1}–{t2})")
    axes[2].imshow(part3, cmap="gray"); axes[2].set_title(f"Светлые (> {t2})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_split3")
    plt.close(fig)

    return part1, part2, part3


# ----------------------------------------------------------------------
# 6. Чёрная круговая рамка
# ----------------------------------------------------------------------
def add_circular_frame(img: np.ndarray) -> np.ndarray:
    """
    Алгоритмически накладывает чёрную круговую рамку:
    всё, что вне круга радиуса R с центром в центре изображения, зануляется.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    R = min(h, w) / 2.0 - 2      # небольшой отступ, чтобы рамка была видна

    # координатная сетка
    Y, X = np.ogrid[:h, :w]
    mask_outside = (X - cx) ** 2 + (Y - cy) ** 2 > R ** 2

    framed = img.copy()
    if framed.ndim == 2:
        framed[mask_outside] = 0
    else:
        framed[mask_outside, :] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);    axes[0].set_title("Исходное")
    axes[1].imshow(framed); axes[1].set_title("В круговой рамке")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_frame")
    plt.close(fig)

    return framed


# ----------------------------------------------------------------------
# 7. Добавление случайного шума
# ----------------------------------------------------------------------
def add_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Добавляет гауссовский шум с нулевым средним и заданным СКО.
    Результат обрезается до диапазона [0, 255].
    """
    noise = np.random.normal(0.0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);   axes[0].set_title("Исходное")
    axes[1].imshow(noisy); axes[1].set_title(f"Шум (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_noisy")
    plt.close(fig)

    return noisy


# ----------------------------------------------------------------------
# 8. Гауссовское размытие
# ----------------------------------------------------------------------
def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Размытие изображения гауссовским фильтром.
    Для цветного — применяется по каждому каналу.
    """
    if img.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
             for c in range(img.shape[2])],
            axis=-1,
        )
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);      axes[0].set_title("Исходное")
    axes[1].imshow(blurred);  axes[1].set_title(f"Размытие (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_blurred")
    plt.close(fig)

    return blurred


# ----------------------------------------------------------------------
# Главная функция запуска задачи (ЭТОГО НЕ ХВАТАЛО ДЛЯ ИМПОРТА)
# ----------------------------------------------------------------------
def run_task50(img: np.ndarray) -> None:
    """
    Последовательно запускает все этапы обработки изображения.
    Принимает на вход исходное цветное изображение RGB (массив numpy).
    """
    print("Запуск обработки изображения (Задача 50)...")
    
    # 1. Каналы
    channels = split_channels(img)
    
    # 2. Поворот
    rotated = rotate_image(img, angle=40.0)
    
    # 3. Ч/б
    gray = to_grayscale(img)
    
    # 4. Гистограмма
    hist_info = gray_histogram(gray)
    
    # 5. Разделение по гистограмме
    p1, p2, p3 = split_by_histogram(gray, hist_info)
    
    # 6. Рамка
    framed = add_circular_frame(img)
    
    # 7. Шум
    noisy = add_noise(img, sigma=25.0)
    
    # 8. Размытие
    blurred = gaussian_blur(img, sigma=2.0)
    
    print("Обработка успешно завершена! Графики сохранены.")



# ----------------------------------------------------------------------
# 9. Sharpening (unsharp masking)
# ----------------------------------------------------------------------
def sharpen_image(img: np.ndarray,
                  sigma: float = 2.0,
                  amount: float = 1.5) -> np.ndarray:
    """
    Операция sharpening через unsharp masking:
        sharpened = original + amount * (original - blurred)
    Значение amount > 0 усиливает контуры.
    """
    blurred = gaussian_filter(img.astype(np.float32),
                              sigma=(sigma, sigma, 0) if img.ndim == 3 else sigma)
    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);       axes[0].set_title("Исходное")
    axes[1].imshow(sharpened); axes[1].set_title(f"Sharpening (amount={amount})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_sharpened")
    plt.close(fig)

    return sharpened


# ----------------------------------------------------------------------
# 10. PCA к изображению
# ----------------------------------------------------------------------
def apply_pca_to_image(gray_img: np.ndarray, n_components: int = 32):
    """
    Применяет PCA к двумерному ч/б изображению.
    Каждая строка изображения выступает как отдельный объект (sample).
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from src.config import save_figure

    # Гарантируем, что матрица строго двумерная (высота, ширина)
    if gray_img.ndim > 2:
        gray_img = gray_img[:, :, 0]
    elif gray_img.ndim == 1:
        # Если картинка пришла как 1D вектор, возвращаем её форму назад в 2D matrix
        side = int(np.sqrt(gray_img.size))
        gray_img = gray_img.reshape(side, side)

    h, w = gray_img.shape
    
    # Защита: число компонент не может превышать размеры матрицы (высоту или ширину)
    # и должно быть строго не меньше 1
    actual_components = max(1, min(n_components, h, w, 4)) 

    # Создаем и обучаем PCA непосредственно на строках-пикселях матрицы
    pca = PCA(n_components=actual_components)
    flat = gray_img.astype(np.float32)
    
    transformed = pca.fit_transform(flat)
    projected = pca.inverse_transform(transformed)
    
    reconstructed = np.clip(projected, 0, 255).astype(np.uint8)

    # Визуализация результатов PCA
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gray_img, cmap="gray")
    axes[0].set_title("Исходное Ч/Б")
    axes[0].axis("off")
    
    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].set_title(f"PCA (компонент: {actual_components})")
    axes[1].axis("off")
        
    save_figure(fig, "task50_pca_result")
    plt.close(fig)

    return reconstructed


# ----------------------------------------------------------------------
# Общая функция запуска задания 50
# ----------------------------------------------------------------------
def run_task50(image_input) -> dict:
    """
    Последовательно выполняет все 10 подзаданий.
    Принимает на вход путь к файлу (str / Path) или готовый массив numpy (np.ndarray).
    Возвращает словарь с результатами.
    """
    # ИСПРАВЛЕНИЕ: Проверяем, что пришло на вход — матрица или путь к файлу
    if isinstance(image_input, np.ndarray):
        img = image_input
    else:
        img = plt.imread(image_input)
        
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    results = {}
    results["channels"] = split_channels(img)
    results["rotated"] = rotate_image(img, angle=40.0)
    results["gray"] = to_grayscale(img)
    results["hist"] = gray_histogram(results["gray"])
    results["parts"] = split_by_histogram(results["gray"], results["hist"])
    results["framed"] = add_circular_frame(img)
    results["noisy"] = add_noise(img, sigma=25.0)
    results["blurred"] = gaussian_blur(img, sigma=2.0)
    results["sharpened"] = sharpen_image(img, sigma=2.0, amount=1.5)
    # ИСПРАВЛЕНО: передаем ч/б изображение results["gray"] вместо цветного img
    results["pca"] = apply_pca_to_image(results["gray"], n_components=32)


    print("Задание 50 выполнено. Все графики сохранены.")
    return results


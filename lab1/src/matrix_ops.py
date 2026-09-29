import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
S = sparse.random(100_000, 100_000, density=1e-5, format="csr")
print("NNZ:", S.nnz)

M = np.random.rand(100, 100)
det = np.linalg.det(M)
if abs(det) > 1e-10:
    M_inv = np.linalg.inv(M)
    # проверка:
    assert np.allclose(M @ M_inv, np.eye(100), atol=1e-6)
else:
    print("Матрица вырождена")

# X — исходные данные (n, m)
X_mm = MinMaxScalerCustom().fit_transform(X)
X_std = StandardScalerCustom().fit_transform(X)

# Собираем многомерный массив F формы (n, m, 3):
F = np.stack([X, X_mm, X_std], axis=-1)
print(F.shape)  # (n, m, 3)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scalers = {"minmax": MinMaxScaler(), "std": StandardScaler()}
results = {}

for name, scaler in scalers.items():
    pipe = Pipeline([("scaler", scaler), ("pca", PCA())])
    pipe.fit(F.reshape(len(F), -1))
    ll = pipe.named_steps["pca"].score_samples(
        pipe.named_steps["scaler"].transform(F.reshape(len(F), -1))
    ).sum()
    results[name] = ll
    print(f"{name}: log-likelihood = {ll:.4f}")

# лучший:
best = max(results, key=results.get)
print("Лучший вариант:", best)
#Логарифмический график правдоподобия:
fig, ax = plt.subplots()
pca = PCA().fit(F.reshape(len(F), -1))
ax.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
ax.set_yscale("log")
ax.set_xlabel("Число компонент")
ax.set_ylabel("Кумулятивная объяснённая дисперсия (log)")

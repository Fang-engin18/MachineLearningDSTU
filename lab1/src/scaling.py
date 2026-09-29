from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

x = data[:, [0]]  # первый столбец
custom_mm = MinMaxScalerCustom().fit(x)
sk_mm = MinMaxScaler().fit(x)

x_custom = custom_mm.transform(x)
x_sk = sk_mm.transform(x)

# Сравнение:
assert np.allclose(x_custom, x_sk, atol=1e-10)

x_back_custom = custom_mm.inverse_transform(x_custom)
x_back_sk = sk_mm.inverse_transform(x_sk)
assert np.allclose(x, x_back_custom, atol=1e-10)
assert np.allclose(x, x_back_sk, atol=1e-10)

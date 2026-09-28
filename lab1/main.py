from pathlib import Path

from src.config import set_seeds
from src.io_utils import read_table, write_table, save_figure
from src.data_loader import check_data_quality, show_table, extract_columns, cast_types
from src.preprocess import handle_missing, interactive_menu, to_numpy, MinMaxScalerCustom, StandardScalerCustom, split_train_test, split_k_folds

import matplotlib.pyplot as plt

set_seeds()
DATA_PATH = Path(r"C:\Users\fang\Desktop\study\MachineLearning\lab1\data\dataset_var1.csv")

#task 1
from src.io_utils import read_table
data = read_table(DATA_PATH)
print(data)

#task 2
# from src.io_utils import write_table
# paths = write_table(data, "dataset")
# print(paths)

#task 3
# import matplotlib.pyplot as plt
# from src.io_utils import save_figure

# # Создаём график
# fig, ax = plt.subplots()

# ax.plot(data["x1"], data["y"])
# ax.set_xlabel("x1")
# ax.set_ylabel("y")
# ax.set_title("Зависимость y от x1")

# # Сохраняем график
# path = save_figure(fig, "x1_vs_y")
# print(path)
# plt.show()

#task 4
# from src.data_loader import check_data_quality
# report = check_data_quality(data)

# print("Пропущенные значения:")
# print(report["missing"])

# print("\nТипы данных:")
# print(report["dtypes"])

# print("\nНекорректные числовые значения:")
# print(report["non_numeric"])

#task 5
#show_table(data, n=5)

#task 6
#print(extract_columns(data, ["x1", "x2", "y"]))

#task7
#print(cast_types(data, { "x2": "int64", "y": "float64"}))

#task 8

#task 9

#task 10

#task 11

#task 12

#task 13


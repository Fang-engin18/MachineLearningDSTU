from src.io_utils import read_table
#task 1
data = read_table(r"C:\Users\fang\Desktop\study\MachineLearning\lab1\data\dataset_var1.csv")
print(data)

#task 2
from src.io_utils import write_table
paths = write_table(data, "dataset")
print(paths)
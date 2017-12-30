import numpy as np
from trlib.experiments.visualization import plot_curves

x = [np.arange(1,10,0.5) for _ in range(7)]
y_mean = [np.log(x[0]) + i for i in range(7)]
y_std = [np.random.randn(x[0].shape[0]) / 3 for _ in range(7)]
names = ["Log " + str(i) for i in range(7)]

plot_curves(x, y_mean, y_std_data = y_std, title = "", x_label = "Episodes", y_label = "Performance", names = names, file_name = "figure")